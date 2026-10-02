"""Compressed HDF5 storage for travel-time maps."""

from __future__ import annotations

import os
import tempfile
from argparse import ArgumentParser
from collections.abc import Mapping
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import RectSphereBivariateSpline

from NuRadioReco.detector.RNO_G.rnog_detector import Detector


def save_map(path: str | Path, tt_map: dict) -> None:
    """Save a travel-time map using gzip-compressed HDF5 datasets.

    Parameters
    ----------
    path : str | Path
        The file path to save the travel-time map.
    tt_map : dict
        The travel-time map to save, containing 'zeniths', 'azimuths', and channel-pair arrays.
    """
    destination = Path(path)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        with h5py.File(temporary_path, "w") as output:
            output["zeniths"] = tt_map["zeniths"]
            output["azimuths"] = tt_map["azimuths"]
            maps = output.create_group("maps")
            for pair, values in tt_map.items():
                if isinstance(pair, tuple):
                    channel_a, channel_b = pair
                    maps.create_dataset(f"{channel_a}_{channel_b}", data=values, compression="gzip")
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise




class InterpolatedMap(Mapping):
    """
    Interpolated travel-time map that lazily computes interpolations for channel-pair arrays.
    Provides access to 'zeniths', 'azimuths', and channel-pair arrays as a read-only mapping.
    """
    def __init__(self, source_zeniths, source_azimuths, target_zeniths, target_azimuths,
                 pairs, dtype=np.float32, path=None, group=None, raw_map=None):
        """
        Initialize an interpolated travel-time map.

        Parameters
        ----------
        source_zeniths : np.ndarray
            Array of source zenith angles.
        source_azimuths : np.ndarray
            Array of source azimuth angles.
        target_zeniths : np.ndarray
            Array of target zenith angles for interpolation.
        target_azimuths : np.ndarray
            Array of target azimuth angles for interpolation.
        pairs : list of tuple
            List of channel pairs to interpolate.
        dtype : data-type, optional
            Data type for the interpolated arrays. Default is np.float32.
        path : str | Path, optional
            Path to the HDF5 file containing the travel-time map. Default is None.
        group : str, optional
            HDF5 group within the file containing the travel-time map. Default is None.
        raw_map : dict, optional
            In-memory dictionary containing the travel-time map. Default is None.
        """
        
        if path is None and raw_map is None:
            raise ValueError("Must provide either `path`/`group` (HDF5) or `raw_map` (in-memory dict)")
        self.path, self.group, self.raw_map, self.pairs = path, group, raw_map, pairs
        self.source_zeniths, self.source_azimuths = source_zeniths, source_azimuths
        self.target_zeniths, self.target_azimuths = target_zeniths, target_azimuths
        self.dtype = dtype
        self.cache = {}  # computed interpolations persist for the lifetime of this object

    def _build_interpolator(self, values):
        """
        Build a spherical bivariate spline interpolator for the given values.

        Parameters
        ----------
        values : np.ndarray
            Array of travel-time values corresponding to the source zenith and azimuth angles.

        Returns
        -------
        RectSphereBivariateSpline
            Spline interpolator for the given values.
        """
        theta = self.source_zeniths
        phi = self.source_azimuths

        # nudge off the strict 0 / 2pi boundary if azimuths hit it dead-on
        eps = 1e-6
        if phi[0] <= 0:
            phi = phi.copy()
            phi[0] = eps
        if phi[-1] >= 2 * np.pi:
            phi[-1] = 2 * np.pi - eps

        return RectSphereBivariateSpline(theta, phi, values, s=1e-5)

    def _get_raw_values(self, key):
        if self.raw_map is not None:
            return self.raw_map[key]
        with h5py.File(self.path, "r") as source:
            return source[self.group][f"{key[0]}_{key[1]}"][:]

    def __getitem__(self, key):
        if key in ("zeniths", "azimuths"):
            return self.target_zeniths if key == "zeniths" else self.target_azimuths
        if key not in self.cache:
            spline = self._build_interpolator(self._get_raw_values(key))
            result = spline(self.target_zeniths, self.target_azimuths, grid=True)
            self.cache[key] = result.astype(self.dtype, copy=False)
        return self.cache[key]

    def __iter__(self):
        return iter(("zeniths", "azimuths", *self.pairs))

    def __len__(self):
        return 2 + len(self.pairs)


def load_map(path: str | Path, zeniths=90 * 10, azimuths=360 * 10) -> dict:
    """Load a map whose channel-pair arrays interpolate lazily on access.

    Parameters
    ----------
    path : str | Path
        The file path to the HDF5 travel-time map.
    zeniths : int | np.ndarray, optional
        Number of steps or array of target zenith angles for interpolation. Default is 90 * 10.
    azimuths : int | np.ndarray, optional
        Number of steps or array of target azimuth angles for interpolation. Default is 360 * 10.

    Returns
    -------
    InterpolatedMap
        Lazily-interpolating travel-time map object.
    """
    with h5py.File(path, "r") as source:
        source_zeniths = source["zeniths"][:]
        source_azimuths = source["azimuths"][:]
        zeniths = np.linspace(0.001, np.pi / 2, zeniths) if isinstance(zeniths, int) else np.asarray(zeniths)
        azimuths = np.linspace(0, 2 * np.pi, azimuths) if isinstance(azimuths, int) else np.asarray(azimuths)
        maps = source["maps"] if "maps" in source else source["time_differences"]
        pairs = [tuple(map(int, name.split("_"))) for name in maps]
        group = maps.name

    return InterpolatedMap(source_zeniths, source_azimuths, zeniths, azimuths, pairs, path=path, group=group)


def map_from_dict(tt_map: dict, zeniths=90 * 10, azimuths=360 * 10) -> InterpolatedMap:
    """Build a lazily-interpolating map directly from an in-memory tt_map dict, no HDF5 round-trip needed.

    Parameters
    ----------
    tt_map : dict
        Dictionary containing the travel-time map with 'zeniths', 'azimuths', and channel-pair arrays.
    zeniths : int | np.ndarray, optional
        Number of steps or array of target zenith angles for interpolation. Default is 90 * 10.
    azimuths : int | np.ndarray, optional
        Number of steps or array of target azimuth angles for interpolation. Default is 360 * 10.

    Returns
    -------
    InterpolatedMap
        Lazily-interpolating travel-time map object.
    """
    source_zeniths = np.asarray(tt_map["zeniths"])
    source_azimuths = np.asarray(tt_map["azimuths"])
    zeniths = np.linspace(0.001, np.pi / 2, zeniths) if isinstance(zeniths, int) else np.asarray(zeniths)
    azimuths = np.linspace(0, 2 * np.pi, azimuths) if isinstance(azimuths, int) else np.asarray(azimuths)
    pairs = [key for key in tt_map if isinstance(key, tuple)]
    raw_map = {pair: tt_map[pair] for pair in pairs}

    return InterpolatedMap(source_zeniths, source_azimuths, zeniths, azimuths, pairs, raw_map=raw_map)


def load_map_non_interp(path: str | Path) -> dict:
    """Load the stored travel-time map arrays without interpolation."""
    with h5py.File(path, "r") as source:
        tt_map = {
            "zeniths": source["zeniths"][:],
            "azimuths": source["azimuths"][:],
        }
        maps = source["maps"] if "maps" in source else source["time_differences"]
        for name, values in maps.items():
            tt_map[tuple(map(int, name.split("_")))] = values[:]
    return tt_map




if __name__ == "__main__":
    
    ## Build and save the travel-time map for the specified station and calibration.
    
    from reconstruction import _build_travel_time_map, _load_ice_model

    
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--station", type=int, required=True, help="Station number")
    arg_parser.add_argument("--calibration", type=str, default=None, help="Path to calibrated detector geometry")
    arg_parser.add_argument("--map", type=str, required=True, help="Prefix for the generated travel-time HDF5 file")
    arg_parser.add_argument("--compression-level", type=int, default=4, help="gzip compression level (0-9)")
    args = arg_parser.parse_args()

    station_ = args.station
    det_file = args.calibration
    
    if station_ == 11 and det_file !="null":
        det_file = "/cvmfs/rnog.opensciencegrid.org/calibration/latest/station_11.json.xz"
    if det_file == "null":
        det_file = None

    channels = [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23]  # All channels TODO: account for new stations' channel mappin
    map_path = args.map

    if det_file is not None:
        # Load calibrated RNO-G detector
        detector = Detector(detector_file=det_file, select_stations=station_)
        if station_ == 14:
            detector.update(pd.to_datetime("2025-05-01T00:00:00"))
        elif station_ == 22:
            detector.update(pd.to_datetime("2023-06-01T00:00:00"))
        else:
            detector.update(pd.to_datetime("2024-06-01T00:00:00"))

            # load calibrated ice model
        ice_model = _load_ice_model(det_file)
        map_path += "_calib.h5"

    else:
        detector = Detector(select_stations=station_)
        if station_ == 14:
            detector.update(pd.to_datetime("2025-05-01T00:00:00"))
        elif station_ == 22:
            detector.update(pd.to_datetime("2023-06-01T00:00:00"))
        else:
            detector.update(pd.to_datetime("2024-06-01T00:00:00"))

        # load standard ice model
        from NuRadioMC.utilities import medium
        ice_model = medium.greenland_3exp_layered()
        map_path += "_null.h5"

    print(f"Building travel-time map for station {station_} with calibration: {args.calibration}")

    prep_map = _build_travel_time_map(detector, ice_model, station_, channels,
                                      zeniths=np.linspace(0.001, np.pi / 2, 90*1),
                                      azimuths=np.linspace(0, 2 * np.pi, 360*1),
                                      #use_multiprocessing=True
                                      )

    save_map(map_path, prep_map)

    file_size = os.path.getsize(map_path) / (1024 * 1024)  # Size in MB
    print(f"Travel-time map saved to {map_path} (Size: {file_size:.2f} MB)")
