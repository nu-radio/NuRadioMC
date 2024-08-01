from NuRadioReco.modules.base.module import register_run
from NuRadioReco.detector.detector_base import DetectorBase
from NuRadioReco.framework import event, station, sim_station, electric_field
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.modules.io.coreas import coreas
import numpy as np
from radiotools.coordinatesystems import cstrafo
from NuRadioReco.utilities import units, bandpass_filter, fft
import cr_pulse_interpolator.signal_interpolation_fourier as traceinterp  # traces
import cr_pulse_interpolator.interpolation_fourier as fluinterp  # fluence
from typing import Optional
import h5py
import logging
from collections import defaultdict

logger = logging.getLogger("NuRadioReco.readCoREASInterpolator")

warning_printed_coreas_py = False


class readCoREASInterpolator:
    def __init__(self,
                 logger_level=logging.WARNING,
                 lowfreq: float = 30.0 * units.MHz,
                 highfreq: float = 500.0 * units.MHz,
                 kind: str = "trace",
                 interpolator_kwargs: dict = None,
                 ):
        logger.setLevel(logger_level)

        self.lowfreq = lowfreq
        self.highfreq = highfreq
        if interpolator_kwargs:
            self.interpolator_kwargs = interpolator_kwargs
        else:
            self.interpolator_kwargs = {}

        self.cs = None
        self.corsika = None
        self.coreas_shower = None
        self.signal_interpolator = None
        assert kind in ["trace", "fluence"]
        self.kind = kind

    def begin(self, filename):
        """
        Initializes the interpolator from a star-shape CoREAS simulation file
        """
        self.corsika = h5py.File(filename)
        self.coreas_shower = coreas.make_sim_shower(self.corsika)
        self.coreas_shower.set_parameter(
            shp.core, [0., 0., self.coreas_shower[shp.observation_level]])
        self.cs = cstrafo(*coreas.get_angles(self.corsika))
        self.sampling_period = self.corsika["CoREAS"].attrs["TimeResolution"] * units.s

        self._set_showerplane_positions_and_signals()

        if self.kind == "trace":
            self.signal_interpolator = traceinterp.interp2d_signal(
                self.starshape_showerplane[..., 0],
                self.starshape_showerplane[..., 1],
                self.signals[..., 1:],
                self.trace_start / units.s,
                lowfreq=self.lowfreq / units.MHz,  # THIS OPTION IS UNUSED  IN traceinterp.interp2d_fourier.__init__
                highfreq=self.highfreq / units.MHz,  # THIS OPTION IS UNUSED  IN traceinterp.interp2d_fourier.__init__
                sampling_period=self.sampling_period / units.s,
                **self.interpolator_kwargs
            )
        elif self.kind == "fluence":
            self.signal_interpolator = fluinterp.interp2d_fourier(
                self.starshape_showerplane[..., 0],
                self.starshape_showerplane[..., 1],
                self.signals,
                fill_value="extrapolate"  # THIS OPTION IS UNUSED  IN fluinterp.interp2d_fourier.__init__
            )

    def _set_showerplane_positions_and_signals(self):
        """
        Reads the positions and signals from the star-shape CoREAS simulation,
        then converts them to the correct coordinate system and units.
        """
        assert self.corsika is not None and self.cs is not None

        starpos = []
        signals = []
        trace_start = []

        for observer in self.corsika['CoREAS/observers'].values():
            position_coreas = observer.attrs['position']
            position_nr = np.array(
                [-position_coreas[1], position_coreas[0], 0]) * units.cm
            starpos.append(position_nr)

            nrr_observer = coreas.observer_to_si_geomagnetic(observer)
            trace_start.append(nrr_observer[0,0])
            signal = self.cs.transform_from_magnetic_to_geographic(
                nrr_observer[:, 1:].T)
            signal = self.cs.transform_from_ground_to_onsky(signal)
            if self.kind == "fluence":
                filter_response = bandpass_filter.get_filter_response(
                    np.fft.rfftfreq(signal.shape[1], d=self.sampling_period / units.s) * units.Hz,
                    [self.lowfreq, self.highfreq], 'rectangular', order=0
                )
                sampling_rate = 1 / self.sampling_period
                signal_fft = fft.time2freq(signal, sampling_rate)
                signal_fft *= filter_response  # filter the signal
                signal = fft.freq2time(signal_fft, sampling_rate, signal.shape[1])
                signal = np.sum(np.square(signal[1:]))  # get the fluence of vxB and vxvB only
            signals.append(signal.T)

            logger.debug(
                f"parsed starshape detector at position {position_nr}"
            )

        starpos = np.array(starpos)
        signals = np.array(signals)
        trace_start = np.array(trace_start)
        starpos_vBvvB = self.cs.transform_from_magnetic_to_geographic(
            starpos.T)
        starpos_vBvvB = self.cs.transform_to_vxB_vxvxB(starpos_vBvvB.T)

        dd = (starpos_vBvvB[:, 0] ** 2 + starpos_vBvvB[:, 1] ** 2) ** 0.5
        logger.info(f"assumed star shape from: {-dd.max()} - {dd.max()}")

        self.starshape_showerplane = starpos_vBvvB
        self.signals = signals
        self.trace_start = trace_start

    @register_run()
    def run(self, det: DetectorBase,
            station_ids: Optional[list] = None,
            requested_channel_ids: Optional[list] = None,
            core_xy: np.ndarray = np.zeros(2, dtype=float),
            # debug: bool = False
            ) -> event.Event:

        evt = event.Event(0, 0)
        core_shift = np.append(core_xy, [0.])
        if station_ids is None:
            station_ids = det.get_station_ids()

        for station_id in station_ids:

            chan_id_per_groupid = select_channels_per_station(
                det=det,
                station_id=station_id,
                requested_channel_ids=requested_channel_ids)

            if len(chan_id_per_groupid.keys()) == 0:
                logger.info(
                    f"station {station_id} did not contain any requested channel_ids")
                continue

            station_position_ground = det.get_absolute_position(station_id)

            chan_positions_ground_per_groupid = {}
            chan_positions_ground_per_groupid_shifted = {}
            for group_id, assoc_channel_ids in chan_id_per_groupid.items():
                chan_positions_ground_per_groupid[group_id] = station_position_ground + det.get_relative_position(
                    station_id, assoc_channel_ids[0])
                chan_positions_ground_per_groupid_shifted[group_id] = chan_positions_ground_per_groupid[group_id] - core_shift

            chan_positions_ground_shifted = np.vstack(
                [pos for pos in chan_positions_ground_per_groupid_shifted.values()])

            chan_positions_vxB_shifted = self.cs.transform_to_vxB_vxvxB(
                chan_positions_ground_shifted, self.coreas_shower[shp.core])
            chan_positions_vxB_per_groupid_shifted = {}
            for group_id, pos in zip(chan_id_per_groupid.keys(), chan_positions_vxB_shifted):
                chan_positions_vxB_per_groupid_shifted[group_id] = pos

            # flattened_positions = np.vstack(
            #     [pos for pos in chan_positions_vxB_per_groupid.values()])
            contained = position_contained_in_starshape(chan_positions_vxB_shifted, self.starshape_showerplane)
            if np.any(~contained):
                logger.warning(
                    "Channel positions are not all contained in the starshape! Will extrapolate."
                )

            stat = station.Station(station_id)
            if self.kind == "trace":
                efields = {}
                trace_start = {}
                for idx, (group_id, position) in enumerate(chan_positions_vxB_per_groupid_shifted.items()):
                    if not contained[idx]:
                        timeseries = np.zeros((3, self.signals.shape[-2]))
                        trace_start_time = self.signal_interpolator.interpolators_arrival_times(*position[:-1])
                    else:
                        timeseries, trace_start_time, _, _= self.signal_interpolator(
                            *position[:-1],
                            lowfreq=self.lowfreq / units.MHz,
                            highfreq=self.highfreq / units.MHz,
                            account_for_timing = True,
                            pulse_centered = True,
                            full_output = True
                            )
                        timeseries = timeseries.T
                        timeseries = np.vstack([np.zeros_like(timeseries[0]), *timeseries]) # add r polarization back to trace, as zeroes
                    efields[group_id] = timeseries
                    trace_start[group_id] = trace_start_time * units.s


                sim_stat = make_sim_station(
                    station_id, self.corsika, chan_id_per_groupid, chan_positions_ground_per_groupid, efields=efields, trace_start=trace_start
                )
                stat.set_sim_station(sim_stat)

            elif self.kind == "fluence":
                fluences = {}
                for group_id, position in chan_positions_vxB_per_groupid_shifted.items():
                    fluences[group_id] = self.signal_interpolator(*position[:-1])
                sim_stat = make_sim_station(
                    station_id, self.corsika, chan_id_per_groupid, chan_positions_ground_per_groupid, fluences=fluences)
                stat.set_sim_station(sim_stat)

            evt.set_station(stat)

        self.coreas_shower.set_parameter(
            shp.core, np.array([0., 0., self.coreas_shower[shp.observation_level]]) + core_shift)
        evt.add_sim_shower(self.coreas_shower)
        return evt

    def end(self):
        self.corsika.close()
        self.corsika = None

    def __del__(self):
        if self.corsika:
            self.corsika.close()


def make_sim_station(station_id,
                     corsika: h5py.File,
                     channel_ids_per_groupid: dict,
                     positions: dict,
                     efields: Optional[dict] = None,
                     fluences: Optional[dict] = None,
                     trace_start: Optional[dict] = None,
                     weight=None):
    """
    creates an NuRadioReco sim station from the (interpolated) observer object of the coreas hdf5 file

    Either `efields` or `fluences` needs to be provided!

    Parameters
    ----------
    station_id : station id
        the id of the station to create
    corsika : hdf5 file object
        the open hdf5 file object of the corsika hdf5 file
    channel_ids_per_groupid : dict
        A dictionary containing the channel ids grouped per channel group id (which are the keys)
    positions : dict
        The positions of the antenna, supplied as a dictionary with the channel group id as keys.
    efields : dict, optional
        A dictionary containing the electric fields associated to an antenna.
    fluences : dict, optional
        A dictionary containing the fluence of an antenna.
    weight : weight of individual station
        weight corresponds to area covered by station

    Returns
    -------
    sim_station: sim station
        simulated station object
    """
    assert (efields is not None) or (fluences is not None)

    zenith, azimuth, magnetic_field_vector = coreas.get_angles(corsika)
    cs = cstrafo(zenith, azimuth, magnetic_field_vector=magnetic_field_vector)

    # prepend trace with zeros to not have the pulse directly at the start
    sim_station_ = sim_station.SimStation(station_id)
    for group_id, assoc_channel_ids in channel_ids_per_groupid.items():
        # expect time, Ex, Ey, Ez (ground coordinates)
        electric_field_ = electric_field.ElectricField(
            assoc_channel_ids, position=positions[group_id])

        if fluences is not None:
            electric_field_.set_parameter(efp.signal_energy_fluence, fluences[group_id])

        if efields is not None:
            efield = np.copy(efields[group_id])

            sampling_rate = 1. / (corsika['CoREAS'].attrs['TimeResolution'] * units.second)
            electric_field_.set_trace(efield, sampling_rate)
            if trace_start is not None:
                electric_field_.set_trace_start_time(trace_start[group_id])
            else:
                logger.warn("No trace start passed, set to zero under the assumption that all traces start at the same time.")
                electric_field_.set_trace_start_time(0.)

        electric_field_.set_parameter(efp.ray_path_type, 'direct')
        electric_field_.set_parameter(efp.zenith, zenith)
        electric_field_.set_parameter(efp.azimuth, azimuth)
        sim_station_.add_electric_field(electric_field_)
    sim_station_.set_parameter(stnp.azimuth, azimuth)
    sim_station_.set_parameter(stnp.zenith, zenith)
    energy = corsika['inputs'].attrs["ERANGE"][0] * units.GeV
    sim_station_.set_parameter(stnp.cr_energy, energy)
    sim_station_.set_magnetic_field_vector(magnetic_field_vector)
    sim_station_.set_parameter(
        stnp.cr_xmax, corsika['CoREAS'].attrs['DepthOfShowerMaximum'])
    try:
        sim_station_.set_parameter(
            stnp.cr_energy_em, corsika["highlevel"].attrs["Eem"])
    except KeyError:
        global warning_printed_coreas_py
        if not warning_printed_coreas_py:
            logger.warning(
                "No high-level quantities in HDF5 file, not setting EM energy, this warning will be only printed once")
            warning_printed_coreas_py = True
    sim_station_.set_is_cosmic_ray()
    sim_station_.set_simulation_weight(weight)
    return sim_station_


def select_channels_per_station(det: DetectorBase, station_id: int,
                                requested_channel_ids: Optional[list]) -> defaultdict:
    """
    Returns a defaultdict object containing the requeasted channel ids that are in the given station.
    This dict contains the channel group ids as keys with lists of channel ids as values.

    Parameters
    ----------
    det : DetectorBase
        The detector object that contains the station
    station_id : int
        The station id to select channels from
    requested_channel_ids : list
        List of requested channel ids
    """
    if requested_channel_ids is None:
        requested_channel_ids = det.get_channel_ids(station_id)

    channel_ids = defaultdict(list)
    for channel_id in requested_channel_ids:
        if channel_id in det.get_channel_ids(station_id):
            channel_group_id = det.get_channel_group_id(station_id, channel_id)
            if channel_group_id == -1:
                channel_group_id = channel_id
            channel_ids[channel_group_id].append(channel_id)

    return channel_ids


def position_contained_in_starshape(channel_positions: np.ndarray, starhape_positions: np.ndarray):
    """
    Verify if `station_positions` lie within the starshape defined by `starshape_positions`.
    Ensures interpolation. Projects out z-component.

    station_positions: np.ndarray (n, 3)

    starshape_positions: np.ndarray (m, 3)
    """
    star_radius = np.max(np.linalg.norm(starhape_positions[:, :-1], axis=-1))
    contained = np.linalg.norm(
        channel_positions[:, :-1], axis=-1) <= star_radius
    return contained

