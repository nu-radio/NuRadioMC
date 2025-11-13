"""
This is a pipeline to generate data from the CoREAS hdf5 files.
It uses the NuRadioReco Interpolator and Interferometry packages
to get from the CoREAS format to a generic detector. It takes the ```hdf5```
file and outputs fluence plots (detector and meshgrid), long files,
longitudonal development, an interferometry datadump containing final
results and parameters including the longitudonal development
calculated interferometrically. These are outputted for both the
LOFAR-detector and a generic star-shaped detector.
```
dataset
├──nur_files─────────event_id_sim_id.nur
|
├──long──────────────event_id_sim_id.long
|
├──fluence_mesh──────event_id_sim_id.png
|
├──fluence_det───────event_id_sim_id.png
|
├──long_depth────────event_id_sim_id.png
|
├──long_depth_inter──event_id_sim_id.png
|
├──dict_dump─────────dump.pkl
|
├──diagnostic_plots──traces + sum_traces
|
├──plots─────────general overview plots

```
"""

import pickle as pkl
import random
import glob
import os
import logging
import datetime
import re
import shutil
import plotly.graph_objects
import matplotlib.figure

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from typing import Literal, Tuple, List
from pathlib import Path
from matplotlib.colors import Normalize
from astropy.time import Time
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from NuRadioReco.framework.parameters import showerParameters as shp

# -----------NRR imports-----------
import NuRadioReco.framework.event
import NuRadioReco.detector
import NuRadioReco.detector.generic_detector
import NuRadioReco.detector.detector_base
import NuRadioReco.modules.io.coreas.coreasInterpolator

from NuRadioReco.modules.io.LOFAR.readLOFARData import LOFAR_event_id_to_unix
from NuRadioReco.modules.efieldRadioInterferometricReconstruction import (
    efieldInterferometricDepthReco,
)

# Used to read and interpolate CoREAS simulation files to detector positions
from NuRadioReco.modules.io.coreas.readCoREASDetector import readCoREASDetector

# The main class for handling detector geometry
from NuRadioReco.detector import detector as Detector

# A collection of physical units and constants
from NuRadioReco.utilities import units

# A utility function to calculate the energy fluence of an electric field
from NuRadioReco.utilities.trace_utilities import get_electric_field_energy_fluence

# Classes for handling shower and event parameters
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import electricFieldParameters as efp


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(processName)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# individual files at filepath + event_id + run_id + proton/iron +  DAT + sim_id + .long
LONG_FILEPATH = Path("/vol/astro7/lofar/sim/pipeline/events")

# individual files at filepath + event_id + run_id + proton/iron +  SIM + sim_id + .hdf5
HDF5_FILEPATH = Path("/vol/astro7/lofar/sim/hdf5_files")

LOFAR_PATH = (
    "/vol/astro5/lofar/tgottmer/NuRadioMC/NuRadioReco/detector/LOFAR/LOFAR.json"
)
STAR_PATH = "/vol/astro5/lofar/tgottmer/simfiles/sim-detector.json"


def set_fluence_of_efields(
    function, sim_station, quantity=efp.signal_energy_fluence
) -> None:
    """
    This helper function is used to set the fluence quantity of all electric fields in a SimStation.
    Use this to calculate the fluences to use for interpolation.

    One option to use as `function` is `trace_utilities.get_electric_field_energy_fluence()`.

    Parameters
    ----------
    function: callable
        The function to apply to the traces in order to calculate the fluence. Should take in a (3, n_samples) shaped
        array and return a float (or an array with 3 elements if you want the fluence per polarisation).
    sim_station: SimStation
        The simulated station object
    quantity: electric field parameter, default=efp.signal_energy_fluence
        The parameter where to store the result of the fluence calculation
    """
    for electric_field in sim_station.get_electric_fields():
        fluence = function(electric_field.get_trace(), electric_field.get_times())
        electric_field.set_parameter(quantity, fluence)

    return None


def read_event(
    event_file: str,
    detector: Literal["star", "LOFAR"],
    station_ids: list,
    core_position: List[float],
    output_dir: Path,
    event_id: int,
    sim_id: int,
) -> Tuple[
    NuRadioReco.framework.event.Event,
    NuRadioReco.detector.detector_base.DetectorBase
    | NuRadioReco.detector.generic_detector.GenericDetector,
    NuRadioReco.modules.io.coreas.coreasInterpolator.coreasInterpolator,
]:
    """Reads an hdf5 file and corresponding detector
    and returns a NRR-event, detector and interpolator.

    Args:
        event_file (Path): path to event file
        detector (Literal["star", "LOFAR"]): detector type, either star-shaped or LOFAR
        station_ids (list): list of stations to use, 1 for star-shape
        core_position (List[float, float]): core position of shower to use for interpolator
        output_dir (Path): path to output directory
        event_id (int): id of event
        sim_id (int): id of simulation of event

    Returns:
        NuRadioReco.framework.event.Event: NRR event
        NuRadioReco.detector.detector_base.DetectorBase: NRR detector description
        NuRadioReco.modules.io.coreas.coreasInterpolator: Interpolator used to construct event
    """

    if detector == "LOFAR":
        detector_file = LOFAR_PATH
    elif detector == "star":
        detector_file = STAR_PATH

    logger.info(f"Loading detector description from: {detector_file}")
    det = Detector.Detector(detector_file, source="json", antenna_by_depth=False)

    # Set a specific time for the detector.
    event_time = Time(LOFAR_event_id_to_unix(event_id), format="unix")

    det.update(event_time)
    logger.info(f"Detector time set to: {event_time}")

    # --- Step 1: Interpolate E-fields directly onto the detector antenna positions ---
    logger.info(
        f"Interpolating E-fields onto LOFAR antenna positions for stations: {station_ids}"
    )
    readCoREASDetector_inst = readCoREASDetector()

    readCoREASDetector_inst.begin(
        event_file,
        log_level=logging.WARNING,
        site="lofar",
        interp_highfreq=80.0 * units.MHz,
    )
    interpolator = readCoREASDetector_inst.coreas_interpolator

    try:
        station_channel_map = {sid: None for sid in station_ids}

        # The .run() method performs the interpolation and returns an event generator
        event_generator = readCoREASDetector_inst.run(
            det, [core_position], selected_station_channel_ids=station_channel_map
        )

        # A CoREAS file typically contains only one event
        event = next(event_generator)
        logger.info(f"Successfully interpolated event ID {event.get_id()}.")

        # --- Step 2: Update event object with correct time information ---
        # Set the correct time for each station in the newly created event
        for station in event.get_stations():
            station.set_station_time(event_time)

        # --- Step 3: Apply physics cuts and generate plots ---
        event_after_cut = apply_cut(event, det, "Cherenkov")

        make_detector_fluence_plot(event, det, output_dir, detector, event_id, sim_id)

    except StopIteration:
        logger.error(f"No events were found or processed in the file: {event_file}")
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while processing {event_file}: {e}",
            exc_info=True,
        )

    return event_after_cut, det, interpolator


def copy_long_file(long_filepath: Path, output_dir: Path) -> None:
    """Copies the long file from the original location to dataset location.
    Also generates the corresponding longitudonal plot.

    Args:
        long_filepath (Path): Path to .long file produced by CoREAS
        output_dir (Path): Output directory of both figure and longfiles

    Returns:
        None
    """
    shutil.copy(long_filepath, output_dir / "long_files")

    with open(long_filepath) as f:
        first_line = f.readline()

    rows_to_read = int(re.search(r"\d+", first_line).group())
    df = pd.read_table(
        long_filepath, header=1, sep=r"\s+", nrows=rows_to_read, index_col=0
    )
    df.drop("GAMMAS", axis=1, inplace=True)
    ax = df.plot(
        xlabel=r"Shower Depth [$g \; cm^{-2}$]",
        ylabel="#N particles [-]",
        title=f"ID {long_filepath.parts[-5]} {long_filepath.stem}",
    )
    fig = ax.get_figure()
    fig.savefig(
        output_dir
        / "long_depth_simulated"
        / f"{long_filepath.parts[-5]}_{long_filepath.stem}.png"
    )
    plt.close(fig)
    return None


def calc_interferometetric_depth(
    event: NuRadioReco.framework.event.Event,
    detector: NuRadioReco.detector.detector_base.DetectorBase,
    reconstructor: efieldInterferometricDepthReco,
    output_dir: Path,
    event_id: int,
    sim_id: int,
) -> None:
    """Takes an event and adds X_rit to its parameters and stores the
    longitudonal depth profile of the air shower.

    Args:
        event (NuRadioReco.framework.event.Event): NRR event
        detector (NuRadioReco.detector.detector_base.DetectorBase): NRR detector description
        reconstructor (
            NuRadioReco.modules.efieldRadioInterferometricReconstruction.efieldInterferometricDepthReco
        ): The reconstructor for X_rit
        output_dir (Path): Output directry for longitudonal profile plot
        event_id (int): id of the event
        sim_id (int): id of the simulation of the event

    Returns:
        None
    """
    # detector.update(Time(LOFAR_event_id_to_unix(event_id), format="unix"))
    diagnostic_dir = output_dir.parents[1] / "diagnostic_plots"
    try:
        reconstructor.run(event, detector, use_MC_geometry=True, use_MC_pulses=True)
    except RuntimeError:
        logger.error(
            f"RuntimeError while fitting {event_id}_{sim_id}, skipping and plotting traces"
        )
        plot_traces_per_station(
            event,
            event_id,
            sim_id,
            output_dir.parents[1] / "diagnostic_plots" / "traces",
        )

    try:
        long_profile_plot = reconstructor._long_profile_plot
        init_sum_trace = reconstructor._initial_sum_trace
        final_sum_trace = reconstructor._final_sum_trace

        long_profile_plot.suptitle(f"Longitudonal profile ID {event_id} {sim_id}")
        init_sum_trace.suptitle(f"Initial sum trace ID {event_id} {sim_id}")
        final_sum_trace.suptitle(f"Final sum trace ID {event_id} {sim_id}")

        long_profile_plot.savefig(output_dir / f"{event_id}_{sim_id}.png")
        init_sum_trace.savefig(
            diagnostic_dir / "sum_traces" / f"init_sum_trace_{event_id}_{sim_id}.png"
        )
        final_sum_trace.savefig(
            diagnostic_dir / "sum_traces" / f"final_sum_trace_{event_id}_{sim_id}.png"
        )
        plt.close(long_profile_plot)
        plt.close(init_sum_trace)
        plt.close(final_sum_trace)
    except AttributeError:
        logger.error(
            f"Was not able to save reconstrucor profile plots of {event_id} {sim_id}"
        )
        plot_traces_per_station(
            event,
            event_id,
            sim_id,
            output_dir.parents[1] / "diagnostic_plots" / "traces",
        )
    return None


def make_detector_fluence_plot(
    event: NuRadioReco.framework.event.Event,
    detector: NuRadioReco.detector.detector_base.DetectorBase,
    output_dir: Path,
    detector_type: Literal["LOFAR", "star"],
    event_id: int,
    sim_id: int,
) -> None:
    """Generates and saves a plot of the detector radio footprint to output_dir

    Args:
        event (NuRadioReco.framework.event.Event): event to generate the footprint for
        detector (NuRadioReco.detector.detector_base.DetectorBase): detector
        output_dir (Path): directory to save footprint
        detector_type (str): type of detector, either LOFAR or star.
        event_id (int): the id of the event
        sim_id (int): the id of the simulation of the event

    Returns:
        None: None
    """
    antenna_x, antenna_y, fluences = [], [], []
    core_position = event.get_first_sim_shower().get_parameter(shp.core)

    for station in event.get_stations():
        sim_station = station.get_sim_station()
        if not sim_station:
            continue
        station_abs_pos = detector.get_absolute_position(station.get_id())
        for efield in sim_station.get_electric_fields():
            antenna_abs_pos = efield.get_position() + station_abs_pos
            fluence = np.sum(
                get_electric_field_energy_fluence(
                    efield.get_trace(), efield.get_times()
                )
            )
            antenna_x.append(antenna_abs_pos[0])
            antenna_y.append(antenna_abs_pos[1])
            fluences.append(fluence)

    antenna_x, antenna_y, fluences = (
        np.array(antenna_x),
        np.array(antenna_y),
        np.array(fluences),
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    valid_fluences = fluences[fluences > 0]
    vmin = np.min(valid_fluences) if len(valid_fluences) > 0 else 1e-10

    scatter = ax.scatter(
        antenna_x,
        antenna_y,
        c=fluences,
        cmap="viridis",
        norm=Normalize(vmin=vmin, vmax=np.max(fluences)),
        s=35,
        edgecolor="black",
        lw=0.5,
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Energy Fluence (eV/m²)", fontsize=12)
    ax.plot(
        core_position[0],
        core_position[1],
        "*",
        ms=15,
        color="red",
        mec="white",
        label="True Shower Core",
    )
    ax.set_title(f"Simulated Radio Footprint at LOFAR Antennas", fontsize=16)
    ax.set_xlabel("Position X (m)", fontsize=12)
    ax.set_ylabel("Position Y (m)", fontsize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{output_dir}/fluence/{detector_type}/{event_id}_{sim_id}.png")
    plt.close(fig)
    return None


def make_fluence_plot(
    event: NuRadioReco.framework.event.Event,
    interpolator: NuRadioReco.modules.io.coreas.coreasInterpolator.coreasInterpolator,
    output_dir: Path,
    event_id: int,
    sim_id: int,
) -> None:
    """Makes a meshgrid representation of the footprint
    using the interpolator.

    Args:
        event (NuRadioReco.framework.event.Event): NRR event
        interpolator (
            NuRadioReco.modules.io.coreas.coreasInterpolator.coreasInterpolator
        ): Interpolator corresponding to the event
        output_dir (Path): Path to output directory for footprint plot
        event_id (int): id of event
        sim_id (int): id of simulation

    Returns:
        None
    """
    for station in event.get_stations():
        set_fluence_of_efields(
            get_electric_field_energy_fluence, station.get_sim_station()
        )

    interpolator.sim_station = event.get_station(1).get_sim_station()
    interpolator.initialize_fluence_interpolator()
    fig, ax = interpolator.plot_fluence_footprint()

    fig.savefig(output_dir / "footprint" / f"{event_id}_{sim_id}.png")
    plt.close(fig)
    return None


def generate_data(
    event_files: List[Path],
    long_files: List[Path],
    output_dir: Path,
    station_ids: list,
    core_position: List[float] = [0, 0],
    debug: bool = False,
) -> None:
    """Generates all data for single event by filepath. Data including, event
    x_rit, longitudonal profiles and footprints. This is done in the directory
    corresponding to output_dir.

    Args:
        event_files (List[Path]): List of event files to process
        long_files (List[Path]): list of .long files to process
        output_dir (Path): Directory of all output files and directories
        station_ids (list): list of stations to use, only relevant for LOFAR
        core_position (List[float, float], optional): Core offset to interpolate the core to. Defaults to [0, 0].
        debug (bool, optional): Switches debug mode. Defaults to False.

    Returns:
        None
    """
    make_directories(output_dir)
    bad_events = []
    star_dict = {}
    lofar_dict = {}
    station_ids_star = [1]
    interferometric_depth_module = efieldInterferometricDepthReco()
    interferometric_depth_module.begin(debug=debug)

    for i, (event_file, long_file) in enumerate(zip(event_files, long_files)):
        failed = False  # used to check if reading failed -> if so skip interferometry but store data
        event_id, sim_id = int(event_file.parts[-4]), event_file.stem

        logger.info(
            f"processing event_id {i+1} of {len(event_files)} - {event_id} {sim_id}"
        )

        copy_long_file(long_file, output_dir)

        try:
            event_star, det_star, star_interpolator = read_event(
                event_file,
                "star",
                station_ids_star,
                core_position,
                output_dir,
                event_id,
                sim_id,
            )
            event_lofar, det_lofar, lofar_interpolator = read_event(
                event_file,
                "LOFAR",
                station_ids,
                core_position,
                output_dir,
                event_id,
                sim_id,
            )

        except Exception as err:
            logger.exception(
                f"Failed reading or interpolating {event_id} {sim_id}, traceback: {err}"
            )
            bad_events.append(f"{event_id}_{sim_id}")

            if not ("event_star" in locals() and "event_lofar" in locals()):
                logger.info(
                    "No event_lofar and/or event_star exists, skipping iteration"
                )
                continue  # prevents crashing thread if reading event fails

            event_star.get_first_sim_shower().set_parameter(
                shp.fail_mode, "interpolator"
            )
            event_lofar.get_first_sim_shower().set_parameter(
                shp.fail_mode, "interpolator"
            )
            failed = True

        if not failed:
            try:
                make_fluence_plot(
                    event_star, star_interpolator, output_dir, event_id, sim_id
                )

                calc_interferometetric_depth(
                    event_star,
                    det_star,
                    interferometric_depth_module,
                    output_dir / "long_depth_interpolated" / "star",
                    event_id,
                    sim_id,
                )
                calc_interferometetric_depth(
                    event_lofar,
                    det_lofar,
                    interferometric_depth_module,
                    output_dir / "long_depth_interpolated" / "LOFAR",
                    event_id,
                    sim_id,
                )
            except Exception as err:
                logger.exception(
                    f"Failed interferometry for {event_id} {sim_id}, traceback: {err}"
                )
                bad_events.append(f"{event_id}_{sim_id}")
                event_star.get_first_sim_shower().set_parameter(
                    shp.fail_mode, "interferometry"
                )
                event_lofar.get_first_sim_shower().set_parameter(
                    shp.fail_mode, "interferometry"
                )

        try:
            star_dict[event_id][
                sim_id
            ] = event_star.get_first_sim_shower().get_parameters()

            lofar_dict[event_id][
                sim_id
            ] = event_lofar.get_first_sim_shower().get_parameters()
        except KeyError:
            star_dict[event_id] = {}
            star_dict[event_id][
                sim_id
            ] = event_star.get_first_sim_shower().get_parameters()

            lofar_dict[event_id] = {}
            lofar_dict[event_id][
                sim_id
            ] = event_lofar.get_first_sim_shower().get_parameters()

        if (i % 25 == 0 and i != 0) or (i == len(event_files) - 1):
            logger.info(f"Dumping events -- Last event {event_id} {sim_id}")
            with open(
                output_dir / "data" / "star" / f"star_dump_{event_id}_{sim_id}.pkl",
                "wb",
            ) as f:
                pkl.dump(star_dict, f)

            with open(
                output_dir / "data" / "LOFAR" / f"lofar_dump_{event_id}_{sim_id}.pkl",
                "wb",
            ) as f:
                pkl.dump(lofar_dict, f)

            star_dict = {}
            lofar_dict = {}

    with open(output_dir / "bad_events.pkl", "wb") as f:
        pkl.dump(bad_events, f)

    return None


def apply_cut(
    event: NuRadioReco.framework.event.Event,
    detector: NuRadioReco.detector.detector_base.DetectorBase,
    cut_type: Literal["Cherenkov", "Percent"],
) -> NuRadioReco.framework.event.Event:
    """Applies a fluence cut on detectors containing a certain
    amount of the maximum fluence. Either cherenkov-cone based or
    percentage of maximum fluence based (5%)

    Args:
        event (NuRadioReco.framework.event.Event): NRR-event to cut
        detector (NuRadioReco.detector.detector_base.DetectorBase): NRR detector description
        cut_type (Literal["Cherenkov", "Percent"]): Cut type; either by Cherenkov
           radius or by percentage of the maximum fluence.

    Returns:
        NuRadioReco.framework.event.Event: NRR event with specified cut applied
    """
    if cut_type == "Cherenkov":
        logger.info("Calculating Cherenkov radius and applying 3*R_c cut...")

        sim_shower = event.get_first_sim_shower()
        if not sim_shower:
            logger.warning(
                "No simulation shower found in the event. Skipping Cherenkov cut."
            )
            return event

        x_max = sim_shower.get_parameter(shp.shower_maximum)
        zenith = sim_shower.get_parameter(shp.zenith)
        core_position = sim_shower.get_parameter(shp.core)

        # 1. Calculate the altitude of the shower maximum
        altitude_of_x_max = slant_depth_to_altitude(
            x_max / (units.g / units.cm2), zenith
        )
        logger.info(
            f"  - Estimated altitude of shower maximum (Xmax): {altitude_of_x_max / units.km:.2f} km"
        )

        # 2. Calculate the Cherenkov angle in air
        refractive_index_air = 1.000293
        cherenkov_angle_rad = np.arccos(1 / refractive_index_air)
        logger.info(
            f"  - Cherenkov angle in air: {np.rad2deg(cherenkov_angle_rad):.2f} degrees"
        )

        # 3. Calculate the Cherenkov radius on the ground
        cherenkov_radius_m = altitude_of_x_max * np.tan(cherenkov_angle_rad)
        logger.info(
            f"  - Projected Cherenkov radius at ground (R_c): {cherenkov_radius_m:.2f} m"
        )

        # 4. Define the cutoff radius
        cutoff_radius = 3.0 * cherenkov_radius_m
        logger.info(f"  - Applying signal cut at 3 * R_c = {cutoff_radius:.2f} m")

        zeroed_antennas_count = 0
        for station in event.get_stations():
            sim_station = station.get_sim_station()
            if not sim_station:
                continue
            station_abs_pos = detector.get_absolute_position(station.get_id())
            for efield in sim_station.get_electric_fields():
                antenna_abs_pos = efield.get_position() + station_abs_pos
                distance_from_core = np.linalg.norm(
                    antenna_abs_pos[:2] - core_position[:2]
                )
                if distance_from_core > cutoff_radius:
                    efield.set_trace(
                        np.zeros_like(efield.get_trace()), efield.get_sampling_rate()
                    )
                    zeroed_antennas_count += 1
        logger.info(
            f"  - Finished: Zeroed out the E-field traces for {zeroed_antennas_count} antennas beyond the cutoff radius."
        )
    elif cut_type == "Percent":
        logger.info("Calculating maximum flux and cutting antannae at <5%")
        efields, fluences = [], []

        for station in event.get_stations():
            sim_station = station.get_sim_station()
            if not sim_station:
                continue
            for efield in sim_station.get_electric_fields():
                fluence = np.sum(
                    get_electric_field_energy_fluence(
                        efield.get_trace(), efield.get_times()
                    )
                )
                fluences.append(fluence)

        fluences = np.array(fluences)
        antennas_to_cut = fluences < (0.05 * fluences.max())

        for i, efield in enumerate(efields):
            if antennas_to_cut[i]:
                efield.set_trace(
                    np.zeros_like(efield.get_trace()), efield.get_sampling_rate()
                )

        logger.info(f"  - Finshed: Zeroed E-field for {fluences.sum()} antennas")

    return event


def get_filepaths(
    event_ids: List[int], sample_size: int = None
) -> Tuple[List[Path], List[Path]]:
    """Generates a list containing the filepaths for both
    long-files and corresponding CoREAS simulation files.

    Args:
        event_ids (List[int]): ids of events to copy files from
        sample_size (int): num of simulations per event to use

    Returns:
        hdf5_paths (List[Path]): filpaths to hdf5 simulation files for
            event_ids.
        long_paths (List[Path]): filepaths to long files for event_ids.
    """
    hdf5_paths = []
    long_paths = []

    for event_id in event_ids:
        event_path = HDF5_FILEPATH / str(event_id)
        event_sim_files = list(event_path.rglob("*.hdf5"))

        if sample_size:
            try:
                event_sim_files = random.sample(event_sim_files, sample_size)
            except ValueError:
                pass

        hdf5_paths.extend(event_sim_files)

    for fp in hdf5_paths:
        sim_file = fp.stem
        sim_id = sim_file[3:]
        long_path = (
            LONG_FILEPATH
            / "/".join(fp.parts[-4:-2])
            / "coreas"
            / fp.parts[-2]
            / f"DAT{sim_id}.long"
        )
        long_paths.append(long_path)

    return hdf5_paths, long_paths


def slant_depth_to_altitude(x_max_gpcm2: float, zenith_rad: float) -> float:
    """
    Converts a shower maximum slant depth (Xmax) into an altitude in meters.

    This function uses a standard atmospheric parameterization to estimate the
    height above sea level where the shower reached its maximum development.
    This altitude is crucial for calculating the Cherenkov radius.

    Args:
        x_max_gpcm2 (float): Slant depth of shower maximum in g/cm^2.
        zenith_rad (float): The zenith angle of the shower in radians.

    Returns:
        float: The estimated altitude of Xmax in meters.
    """
    # First, convert slant depth to vertical depth
    vertical_depth_gpcm2 = x_max_gpcm2 * np.cos(zenith_rad)
    # Apply the atmospheric model to convert vertical depth to altitude
    altitude_km = 44.33 * (1 - (vertical_depth_gpcm2 / 1173.3) ** 0.19)
    # Convert kilometers to meters for consistency
    return altitude_km * 1000.0 * units.m


def dict_to_df(data: dict) -> pd.DataFrame:
    """Takes a dictionairy containing the events with format
    {event_id: {sim_id: }}. Converts this to a pandas dataframe.
    Also adds columns containing degrees and correct units for
    X_max, X_rit and their difference.

    Args:
        data (dict): dictionairy as described

    Returns:
        pd.DataFrame: df containing the events
    """
    dfs = []

    for event_dict in data:
        df = pd.DataFrame.from_dict(data[event_dict], orient="index")
        df["event_id"] = event_dict
        dfs.append(df)

    full_df = pd.concat(dfs)

    full_df["zenith_deg"] = np.rad2deg(full_df[shp.zenith])
    full_df["azimuth_deg"] = np.rad2deg(full_df[shp.azimuth])
    full_df["X_max"] = full_df[shp.shower_maximum] / (units.g / units.cm2)
    try:
        full_df["X_rit"] = full_df[shp.interferometric_shower_maximum] / (
            units.g / units.cm2
        )
        full_df["diff_max_rit"] = full_df["X_rit"] / full_df["X_max"]
    except KeyError:
        full_df["X_rit"] = np.nan
        full_df["diff_max_rit"] = np.nan

    return full_df


def read_events_from_pkl(directory: Path) -> pd.DataFrame:
    """Converts the pickle files in a directory to a pandas dataframe.

    Args:
        directory (Path): path to directory containing pickle files

    Returns:
        pd.DataFrame: DataFrame containing all events
    """
    files = directory.glob("*.pkl")
    event_dicts = []

    for file in files:
        with open(file, "rb") as f:
            data = pkl.load(f)
            event_dicts.append(data)

    dfs = []

    for data_dict in event_dicts:
        df = dict_to_df(data_dict)
        dfs.append(df)

    full_df = pd.concat(dfs)

    return full_df


def full_plot(data_df: pd.DataFrame, title: str) -> matplotlib.figure.Figure:
    """Makes a full breakdown plot consisting of two subplots. Both plots plot
    X_max vs X_rit, but one has the color scale determined by the azimuth and
    the other by it's zenith angle. Clips lower end of X_rit at -1000, NaNs are
    plotted to X_rit = 0.

    Args:
        data_df (pd.DataFrame): Df containing the data-to-plot
        title (str): title of the plot

    Returns:
        matplotlib.figure.Figure: figure containing the subplots
    """
    # copy x_rit column into new series to avoid tampering with original copy
    x_rit = data_df["X_rit"].fillna(0).clip(-1000, None)

    fig, ax = plt.subplots(1, 2, figsize=(20, 10), layout="tight")
    plt1 = ax[0].scatter(data_df["X_max"], x_rit, c=data_df["zenith_deg"])
    ax[0].axhline(y=0, color="orange", alpha=0.3, linestyle="dashed")
    ax[0].set_title("By zenith")
    ax[0].set_xlabel(r"$X_{max} \; [g \; cm^{-2}]$")
    ax[0].set_ylabel(r"$ X_{rit} \; [g \; cm^{-2}]$")
    ax1_divider = make_axes_locatable(ax[0])
    cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")

    plt2 = ax[1].scatter(data_df["X_max"], x_rit, c=data_df["azimuth_deg"])
    ax[1].axhline(y=0, color="orange", alpha=0.3, linestyle="dashed")
    ax[1].set_title("By azimuth")
    ax[1].set_xlabel(r"$X_{max} \: [g \: cm^{-2}]$")
    ax[1].set_ylabel(r"$ X_{rit} \: [g \: cm^{-2}]$")
    ax2_divider = make_axes_locatable(ax[1])
    cax2 = ax2_divider.append_axes("right", size="7%", pad="2%")

    fig.colorbar(plt1, cax=cax1, label="Zenith angle [deg]")
    fig.colorbar(plt2, cax=cax2, label="Azimuth angle [deg]")
    fig.suptitle(title, fontsize=24, fontweight="roman")

    return fig


def three_dim_plot(data: pd.DataFrame) -> plotly.graph_objects.Figure:
    """Makes an interactive three dimensional plot of X_max vs azimuth vs zenith
    with a color scale for X_rit. Clips lower bound at -500 and upper bound at
    1200 for X_rit. NaN (failed fits) show up as grey in 3d plot.

    Args:
        data (pd.DataFrame): DF containing the data and necessary columns

    Returns:
        plotly.graph_objects.Figure: figure containing the 3d plot
    """
    fig = px.scatter_3d(
        data,
        "azimuth_deg",
        "zenith_deg",
        "X_max",
        "X_rit",
        size=np.full(data.shape[0], 0.01),
        hover_name="event_id",
        opacity=1,
        height=600,
        range_color=[-500, 1200],
    )
    return fig


def diagnostic_hist(data: pd.DataFrame) -> matplotlib.figure.Figure:
    """Makes a diagnostic histogram plot of the distribution of the
    zenith, azimuth, X_max, X_rit, diff_max_rit and energies of the
    analysed air showers.

    Args:
        data (pd.DataFrame): df containing relevant columns

    Returns:
        matplotlib.figure.Figure: figure containing histograms
    """
    hist = data.hist(
        column=[
            "zenith_deg",
            "azimuth_deg",
            "X_max",
            "X_rit",
            "diff_max_rit",
            shp.energy,
        ],
        layout=(2, 3),
    )
    fig = plt.gcf()
    fig.tight_layout()

    return fig


def select_events_to_analyse(num_events: int) -> np.ndarray:
    """Selects a number of events in HDF5_FILEPATHS to analyse

    Args:
        num_events (int): Number of events wanted

    Returns:
        np.ndarray: array containing random selection of event ideas
            of length num_events
    """
    subdirs = [int(id) for id in os.listdir(HDF5_FILEPATH)]
    event_ids = random.sample(subdirs, num_events)

    return np.array(event_ids)


def plot_traces_per_station(
    event: NuRadioReco.framework.event.Event,
    event_id: int,
    sim_id: str,
    output_dir: Path,
    subset_traces: int = None,
) -> None:
    """Makes plots of all traces per station for event supplied over time.
    Saves these to output_dir. Useful for debugging of failed fits. Can
    also do this for a subset of antennaes. Traces are offset so every single
    one can be seen individually. Will produce a very tall plot if all traces are
    used.

    Args:
        event (NuRadioReco.framework.event.Event): event to plot
        event_id (int): id of event
        sim_id (str): simulation id of event
        output_dir (Path): path to output directory
        subset_traces(int): number of traces to plot. Default to None for all traces.

    Returns:
        None
    """
    # Plot traces
    efields_per_station = {}
    for station in event.get_stations():
        efields = station.get_sim_station().get_electric_fields()
        efields_per_station[f"station {station.get_id()}"] = {
            "trace": np.array([field.get_trace() for field in efields]),
            "times": np.array([field.get_times() for field in efields]),
        }

    for station in efields_per_station:
        fig, ax = plt.subplots(1, 1, layout="constrained", figsize=(5, 40))
        signal_group = efields_per_station[station]

        if subset_traces:
            signals = [
                signal_group["trace"][:subset_traces, 0],
                signal_group["trace"][:subset_traces, 1],
                signal_group["trace"][:subset_traces, 2],
            ]
        else:
            signals = [
                signal_group["trace"][:, 0],
                signal_group["trace"][:, 1],
                signal_group["trace"][:, 2],
            ]

        t = signal_group["times"][:].T

        # Center the signals and add offsets
        sigs = signals[0] - signals[0].mean(axis=1, keepdims=True)
        offset = np.max(sigs.max(axis=1)[:-1] - sigs.min(axis=1)[1:])
        sigs = sigs.T + np.arange(len(signals[0])) * offset

        ax.plot(t, sigs)
        ax.tick_params(left=False, labelleft=False)
        ax.set_xlabel("time [ns]")
        ax.set_title("x")

        fig.suptitle(
            f"Efield traces for antennaes of {station} for {event_id} {sim_id}"
        )
        fig.savefig(output_dir / f"{event_id}_{sim_id}_{station}.png")
        plt.close(fig)

    return None


def make_directories(output_dir: Path) -> None:
    """Makes all necessary output directories for the set file structure
    for the analysis of the events. If directories are missing, will make
    these. If directories already exist, do nothing.

    Args:
        output_dir (Path): path to top-level output directory

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir / "long_files", exist_ok=True)
    os.makedirs(output_dir / "footprint", exist_ok=True)
    os.makedirs(output_dir / "fluence" / "star", exist_ok=True)
    os.makedirs(output_dir / "fluence" / "LOFAR", exist_ok=True)
    os.makedirs(output_dir / "long_depth_simulated", exist_ok=True)
    os.makedirs(output_dir / "long_depth_interpolated" / "star", exist_ok=True)
    os.makedirs(output_dir / "long_depth_interpolated" / "LOFAR", exist_ok=True)
    os.makedirs(output_dir / "data" / "star", exist_ok=True)
    os.makedirs(output_dir / "data" / "LOFAR", exist_ok=True)
    os.makedirs(output_dir / "diagnostic_plots" / "traces", exist_ok=True)
    os.makedirs(output_dir / "diagnostic_plots" / "sum_traces", exist_ok=True)
    os.makedirs(output_dir / "plots", exist_ok=True)

    return None
