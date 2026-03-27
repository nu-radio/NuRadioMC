import os
import logging
import pipeline
import re
import shutil

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

from pathlib import Path
from astropy.time import Time
from typing import Literal

import NuRadioReco.modules.eventTypeIdentifier
from NuRadioReco.utilities import units
from NuRadioReco.detector import detector
from NuRadioReco.modules.io.LOFAR.readLOFARData import LOFAR_event_id_to_unix
from NuRadioReco.modules.io.LOFAR import readLOFARData
from NuRadioReco.modules import (
    channelBandPassFilter,
    voltageToEfieldConverter,
    channelResampler,
)
from NuRadioReco.modules.io.eventWriter import eventWriter
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.modules.LOFAR import (
    stationRFIFilter,
    stationGalacticCalibrator,
    stationPulseFinder,
    planeWaveDirectionFitter_LOFAR,
)
from NuRadioReco.modules.efieldRadioInterferometricReconstruction import (
    efieldInterferometricDepthReco,
    efieldInterferometricAxisReco,
)

LOFAR_PATH = (
    "/vol/astro5/lofar/tgottmer/NuRadioMC/NuRadioReco/detector/LOFAR/LOFAR.json"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(processName)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_sim_files(
    df: pd.DataFrame, files: Literal["all", "sample", "match"], n_sims: int = None
) -> dict:
    """Gets closes simulation files based on x_max of the associated
    LOFAR data file.

    Args:
        df (pd.DataFrame): DataFrame containing info about the lofar data files
        files (str): all for all sims, match for only the matching sim

    Returns:
        dict: contains hdf5 and long filepaths to closest matching simulation
            per event_id.
    """
    final_files = {}

    for i, series in df.iterrows():
        event_id = int(series["event_id"])
        x_max = series["xreco"]
        if files == "all" or files == "match":
            hdf5, long = pipeline.get_filepaths([event_id])
        elif files == "sample":
            hdf5, long = pipeline.get_filepaths([event_id], n_sims)
        diff_x_max = 1e3

        if files == "match":
            for fp_5, fp_long in zip(hdf5, long):
                try:
                    with open(fp_long) as f:
                        first_line = f.readline()
                except FileNotFoundError:
                    continue

                try:
                    rows_to_read = int(re.search(r"\d+", first_line).group())
                except AttributeError:
                    continue

                df = pd.read_table(
                    fp_long, header=1, sep=r"\s+", nrows=rows_to_read, index_col=0
                )
                sim_x_max = df["CHARGED"].idxmax()
                diff = abs(sim_x_max - x_max)

                if diff < diff_x_max:
                    closest_sim_files = [fp_5, fp_long]
                    diff_x_max = diff

            final_files[event_id] = closest_sim_files
        else:
            final_files[event_id] = (hdf5, long)

    return final_files


def lofar_data_processing(
    df: pd.DataFrame, output_dir: Path, sim_files: dict, snr: float, write_out: bool=False, use_simulations: bool=True
) -> None:
    det = detector.Detector(LOFAR_PATH, source="json", antenna_by_depth=False)
    reader = readLOFARData.readLOFARData(
        restricted_station_set=[
            "CS002",
            "CS003",
            "CS004",
            "CS005",
            "CS006",
            "CS007",
        ]
    )
    eventTypeIdentifier = NuRadioReco.modules.eventTypeIdentifier.eventTypeIdentifier()
    rfi_filter = stationRFIFilter.stationRFIFilter()
    bandpass_filter = channelBandPassFilter.channelBandPassFilter()
    calibrator = stationGalacticCalibrator.stationGalacticCalibrator()
    pulse_finder = stationPulseFinder.stationPulseFinder()
    event_writer = eventWriter()
    resampler = channelResampler.channelResampler()

    sim_id = 0
    diagnostic_dir = output_dir / "diagnostic_plots"
    nur_dir = output_dir / "nur"
    py_dir = output_dir / "python_files"
    data_dict = {}

    interferometric_depth_module = efieldInterferometricDepthReco()
    interferometric_depth_module.begin(debug=True)
    axis_reco = efieldInterferometricAxisReco()
    axis_reco.begin(debug=False)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(diagnostic_dir, exist_ok=True)
    os.makedirs(nur_dir, exist_ok=True)
    os.makedirs(output_dir / "pickle", exist_ok=True)
    os.makedirs(py_dir, exist_ok=True)

    # Copy relevant pythoon files containing all parameters as i forget
    shutil.copy("NuRadioMC/gen_data.py", py_dir)
    shutil.copy("NuRadioMC/gen_data.sh", py_dir)
    shutil.copy("NuRadioMC/gen_lofar_data.py", py_dir)
    shutil.copy("NuRadioMC/lofar_processing.py", py_dir)
    shutil.copy("NuRadioMC/pipeline.py", py_dir)
    shutil.copy(
        "NuRadioMC/NuRadioReco/modules/efieldRadioInterferometricReconstruction.py",
        py_dir,
    )

    for i, series in df.iterrows():
        event_id = series.at["event_id"]
        logger.info(f"Reading {event_id}")
        det.update(Time(LOFAR_event_id_to_unix(event_id), format="unix"))

        try:
            reader.begin(event_id)
        except FileNotFoundError:
            logger.warning(f"Couldn't find file for {event_id}")
            reader = readLOFARData.readLOFARData(
                restricted_station_set=[
                    "CS002",
                    "CS003",
                    "CS004",
                    "CS005",
                    "CS006",
                    "CS007",
                ]
            )
            continue
        except ValueError:
            logger.warning(f"Error during reading of event {event_id}")
            reader = readLOFARData.readLOFARData(
                restricted_station_set=[
                    "CS002",
                    "CS003",
                    "CS004",
                    "CS005",
                    "CS006",
                    "CS007",
                ]
            )
            continue

        evt = next(reader.run(det))

        eventTypeIdentifier.begin()
        for station in evt.get_stations():
            eventTypeIdentifier.run(
                evt, station, mode="forced", forced_event_type="cosmic_ray"
            )

        logger.info("Cleaning RFI...")
        # Apply RFI cleaning and bandpass filters
        rfi_filter.begin(reader=reader, rfi_cleaning_trace_length=8192)
        rfi_filter.run(evt)
        rfi_filter.end()

        logger.info("Bandpassing...")
        for station in evt.get_stations():
            bandpass_filter.run(
                evt,
                station,
                det,
                passband=[30 * units.MHz, 80 * units.MHz],
                filter_type="gaussian_tapered",
                roll_width=2.5 * units.MHz,
            )
            bandpass_filter.run(
                evt,
                station,
                det,
                passband=[30 * units.MHz, 80 * units.MHz],
                filter_type="hann_tapered",
                half_hann_percent=0.1,
            )

        logger.info("Calibrating")
        # Calibrate the traces based on Galactic noise
        calibrator.begin()
        calibrator.run(evt, det)
        calibrator.end()

        logger.info("Finding pulses...")
        # Find stations with significant pulse and corresponding good channels in those stations
        pulse_finder.begin()
        pulse_finder.run(evt, det)
        pulse_finder.end()

        # Upsample to 0.4 GHz
        resampler.begin()
        for station in evt.get_stations():
            resampler.run(evt, station, det, sampling_rate=0.4 * units.GHz)
        resampler.end()

        evt.get_first_shower().set_parameter(shp.azimuth, series.at["azimuth"])
        evt.get_first_shower().set_parameter(shp.zenith, series.at["zenith"])
        evt.get_first_shower().set_parameter(
            shp.core, [series.at["core_x"], series.at["core_y"], 7.6]
        )
        evt.get_first_shower().set_parameter(shp.atmospheric_model, np.int64(1))
        evt.get_first_shower().set_parameter(
            shp.refractive_index_at_ground, np.float64(1.000292)
        )
        evt.get_first_shower().set_parameter(
            shp.shower_maximum, series.at["xreco"] * (units.g / units.cm2)
        )
        evt.get_first_shower().get_parameters()
        if write_out:
            event_writer.begin(str(nur_dir / f"{event_id}.nur"))
            event_writer.run(evt)

        logger.info("Cutting on snr...")
        evt = pipeline.apply_cut(evt, det, "SNR", snr_cut=snr)

        try:
            logger.info("Running axis interferometry...")
            axis_reco.run(
                evt,
                det,
                use_MC_geometry=False,
                use_MC_pulses=False,
                use_voltage_traces=True,
                n_samples=2048,
                cross_section_size=200,
                cross_section_spacing=[10, 10, 10, 10, 10],
                depths=[400, 500, 600, 700, 800, 900],
                bootstrap=True,
            )

            logger.info("Running depth interferometry...")
            interferometric_depth_module.run(
                evt,
                det,
                use_MC_geometry=False,
                use_MC_pulses=False,
                use_voltage_traces=True,
                n_samples=1024,
                use_interferometric_axis=True,
            )
        except RuntimeError:
            logger.warning(f"RuntimeError during interferometry of {event_id}")

        try:
            data_dict[event_id] = evt.get_first_shower().get_parameters()
        except KeyError:
            data_dict[event_id] = {}
            data_dict[event_id] = evt.get_first_shower().get_parameters()

        try:
            data_dict[f"{event_id}_axis_cov"] = axis_reco._axis_pcov
            data_dict[f"{event_id}_depth_cov"] = (
                interferometric_depth_module._peak_fit_pcov
            )
        except AttributeError:
            logger.info(f"Wasn't able to store covariances for event {event_id}")

        logger.info(f"Dumping event -- Last event {event_id}")
        with open(
            output_dir / "pickle" / f"data_dump_{event_id}.pkl",
            "wb",
        ) as f:
            pkl.dump(data_dict, f)

        data_dict = {}

        long_profile_plot = interferometric_depth_module._long_profile_plot
        init_sum_trace = interferometric_depth_module._initial_sum_trace
        final_sum_trace = interferometric_depth_module._final_sum_trace

        long_profile_plot.suptitle(f"Longitudonal profile ID {event_id} {sim_id}")
        init_sum_trace.suptitle(f"Initial sum trace ID {event_id} {sim_id}")
        # final_sum_trace.suptitle(f"Final sum trace ID {event_id} {sim_id}")

        long_profile_plot.savefig(output_dir / f"{event_id}_{sim_id}.png")
        init_sum_trace.savefig(
            diagnostic_dir / f"init_sum_trace_{event_id}_{sim_id}.png"
        )
        # final_sum_trace.savefig(
        #     diagnostic_dir / f"final_sum_trace_{event_id}_{sim_id}.png"
        # )
        plt.close(long_profile_plot)
        plt.close(init_sum_trace)
        plt.close(final_sum_trace)
        if use_simulations:
            logger.info("Generating simulation data...")
            pipeline.generate_data(
                sim_files[event_id][0],
                sim_files[event_id][1],
                output_dir.parents[0] / "sim",
                [2, 3, 4, 5, 6, 7],
                core_position=[series.at["core_x"], series.at["core_y"]],
            )
    interferometric_depth_module.end()
