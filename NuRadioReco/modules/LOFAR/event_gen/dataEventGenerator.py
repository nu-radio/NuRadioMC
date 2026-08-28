#!/usr/bin/env python3
"""
NuRadioReco-native data event generator for LOFAR.
"""
import logging
import os

import numpy as np  # noqa: E402

import NuRadioReco.detector.detector  # noqa: E402
import NuRadioReco.modules.eventTypeIdentifier  # noqa: E402
import NuRadioReco.modules.io.eventWriter  # noqa: E402
from NuRadioReco.framework.parameters import stationParameters, showerParameters  # noqa: E402
from NuRadioReco.modules import channelBandPassFilter  # noqa: E402
from NuRadioReco.modules.LOFAR import planeWaveDirectionFitter_LOFAR  # noqa: E402
from NuRadioReco.modules.LOFAR import stationGalacticCalibrator  # noqa: E402
from NuRadioReco.modules.LOFAR import stationPulseFinder  # noqa: E402
from NuRadioReco.modules.LOFAR import stationRFIFilter  # noqa: E402
from NuRadioReco.utilities.LOFAR import (
    DEFAULT_STATIONS,
    TBB_DIRECTORY,
    JSON_DIRECTORY,
    META_DATA_DIRECTORY,
    BLOCK_NUMBER_FILE,
    DATA_TRACE_LENGTH,
    RFI_CLEANING_TRACE_LENGTH,
    CR_SNR,
    PASS_BAND,
)  # noqa: E402

from NuRadioReco.modules.io.LOFAR import readLOFARData  # noqa: E402
from NuRadioReco.utilities import units  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

LOGGER = logging.getLogger("NuRadioReco.LOFAR.dataEventGenerator")


class dataEventGenerator:
    """
    A NuRadioReco-native data event generator for LOFAR.

    The pipeline here is based on the LOFAR data analysis pipeline used in KRATOS, but is implemented in a NuRadioReco-native way. It reads the data from the TBB files, applies RFI filtering, bandpass filtering, galactic calibration, pulse finding, and plane-wave direction fitting.

    .. moduleauthor:: Karen Terveer <karen.terveer@fau.de> & Keito Watanabe <keito.watanabe@kit.edu>
    """

    def __init__(
        self,
        detector=None,
        tbb_directory=TBB_DIRECTORY,
        json_directory=JSON_DIRECTORY,
        meta_data_directory=META_DATA_DIRECTORY,
        block_number_file=BLOCK_NUMBER_FILE,
        output_directory=None,
        log_level=logging.INFO,
    ):
        """
        Parameters:
        ------------
        detector: NuRadioReco.detector.detector.Detector object
            The detector object to use. If None, a default LOFAR detector is initialised.
        tbb_directory: str
            The directory where the TBB data is stored. Defaults to location in Radboud cluster.
        json_directory: str
            The directory where the JSON files are stored. Defaults to location in Radboud cluster.
        meta_data_directory: str
            The directory where the metadata files are stored. Defaults to location in Radboud cluster.
        block_number_file: str
            The file where the block numbers are stored. Defaults to location in Radboud cluster.
        output_directory: str
            The directory where the output files, such as the debug plots and resulting nur files, will be saved. If None, no output files will be saved.
        """

        self.tbb_directory = tbb_directory
        self.json_directory = json_directory
        self.meta_data_directory = meta_data_directory
        self.block_number_file = block_number_file
        self.output_directory = output_directory
        os.makedirs(self.output_directory, exist_ok=True)

        LOGGER.setLevel(log_level)

        self._initialise_detector(detector)

        LOGGER.debug(f"Using TBB directory: {self.tbb_directory}")
        LOGGER.debug(f"Using JSON directory: {self.json_directory}")
        LOGGER.debug(f"Using metadata directory: {self.meta_data_directory}")
        LOGGER.debug(f"Using block number file: {self.block_number_file}")

        LOGGER.info(
            f"Using {DEFAULT_STATIONS} stations. Change in macros.py if you want to use a different set of stations."
        )

        self._initialise_modules()

        self.debug_dir = (
            os.path.join(self.output_directory, "debug_plots")
            if self.output_directory
            else None
        )

    def _initialise_detector(self, detector):
        """
        Initialise the detector object. If no detector is provided, a default LOFAR detector is initialised.
        """
        if detector is None:
            LOGGER.info("No detector provided. Initialising default LOFAR detector.")
            self.detector = NuRadioReco.detector.detector.Detector(
                "LOFAR/LOFAR.json",
                source="json",
                antenna_by_depth=False,
            )
        else:
            self.detector = detector

    def _initialise_modules(self):
        """
        Initialise the modules used in the data event generator.
        """
        LOGGER.info("Initialising modules for data event generator...")
        self.reader = readLOFARData.readLOFARData(
            restricted_station_set=DEFAULT_STATIONS,
            tbb_directory=self.tbb_directory,
            json_directory=self.json_directory,
            metadata_directory=self.meta_data_directory,
        )

        self.event_type_identifier = (
            NuRadioReco.modules.eventTypeIdentifier.eventTypeIdentifier()
        )
        self.rfi_filter = stationRFIFilter.stationRFIFilter()
        self.bandpass_filter = channelBandPassFilter.channelBandPassFilter()
        self.calibrator = stationGalacticCalibrator.stationGalacticCalibrator()
        self.pulse_finder = stationPulseFinder.stationPulseFinder()
        self.direction_fitter = (
            planeWaveDirectionFitter_LOFAR.planeWaveDirectionFitter()
        )

    def process_event(self, event_id, save_debug_plots=False, write_event = False):
        """
        Run the base pipeline for a single event_id. This function reads the data for the given event_id, processes it through the pipeline, and saves the output (if True).

        Parameters:
        ------------
        event_id: str
            The event ID to process.
        save_debug_plots: bool
            If True, debug plots will be saved in the output directory. Defaults to False.
        write_event: bool
            If True, the processed event will be written to a .nur file in the output directory. Defaults to False.

            It is better to set this to False if you want to use the processed event for further analysis, such as reconstruction of shower parameters using the IFT reconstructor or other reconstruction modules. In that case, the processed event will be returned by this function.

        Returns:
        ------------
        processed_event: NuRadioReco.framework.event.Event object
            The processed event object. 

            This can then be used for further analysis, such as reconstruction of shower parameters using the IFT reconstructor or other reconstruction modules.
        """
        LOGGER.info(f"Processing event {event_id}...")
        # initialize the event reader here
        self.reader.begin(event_id=event_id, block_number_file=self.block_number_file)

        event_debug_dir = (
            os.path.join(self.debug_dir, f"{event_id}") if save_debug_plots else None
        )
        nur_file = f"{event_id}.nur"

        processed_event = None
        for event in self.reader.run(self.detector, trace_length=DATA_TRACE_LENGTH):
            if save_debug_plots:
                self._save_trace_snapshot(event, event_debug_dir, "01_reader")

            self.event_type_identifier.begin()
            for station in event.get_stations():
                self.event_type_identifier.run(
                    event, station, mode="forced", forced_event_type="cosmic_ray"
                )

            LOGGER.info("Running LOFAR RFI filter")
            self.rfi_filter.begin(
                reader=self.reader,
                rfi_cleaning_trace_length=RFI_CLEANING_TRACE_LENGTH,
                debug_plot_dir=event_debug_dir,
            )
            self.rfi_filter.run(event)
            self.rfi_filter.end(event if save_debug_plots else None)
            if save_debug_plots:
                self._save_trace_snapshot(event, event_debug_dir, "02_rfi")

            LOGGER.info("Running bandpass filter")
            for station in event.get_stations():
                self.bandpass_filter.run(
                    event,
                    station,
                    self.detector,
                    passband=[PASS_BAND[0] * units.MHz, PASS_BAND[1] * units.MHz],
                    filter_type="gaussian_tapered",
                    roll_width=2.5 * units.MHz,
                )
                self.bandpass_filter.run(
                    event,
                    station,
                    self.detector,
                    passband=[PASS_BAND[0] * units.MHz, PASS_BAND[1] * units.MHz],
                    filter_type="hann_tapered",
                    half_hann_percent=0.1,
                )
            if save_debug_plots:
                self._save_trace_snapshot(event, event_debug_dir, "03_bandpass")

            LOGGER.info("Running galactic calibration")
            self.calibrator.begin()
            self.calibrator.run(event, self.detector)
            self.calibrator.end()
            if save_debug_plots:
                self._save_trace_snapshot(
                    event, event_debug_dir, "04_galactic_calibration"
                )

            LOGGER.info("Running pulse finder")
            self.pulse_finder.begin(cr_snr=CR_SNR, good_channels=6)
            self.pulse_finder.run(event, self.detector)
            self.pulse_finder.end()

            LOGGER.info("Running plane-wave direction fitter")
            self.direction_fitter.begin(
                debug=save_debug_plots,
                debug_plot_dir=event_debug_dir,
            )
            self.direction_fitter.run(event, self.detector)
            self.direction_fitter.end()

            if write_event:
                writer = NuRadioReco.modules.io.eventWriter.eventWriter()
                writer.begin(nur_file)
                writer.run(event)
                writer.end()

            processed_event = event
            break
        
        if processed_event is None:
            raise RuntimeError(f"Pipeline did not produce event {event_id}")
        LOGGER.info("Finished event %s with %d stations", processed_event.get_id(), len(processed_event.get_station_ids()))

        return processed_event 

    def _save_trace_snapshot(self, event, output_dir, stage):
        """
        Save a snapshot of the traces of the first 8 channels of each station in the event.
        """

        max_channels = 8
        os.makedirs(output_dir, exist_ok=True)
        for station in event.get_stations():
            channels = list(station.iter_channels())[:max_channels]
            if not channels:
                continue

            fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
            ytick_positions = []
            ytick_labels = []
            for i_ch, channel in enumerate(channels):
                trace = channel.get_trace()
                if len(trace) == 0:
                    continue
                stride = max(1, len(trace) // 4096)
                samples = np.arange(0, len(trace), stride)
                scale = np.nanmax(np.abs(trace)) or 1.0
                ax.plot(samples, trace[::stride] / scale + i_ch, lw=0.7, alpha=0.8)
                ytick_positions.append(i_ch)
                ytick_labels.append(str(channel.get_id()))

            ax.set_yticks(ytick_positions)
            ax.set_yticklabels(ytick_labels)
            ax.set_xlabel("Sample")
            ax.set_ylabel("Channel ID")
            ax.set_title(
                f"{stage}: CS{station.get_id():03d} first {len(channels)} channels"
            )
            fig.savefig(
                os.path.join(
                    output_dir,
                    f"{stage}_traces_CS{station.get_id():03d}_{event.get_id()}.png",
                ),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)

    def _write_params_nur(self, event, output_dir):
        """Write a .nur file containing only shower/station parameters (no channel traces or e-fields)."""
        for station in event.get_stations():
            for ch_id in list(station.get_channel_ids()):
                station.remove_channel(ch_id)
            station.set_electric_fields([])
        path = os.path.join(output_dir, f"{event.get_id()}.nur")
        writer = NuRadioReco.modules.io.eventWriter.eventWriter()
        writer.begin(path)
        writer.run(event)
        writer.end()
        LOGGER.info("Wrote parameters-only .nur to %s", path)
