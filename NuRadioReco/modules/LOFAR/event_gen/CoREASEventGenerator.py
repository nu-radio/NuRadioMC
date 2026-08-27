"""
Simulated event generator for LOFAR.
"""

import os
import h5py
import logging
import argparse
import numpy as np
from collections import defaultdict

from matplotlib import pyplot as plt
from matplotlib import cm as cm
from scipy import optimize as opt
from scipy import constants
from astropy import time
import datetime

import NuRadioReco.framework.event
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.eventTypeIdentifier
import NuRadioReco.modules.measured_noise.channelMeasuredNoiseAdder

from NuRadioReco.utilities import units, trace_utilities
from NuRadioReco.detector import detector
from NuRadioReco.modules.io.coreas import coreas, coreasInterpolator, readCoREASDetector
from NuRadioReco.utilities.dataservers import download_from_dataserver
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import stationParameters as stp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.utilities.trace_utilities import get_electric_field_energy_fluence

from NuRadioReco.modules.LOFAR import planeWaveDirectionFitter_LOFAR  # noqa: E402
from NuRadioReco.modules.LOFAR import stationPulseFinder  # noqa: E402
from NuRadioReco.modules.LOFAR import LORASimulator
from NuRadioReco.modules.measured_noise.LOFAR import LOFARnoiseLibraryConverter
from NuRadioReco.utilities.LOFAR import (
    DEFAULT_STATIONS,
    CR_SNR,
    PASS_BAND,
    START_TIME,
    NOISE_LIBRARY_DIRECTORY,
    NOISE_LIBRARY_NUR_FILEPATH,
    ALWAYS_REMOVED_CHANNEL_IDS
)  # noqa: E402

import matplotlib.cm as cm
import matplotlib.colors as mcolors

from NuRadioReco.detector.antennapattern import AntennaPatternProvider, preprocess_LOFAR_txt

import json

LOGGER = logging.getLogger("NuRadioReco.LOFAR.CoREASEventGenerator")

class CoREASEventGenerator:
    """
    A NuRadioReco-native simulated event generator for LOFAR. This class provides a framework for generating simulated events based on CoREAS simulations, incorporating the LOFAR hardware response, and processing the events through a pipeline that includes noise addition, filtering, and signal reconstruction.

    .. moduleauthor:: Karen Terveer <karen.terveer@fau.de> & Keito Watanabe <keito.watanabe@kit.edu>
    """
    def __init__(
        self,
        detector=None,
        output_directory = None,
        log_level=logging.INFO,
    ):

        self.selected_station_channel_ids = {}
        
        self._initialise_detector(detector)
        self._initialise_modules()

        self.output_directory = output_directory
        self.debug_dir = (
            os.path.join(self.output_directory, "debug_plots")
            if self.output_directory
            else None
        )
        LOGGER.setLevel(log_level)

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

        self.detector.update(START_TIME)
        
        for station_name in DEFAULT_STATIONS:
            # NOTE: this assumes that the station names are in the format "CS###", where ### is the station ID. If the station names are different, this will need to be modified accordingly.
            staid = int(station_name.replace("CS", ""))
            self.selected_station_channel_ids[staid] = self.detector.get_channel_ids(staid)

    def _initialise_modules(self):
        """
        Initialise all modules used in the simulation pipeline, including:

        - readCoREASDetector
        - efieldToVoltageConverter
        - channelMeasuredNoiseAdder
        - triggerSimulator
        - channelBandPassFilter
        - eventTypeIdentifier
        - stationPulseFinder
        - planeWaveDirectionFitter_LOFAR

        Note: only the modules are initialised, and the begin functions are called in the run_pipeline function, since some modules require event information to be initialised.
        """
        self.LORASimulator = LORASimulator.LORASimulator()
        
        # Initialize the modules
        self.efieldToVoltageConverter = (
            NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter(
                log_level=logging.INFO
            )
        )
        
        self.channelBandPassFilter = (
            NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
        )

        self.channelResampler = (
            NuRadioReco.modules.channelResampler.channelResampler()
        )

        self.triggerSimulator = (
            NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()
        )

        self.eventTypeIdentifier = (
            NuRadioReco.modules.eventTypeIdentifier.eventTypeIdentifier()
        )

        self.pulse_finder = stationPulseFinder.stationPulseFinder()
        
        self.direction_fitter = (
            planeWaveDirectionFitter_LOFAR.planeWaveDirectionFitter()
        )

        self.coreas_reader = readCoREASDetector.readCoREASDetector()

        self.LOFARnoiseLibraryConverter = LOFARnoiseLibraryConverter()
        self.channelMeasuredNoiseAdder = (
            NuRadioReco.modules.measured_noise.channelMeasuredNoiseAdder.channelMeasuredNoiseAdder()
        )



    def process_event(
        self,
        coreas_hdf5_file : str,
        sky_model : str = "gsm2016",
        noise_temperature : float = 300.0,
        save_debug_plots=False, 
        write_event = False
    ):
        """
        Run the simulation pipeline for a given CoREAS HDF5 file.

        Parameters
        ----------
        coreas_hdf5_file : str
            Path to the CoREAS HDF5 file.
        sky_model : str, optional
            Sky model to use for galactic noise addition. Default is "LFmap".
        noise_temperature : float, optional
            Noise temperature in Kelvin for generic noise addition. Default is 300.0 K.
        save_debug_plots : bool, optional
            Whether to save debug plots at various stages of the pipeline. Default is False.
        write_event : bool, optional
            Whether to write the final event to a file. Default is False.
        """
        # initialize reader and get the CoREAS as NuRadio Event
        sim_event = self._initialise_reader(coreas_hdf5_file)

        # define debugging output directory if requested
        # For now we just take the event ID from the CoREAS file name, but this should be changed to a more robust method in the future.
        sim_event_label = f"{sim_event.get_id():06d}"

        event_debug_dir = os.path.join(self.debug_dir, sim_event_label) if save_debug_plots else None
        nur_file = os.path.join(self.output_directory, f"{sim_event_label}.nur") if write_event else None

        # # generate the temporary noise library .nur file
        noise_library_nur_filepath, noise_library_channel_mapping = self._run_noise_library_converter(sim_event)

        # call all begin functions here
        self.LORASimulator.begin()
        # TODO: are these pre_pulse_time and post_pulse_time values reasonable? They are currently set to 0 ns and 400 ns, respectively, which may not be optimal for all cases.
        self.efieldToVoltageConverter.begin(
            debug=False, pre_pulse_time=0 * units.ns, post_pulse_time=400 * units.ns
        )
        self.channelResampler.begin()
        self.channelMeasuredNoiseAdder.begin(
            filenames = [noise_library_nur_filepath],
            restrict_station_id = False, # no 1-1 mapping of station IDs
            station_id = None,  # just use the first station ID 
            channel_mapping = noise_library_channel_mapping,  # maps all channel IDs (in all stations) to a single noise channel ID
            debug=True
        )
        self.channelBandPassFilter.begin()
        self.channelResampler.begin()
        self.triggerSimulator.begin()
        self.eventTypeIdentifier.begin()
        self.pulse_finder.begin(cr_snr=CR_SNR, good_channels=6, window=10, noise_window=150) # window size and noise window reduced since sampling size is smaller for simulated traces. The numbers here are arbitrary at the moment.
        self.direction_fitter.begin(
            debug=save_debug_plots,
            debug_plot_dir=event_debug_dir,
        )

        filter_settings = {
            "filter_type": "butter",
            "order": 10,
            "passband": [PASS_BAND[0] * units.MHz, PASS_BAND[1] * units.MHz],
        }

        processed_event = None
        LOGGER.info(f"Processing event {sim_event.get_id()} from CoREAS file {coreas_hdf5_file}")
        for evt in self.coreas_reader.run(self.detector, None, selected_station_channel_ids=self.selected_station_channel_ids):
            
            LOGGER.info(f"Converting electric field to voltage for event {evt.get_id()} at time {START_TIME}")
            for station in evt.get_stations():

                # set the station time to the start time defined in macros.py
                # TODO: should this not be random / based on a given observation time? 
                # it should be read based on LORA triggered event time.
                station.set_station_time(START_TIME)
                if save_debug_plots:
                    self._save_efield_trace_snapshot(evt, station.get_sim_station(), output_dir=event_debug_dir, stage="01_reader")

                # Convert electric field to voltage
                self.efieldToVoltageConverter.run(evt, station, self.detector)
                if save_debug_plots:
                    self._save_trace_snapshot(evt, station, output_dir=event_debug_dir, stage="02_efieldToVoltage")

                # resample the trace to that of LOFAR (uses the default if not provided)
                self.channelResampler.run(evt, station, self.detector)
                if save_debug_plots:
                    self._save_trace_snapshot(evt, station, output_dir=event_debug_dir, stage="03_resample")

                # Apply bandpass filter
                self.channelBandPassFilter.run(evt, station, self.detector, **filter_settings)
                if save_debug_plots:
                    self._save_trace_snapshot(evt, station, output_dir=event_debug_dir, stage="04_bandpass")

                # # Add measured noise from LOFAR
                self.channelMeasuredNoiseAdder.run(evt, station, self.detector)
                if save_debug_plots:
                    self._save_trace_snapshot(evt, station, output_dir=event_debug_dir, stage="05_measured_noise")

                # Identify event type
                self.eventTypeIdentifier.run(evt, station, mode='forced', forced_event_type='cosmic_ray')

                # now flag channels that we should always remove
                # since in the data its always broken
                self._add_flagged_channel_ids(evt, station)

            # TODO: understand how to treat the trigger in LOFAR. For now, we will skip this step and assume that the event is triggered based on the pulse finding criteria.
            LOGGER.info(f"Finished processing event {evt.get_id()} from CoREAS file {coreas_hdf5_file}")
            LOGGER.info("Running pulse finding")
            # Find pulses in the stations
            self.pulse_finder.run(evt, self.detector)

            LOGGER.info("Running plane-wave direction fitting")
            # Fit direction of arrival
            self.direction_fitter.run(evt, self.detector)
            
            # now we end all functions
            self.LORASimulator.end()
            self.efieldToVoltageConverter.end()
            self.channelResampler.end()
            self.channelBandPassFilter.end()
            self.triggerSimulator.end()
            self.pulse_finder.end()
            self.direction_fitter.end()
            self.LOFARnoiseLibraryConverter.end()
            self.channelMeasuredNoiseAdder.end()

            if write_event:
                writer = NuRadioReco.modules.io.eventWriter.eventWriter()
                writer.begin(nur_file)
                writer.run(evt)
                writer.end()

            processed_event = evt
            break

        if processed_event is None:
            raise RuntimeError(f"No events were processed from CoREAS file {coreas_hdf5_file}. Please check the input file and the selected stations.")
        LOGGER.info(f"Finished processing event {processed_event.get_id()} from CoREAS file {coreas_hdf5_file}")

        return processed_event

    def _initialise_reader(self, coreas_hdf5_file):
        """
        Initialise the CoREAS reader module with the given HDF5 file.

        Currently this is just repeating the begin function of readCoREASDetector, but with a forced vertical core coordinate. This can be replaced with the usual begin function of readCoREASDetector when using a good / correct HDF5 file with the correct CoreCoordinateVertical. 

        Parameters
        ----------
        coreas_hdf5_file : str
            Path to the CoREAS HDF5 file.
        """
        corsika_event = coreas.read_CORSIKA7(coreas_hdf5_file, site='lofar')

        # here we force vertical core coordinate
        corsika_event.get_first_sim_shower().set_parameter(
            shp.core,
            np.array(
                [0, 0, corsika_event.get_first_sim_shower().get_parameter(shp.observation_level)]
            ),
        )

        interpolator = coreasInterpolator.coreasInterpolator(corsika_event)
        interpolator.initialize_efield_interpolator(
            interp_lowfreq=PASS_BAND[0] * units.MHz, interp_highfreq=PASS_BAND[1] * units.MHz
        )
        self.coreas_reader.coreas_interpolator = interpolator  # skip begin() function because HDF5 does not have good CoreCoordinateVertical

        # adding the LORA simulator here for now. This generates a hybrid shower based on 
        # true shower parameters with cores & angles randomly sampled from normal distribution with 
        # LORA uncertainties.
        # TODO: simply return a hybrid shower here and instead make a hybrid shower adder in modules/io, removing code from readCoREASDetector (for more modularity)
        lora_shower = self.LORASimulator.run(corsika_event, self.detector)
        self.coreas_reader._readCoREASDetector__hybrid_shower_name = lora_shower.get_name()  # force the reader to use the hybrid shower generated by LORA simulator

        self.coreas_reader._readCoREASDetector__corsika_evt = corsika_event

        return corsika_event

    def _run_noise_library_converter(self, event):
        """
        Run the noise library converter, separately and only once.

        This generates the conversion from the .npy file to the .nur file, as well as the channel mapping, which maps all detector channels to a single noise channel. This is done since we assume the noise characteristics is the same for all antennas.
        """
        self.LOFARnoiseLibraryConverter.begin(
            library_filename = NOISE_LIBRARY_DIRECTORY,
            nur_filename = NOISE_LIBRARY_NUR_FILEPATH
        )

        channel_mapping = self.LOFARnoiseLibraryConverter.run(event, self.detector) 

        return self.LOFARnoiseLibraryConverter.get_nur_filepath(), channel_mapping

    def _add_flagged_channel_ids(self, evt, station):
        """
        Add flagged channel IDs that we omit for reconstruction, since its known from data that that channel is always faulty.

        The channel to remove is given in the macros.

        This replicates what already exists in readLOFARData.

        Parameters
        ----------
        evt : NuRadioReco event object
            dummy argument
        station : NuRadioReco.framework.Station
            station object
        """
        flagged_nrr_channel_ids: dict = defaultdict(list)
        for channel_id in ALWAYS_REMOVED_CHANNEL_IDS:
            if station.has_channel(channel_id):
                LOGGER.status(f"Removing known-bad channel {channel_id} "
                                   f"from station {station.get_id()}")
                station.remove_channel(station.get_channel(channel_id))
                flagged_nrr_channel_ids[channel_id].append("reader_known_bad_channel")

            # store set of flagged nrr channel ids as station parameter
            station.set_parameter(stp.flagged_channels, flagged_nrr_channel_ids)

    def _calculate_noise_rms(self, station, Tnoise, filter_settings):
        """
        Calculate the RMS noise voltage for a given station and noise temperature.

        Parameters
        ----------
        station : NuRadioReco.framework.station.Station
            The station object for which to calculate the noise RMS.
        Tnoise : float
            The noise temperature in Kelvin.

        Returns
        -------
        float
            The calculated RMS noise voltage in Volts.
        """
        min_freq = 0
        max_freq = 0.5 * self.detector.get_sampling_frequency(station.get_id(), station.get_channel_ids()[0])
        ff = np.linspace(0, max_freq, 10000)
        filt = self.channelBandPassFilter.get_filter(
            ff, station.get_id(), None, self.detector, **filter_settings
        )
        bandwidth = np.trapz(np.abs(filt) ** 2, ff) # TODO: replace with np.trapezoid for np 2.0
        Vrms = (Tnoise * 50 * constants.k * bandwidth / units.Hz) ** 0.5
        LOGGER.info(f"Calculated Vrms: {Vrms:.2e} V for station {station.get_id()} with noise temperature {Tnoise} K")
        return Vrms, min_freq, max_freq

    def _save_trace_snapshot(self, event, station, output_dir, stage):
        """
        Save a snapshot of the traces of the first 8 channels of each station in the event.
        """

        max_channels = 8
        os.makedirs(output_dir, exist_ok=True)
        channels = list(station.iter_channels())[:max_channels]

        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        ytick_positions = []
        ytick_labels = []
        for i_ch, channel in enumerate(channels):
            trace = channel.get_trace()
            if len(trace) == 0:
                continue
            times = channel.get_times()
            scale = np.nanmax(np.abs(trace)) or 1.0
            ax.plot(times / units.ns, trace / scale + i_ch, lw=0.7, alpha=0.8)
            ytick_positions.append(i_ch)
            ytick_labels.append(str(channel.get_id()))

        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(ytick_labels)
        ax.set_xlabel("Time / ns")
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

    def _save_efield_trace_snapshot(self, event, sim_station, output_dir, stage):
        """
        Save a snapshot of the electric field traces of the first 8 channels of each sim_station in the event.
        """
        max_channels = 8
        os.makedirs(output_dir, exist_ok=True)
        efields = [e for e in sim_station.get_electric_fields()][:max_channels]

        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        ytick_positions = []
        ytick_labels = []
        pol_colors = ['r', 'b', 'g']
        for i_ch, efield in enumerate(efields):
            trace = efield.get_trace()
            efield_id = efield.get_unique_identifier()[0][0]
            if len(trace) == 0:
                continue
            times = efield.get_times()
            scale = np.nanmax(np.abs(trace)) or 1.0
            for i in range(3): # iterate over polarisation
                ax.plot(times / units.ns, trace[i, :] / scale + i_ch, lw=0.7, alpha=0.8, color=pol_colors[i])
            ytick_positions.append(i_ch)
            ytick_labels.append(str(efield_id))

        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(ytick_labels)
        ax.set_xlabel("Time / ns")
        ax.set_ylabel("Channel ID")
        ax.set_title(
            f"{stage}: CS{sim_station.get_id():03d} first {len(efields)} channels"
        )
        fig.savefig(
            os.path.join(
                output_dir,
                f"{stage}_efield_traces_CS{sim_station.get_id():03d}_{event.get_id()}.png",
            ),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)