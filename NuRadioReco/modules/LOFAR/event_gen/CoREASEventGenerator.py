"""
Simulated event generator for LOFAR.
"""

import os
import h5py
import logging
import argparse
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm as cm
from scipy import optimize as opt
from scipy import constants
from astropy import time
import datetime

import NuRadioReco.framework.event
import NuRadioReco.modules.channelGalacticNoiseAdder
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.eventTypeIdentifier

from NuRadioReco.utilities import units, trace_utilities
from NuRadioReco.detector import detector
from NuRadioReco.modules.io.coreas import coreas, coreasInterpolator, readCoREASDetector
from NuRadioReco.utilities.dataservers import download_from_dataserver
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.utilities.trace_utilities import get_electric_field_energy_fluence

from NuRadioReco.modules.LOFAR import planeWaveDirectionFitter_LOFAR  # noqa: E402
from NuRadioReco.modules.LOFAR import stationPulseFinder  # noqa: E402
from NuRadioReco.modules.LOFAR import LORASimulator
from NuRadioReco.utilities.LOFAR import (
    DEFAULT_STATIONS,
    COREAS_DIRECTORY,
    ANTENNA_RESPONSE_DIRECTORY,
    CR_SNR,
    PASS_BAND,
    START_TIME,
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
        coreas_directory=COREAS_DIRECTORY,
        antenna_response_directory=ANTENNA_RESPONSE_DIRECTORY,
        output_directory = None,
        log_level=logging.INFO,
    ):
        self.coreas_directory = coreas_directory
        self.antenna_response_directory = antenna_response_directory

        self.selected_station_channel_ids = {}
        
        self._initialise_detector(detector)
        self._initialise_modules()

        self.output_directory = output_directory
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

        self.detector.update(START_TIME)
        preprocess_LOFAR_txt(self.antenna_response_directory, orientation="Y")
        preprocess_LOFAR_txt(self.antenna_response_directory, orientation="X")
        
        for station_name in DEFAULT_STATIONS:
            staid = self.detector.get_station_id(station_name) # maybe?
            self.selected_station_channel_ids[staid] = self.detector.get_channel_ids(staid)

    def _initialise_modules(self):
        """
        Initialise all modules used in the simulation pipeline, including:

        - readCoREASDetector
        - efieldToVoltageConverter
        - channelGalacticNoiseAdder
        - channelGenericNoiseAdder
        - triggerSimulator
        - channelBandPassFilter
        - eventTypeIdentifier
        - stationPulseFinder
        - planeWaveDirectionFitter_LOFAR

        Note: only the modules are initialised, and the begin functions are called in the run_pipeline function, since some modules require event information to be initialised.
        """
        self.LORASimulator = LORASimulator()
        
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

        self.channelGalacticNoiseAdder = (
            NuRadioReco.modules.channelGalacticNoiseAdder.channelGalacticNoiseAdder()
        )

        self.channelGenericNoiseAdder = (
            NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
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



    def process_event(
        self,
        coreas_hdf5_file : str,
        sky_model : str = "LFmap",
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
        # TODO: find a good naming scheme based on the coreas filename, e.g. by using the shower id and the coreas file name
        # for now using the run number, but in principle should be connected to the event ID of the actual measured event
        sim_event_id = sim_event.get_id()

        event_debug_dir = os.path.join(self.debug_dir, sim_event_id) if save_debug_plots else None
        nur_file = f"{sim_event_id}.nur" if write_event else None

        # call all begin functions here
        self.LORASimulator.begin()
        # TODO: are these pre_pulse_time and post_pulse_time values reasonable? They are currently set to 0 ns and 400 ns, respectively, which may not be optimal for all cases.
        self.efieldToVoltageConverter.begin(
            debug=False, pre_pulse_time=0 * units.ns, post_pulse_time = 400 * units.ns
        )
        self.channelResampler.begin()
        self.channelGalacticNoiseAdder.begin(sky_model=sky_model)
        self.channelGenericNoiseAdder.begin()
        self.channelBandPassFilter.begin()
        self.channelResampler.begin()
        self.triggerSimulator.begin()
        self.eventTypeIdentifier.begin()
        self.pulse_finder.begin(cr_snr=CR_SNR, good_channels=6)
        self.direction_fitter.begin(
            debug=save_debug_plots,
            debug_plot_dir=event_debug_dir,
        )

        #TODO: LORA core simulator here
        # this should generate a core, zenith, and azimuth angle based on the LORA trigger information, and then pass it to the CoREAS reader to generate the electric field traces for the selected stations.
        core_xyz = self.LORASimulator.run(sim_event, self.detector)
        core_xy = core_xyz[:2]  # only use the x and y coordinates for the core position

        processed_event = None
        LOGGER.info(f"Processing event {sim_event.get_id()} from CoREAS file {coreas_hdf5_file}")
        for evt in self.coreas_reader.run(self.detector, [core_xy], selected_station_channel_ids=self.selected_station_channel_ids):
            
            LOGGER.info(f"Converting electric field to voltage for event {evt.get_id()} at time {START_TIME}")
            for station in evt.get_stations():
                # set the station time to the start time defined in macros.py
                station.set_station_time(START_TIME)
                if save_debug_plots:
                    self._save_trace_snapshot(evt, station, output_dir=event_debug_dir, stage="01_reader")

                # Convert electric field to voltage
                self.efieldToVoltageConverter.run(evt, station, self.detector)
                if save_debug_plots:
                    self._save_trace_snapshot(evt, station, output_dir=event_debug_dir, stage="02_efieldToVoltage")

                # resample the trace to that of LOFAR (uses the default if not provided)
                self.channelResampler.run(evt, station, self.detector)
                if save_debug_plots:
                    self._save_trace_snapshot(evt, station, output_dir=event_debug_dir, stage="03_resample")

                # Apply bandpass filter
                self.channelBandPassFilter.run(evt, station, self.detector, passband=[PASS_BAND[0] * units.MHz, PASS_BAND[1] * units.MHz], filter_type="butter", order=10)
                if save_debug_plots:
                    self._save_trace_snapshot(evt, station, output_dir=event_debug_dir, stage="04_bandpass")

                #TODO: replace this with module based on measured noise in LOFAR.
                # Add galactic noise
                self.channelGalacticNoiseAdder.run(evt, station, self.detector, sky_model=sky_model)
                if save_debug_plots:
                    self._save_trace_snapshot(evt, station, output_dir=event_debug_dir, stage="05_galactic_noise")

                # Add generic noise
                Vrms, min_freq, max_freq = self._calculate_noise_rms(station, noise_temperature)
                self.channelGenericNoiseAdder.run(evt, station, self.detector, type='rayleigh', amplitude=Vrms, min_freq=min_freq, max_freq=max_freq)
                if save_debug_plots:
                    self._save_trace_snapshot(evt, station, output_dir=event_debug_dir, stage="06_gaussian_noise")

            # TODO: understand how to treat the trigger in LOFAR. For now, we will skip this step and assume that the event is triggered if any station has a signal above the threshold.
            #     # Simulate trigger
            #     self.triggerSimulator.run(evt, station, self.detector, num_coincidences=1, threshold=CR_SNR * Vrms)

            # # now that all stations have been processed, we can check if the event is triggered
            # for station in evt.get_stations():
            #     if station.get_trigger("default_simple_threshold").has_triggered():
            #         LOGGER.info(f"Event {evt.get_id()} triggered at station {station.get_id()}")

            # Identify event type
            self.eventTypeIdentifier.run(evt, station, mode='forced', event_type='cosmic_ray')

            LOGGER.info(f"Finished processing event {evt.get_id()} from CoREAS file {coreas_hdf5_file}")
            LOGGER.info("Running pulse finding")
            # Find pulses in the stations
            self.pulse_finder.run(evt, self.detector)

            LOGGER.info("Running plane-wave direction fitting")
            # Fit direction of arrival
            self.direction_fitter.run(evt, self.detector)

            # if save_debug_plots:
            #     self._save_debug_plots(evt)
            
            # now we end all functions
            self.LORASimulator.end()
            self.efieldToVoltageConverter.end()
            self.channelResampler.end()
            self.channelGalacticNoiseAdder.end()
            self.channelGenericNoiseAdder.end()
            self.channelBandPassFilter.end()
            self.triggerSimulator.end()
            self.eventTypeIdentifier.end()
            self.pulse_finder.end()
            self.direction_fitter.end()

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

        This can be replaced with the usual begin function of readCoREASDetector when using a good / correct HDF5 file with the correct CoreCoordinateVertical. For now, we will force the vertical core coordinate to be at the observation level of the first shower in the event.

        Parameters
        ----------
        coreas_hdf5_file : str
            Path to the CoREAS HDF5 file.
        """
        evt = coreas.read_CORSIKA7(coreas_hdf5_file, site='lofar')

        # here we force vertical core coordinate
        evt.get_first_sim_shower().set_parameter(
            shp.core,
            np.array(
                [0, 0, evt.get_first_sim_shower().get_parameter(shp.observation_level)]
            ),
        )

        interpolator = coreasInterpolator.coreasInterpolator(evt)
        interpolator.initialize_efield_interpolator(
            interp_lowfreq=PASS_BAND[0] * units.MHz, interp_highfreq=PASS_BAND[1] * units.MHz
        )
        self.coreas_reader.coreas_interpolator = interpolator  # skip begin() function because HDF5 does not have good CoreCoordinateVertical
        self.coreas_reader._readCoREASDetector__corsika_evt = evt

        return evt

    def _calculate_noise_rms(self, station, Tnoise):
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
            ff, station.get_id(), None, self.detector, **self.filter_settings
        )
        bandwidth = np.trapz(np.abs(filt) ** 2, ff)
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