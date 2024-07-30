from __future__ import absolute_import, division, print_function
import numpy as np
from radiotools import helper as hp
from radiotools import coordinatesystems as cstrans
from NuRadioMC.SignalGen import askaryan
from NuRadioMC.SignalGen import emitter
from NuRadioReco.utilities import units
from NuRadioMC.utilities import medium
from NuRadioReco.utilities import fft
from NuRadioMC.utilities.earth_attenuation import get_weight
from NuRadioMC.SignalProp import propagation
from numpy.random import Generator, Philox
import h5py
import time
import six
import copy
import json
from scipy import constants
# import detector simulation modules
import NuRadioReco.modules.io.eventWriter
import NuRadioReco.modules.channelSignalReconstructor
import NuRadioReco.modules.electricFieldResampler
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.efieldToVoltageConverterPerEfield
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.channelAddCableDelay
import NuRadioReco.modules.channelResampler
import NuRadioReco.detector.detector as detector
import NuRadioReco.framework.sim_station
import NuRadioReco.framework.electric_field
import NuRadioReco.framework.particle
import NuRadioReco.framework.event
import NuRadioReco.framework.sim_emitter
from NuRadioReco.detector import antennapattern
from NuRadioReco.utilities import geometryUtilities as geo_utl
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import showerParameters as shp
# parameters describing simulated Monte Carlo particles
from NuRadioReco.framework.parameters import particleParameters as simp
from NuRadioReco.framework.parameters import emitterParameters as ep
# parameters set in the event generator
from NuRadioReco.framework.parameters import generatorAttributes as genattrs
import datetime
import logging
import warnings
from six import iteritems
import yaml
import os
import collections
from NuRadioMC.utilities.Veff import remove_duplicate_triggers
import logging
from NuRadioReco.utilities.logging import LOGGING_STATUS, setup_logger

logger = setup_logger("NuRadioMC.simulation")


def pretty_time_delta(seconds):
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return '%dd%dh%dm%ds' % (days, hours, minutes, seconds)
    elif hours > 0:
        return '%dh%dm%ds' % (hours, minutes, seconds)
    elif minutes > 0:
        return '%dm%ds' % (minutes, seconds)
    else:
        return '%ds' % (seconds,)


def merge_config(user, default):
    if isinstance(user, dict) and isinstance(default, dict):
        for k, v in iteritems(default):
            if k not in user:
                user[k] = v
            else:
                user[k] = merge_config(user[k], v)
    return user


class simulation:

    def __init__(self, inputfilename,
                 outputfilename,
                 detectorfile=None,
                 det=None,
                 det_kwargs={},
                 outputfilenameNuRadioReco=None,
                 debug=False,
                 evt_time=datetime.datetime(2018, 1, 1),
                 config_file=None,
                 log_level=LOGGING_STATUS,
                 default_detector_station=None,
                 default_detector_channel=None,
                 file_overwrite=False,
                 write_detector=True,
                 event_list=None,
                 log_level_propagation=logging.WARNING,
                 ice_model=None,
                 **kwargs):
        """
        initialize the NuRadioMC end-to-end simulation

        Parameters
        ----------
        inputfilename: string, or pair
            the path to the hdf5 file containing the list of neutrino events
            alternatively, the data and attributes dictionary can be passed directly to the method
        outputfilename: string
            specify hdf5 output filename.
        detectorfile: string
            path to the json file containing the detector description
        det: detector object
            Pass a detector class object
        det_kwargs: dict
            Pass arguments to the detector (only used when det == None)
        station_id: int
            the station id for which the simulation is performed. Must match a station
            deself._fined in the detector description
        outputfilenameNuRadioReco: string or None
            outputfilename of NuRadioReco detector sim file, this file contains all
            waveforms of the triggered events
            default: None, i.e., no output file will be written which is useful for
            effective volume calculations
        debug: bool
            True activates debug mode, default False
        evt_time: datetime object
            the time of the events, default 1/1/2018
        config_file: string
            path to config file
        log_level: logging.LEVEL
            the log level
        default_detector_station: int or None
            DEPRECATED: Define reference stations in the detector JSON file instead
        default_detector_channel: int or None
            DEPRECATED: Define reference channels in the detector JSON file instead
        file_overwrite: bool
            True allows overwriting of existing files, default False
        write_detector: bool
            If true, the detector description is written into the .nur files along with the events
            default True
        event_list: None or list of ints
            if provided, only the event listed in this list are being simulated
        log_level_propagation: logging.LEVEL
            the log level of the propagation module
        ice_model: medium object (default None)
            allows to specify a custom ice model. This model is used if the config file specifies the ice model as "custom".
        """
        logger.setLevel(log_level)
        if 'write_mode' in kwargs:
            logger.warning('Parameter write_mode is deprecated. Define the output format in the config file instead.')

        self._log_level = log_level
        self._log_level_ray_propagation = log_level_propagation
        config_file_default = os.path.join(os.path.dirname(__file__), 'config_default.yaml')
        logger.status('reading default config from {}'.format(config_file_default))
        with open(config_file_default, 'r') as ymlfile:
            self._cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
        if config_file is not None:
            logger.status('reading local config overrides from {}'.format(config_file))
            with open(config_file, 'r') as ymlfile:
                local_config = yaml.load(ymlfile, Loader=yaml.FullLoader)
                new_cfg = merge_config(local_config, self._cfg)
                self._cfg = new_cfg

        if self._cfg['seed'] is None:
            # the config seeting None means a random seed. To have the simulation be reproducable, we generate a new
            # random seed once and save this seed to the config setting. If the simulation is rerun, we can get
            # the same random sequence.
            self._cfg['seed'] = np.random.randint(0, 2 ** 32 - 1)

        self._rnd = Generator(Philox(self._cfg['seed']))

        self._outputfilename = outputfilename
        if os.path.exists(self._outputfilename):
            msg = f"hdf5 output file {self._outputfilename} already exists"
            if file_overwrite == False:
                logger.error(msg)
                raise FileExistsError(msg)
            else:
                logger.warning(msg)
        self._detectorfile = detectorfile
        self._n_reflections = int(self._cfg['propagation']['n_reflections'])
        self._outputfilenameNuRadioReco = outputfilenameNuRadioReco
        self._debug = debug
        self._evt_time = evt_time
        self.__write_detector = write_detector
        logger.status("setting event time to {}".format(evt_time))
        self._event_group_list = event_list

        self._antenna_pattern_provider = antennapattern.AntennaPatternProvider()

        # initialize propagation module
        self._prop = propagation.get_propagation_module(self._cfg['propagation']['module'])

        if self._cfg['propagation']['ice_model'] == "custom":
            if ice_model is None:
                logger.error("ice model is set to 'custom' in config file but no custom ice model is provided.")
                raise AttributeError("ice model is set to 'custom' in config file but no custom ice model is provided.")
            self._ice = ice_model
        else:
            self._ice = medium.get_ice_model(self._cfg['propagation']['ice_model'])

        self._mout = collections.OrderedDict()
        self._mout_groups = collections.OrderedDict()
        self._mout_attrs = collections.OrderedDict()

        # Initialize detector
        if det is None:
            logger.status("Detectorfile {}".format(os.path.abspath(self._detectorfile)))
            kwargs = dict(json_filename=self._detectorfile, default_station=default_detector_station,
                              default_channel=default_detector_channel, antenna_by_depth=False)
            kwargs.update(det_kwargs)
            self._det = detector.Detector(**kwargs)
        else:
            self._det = det

        self._det.update(evt_time)

        self._station_ids = self._det.get_station_ids()
        self._event_ids_counter = {}
        for station_id in self._station_ids:
            self._event_ids_counter[station_id] = -1  # we initialize with -1 becaue we increment the counter before we use it the first time

        # print noise information
        logger.status("running with noise {}".format(bool(self._cfg['noise'])))
        logger.status("setting signal to zero {}".format(bool(self._cfg['signal']['zerosignal'])))
        if bool(self._cfg['propagation']['focusing']):
            logger.status("simulating signal amplification due to focusing of ray paths in the firn.")

        # read sampling rate from config (this sampling rate will be used internally)
        self._dt = 1. / (self._cfg['sampling_rate'] * units.GHz)

        if isinstance(inputfilename, str):
            logger.status(f"reading input from {inputfilename}")
            self._inputfilename = inputfilename
            self._read_input_hdf5()  # we read in the full input file into memory at the beginning to limit io to the beginning and end of the run
        else:
            logger.status("getting input on-the-fly")
            self._inputfilename = "on-the-fly"
            self._fin = inputfilename[0]
            self._fin_attrs = inputfilename[1]
            self._fin_stations = {}
        # store all relevant attributes of the input file in a dictionary
        self._generator_info = {}
        for enum_entry in genattrs:
            if enum_entry.name in self._fin_attrs:
                self._generator_info[enum_entry] = self._fin_attrs[enum_entry.name]

        # check if the input file contains events, if not save empty output file (for book keeping) and terminate simulation
        if len(self._fin['xx']) == 0:
            logger.status(f"input file {self._inputfilename} is empty")
            return

        # Perfom a dummy detector simulation to determine how the signals are filtered.
        # This variable stores the integrated channel response for each channel, i.e.
        # the integral of the squared signal chain response over all frequencies, int S21^2 df.
        # For a system without amplification, it is equivalent to the bandwidth of the system.
        self._integrated_channel_response = {}

        self._integrated_channel_response_normalization = {}
        self._max_amplification_per_channel = {}
        self.__noise_adder_normalization = {}

        # first create dummy event and station with channels
        self._Vrms = 1
        for iSt, self._station_id in enumerate(self._station_ids):
            self._shower_index = 0
            self._primary_index = 0
            self._evt = NuRadioReco.framework.event.Event(0, self._primary_index)

            self._sampling_rate_detector = self._det.get_sampling_frequency(self._station_id, 0)
#                 logger.warning('internal sampling rate is {:.3g}GHz, final detector sampling rate is {:.3g}GHz'.format(self.get_sampling_rate(), self._sampling_rate_detector))
            self._n_samples = self._det.get_number_of_samples(self._station_id, 0) / self._sampling_rate_detector / self._dt
            self._n_samples = int(np.ceil(self._n_samples / 2.) * 2)  # round to nearest even integer
            self._ff = np.fft.rfftfreq(self._n_samples, self._dt)
            self._tt = np.arange(0, self._n_samples * self._dt, self._dt)

            self._create_sim_station()
            for channel_id in range(self._det.get_number_of_channels(self._station_id)):
                electric_field = NuRadioReco.framework.electric_field.ElectricField([channel_id], self._det.get_relative_position(self._sim_station.get_id(), channel_id))
                trace = np.zeros_like(self._tt)
                trace[self._n_samples // 2] = 100 * units.V  # set a signal that will satisfy any high/low trigger
                trace[self._n_samples // 2 + 1] = -100 * units.V
                electric_field.set_trace(np.array([np.zeros_like(self._tt), trace, trace]), 1. / self._dt)
                electric_field.set_trace_start_time(0)
                electric_field[efp.azimuth] = 0
                electric_field[efp.zenith] = 100 * units.deg
                electric_field[efp.ray_path_type] = 0
                self._sim_station.add_electric_field(electric_field)

            self._station = NuRadioReco.framework.station.Station(self._station_id)
            self._station.set_sim_station(self._sim_station)
            self._station.set_station_time(self._evt_time)
            self._evt.set_station(self._station)

            # run detector simulation
            self._detector_simulation_filter_amp(self._evt, self._station, self._det)

            self._integrated_channel_response[self._station_id] = {}
            self._integrated_channel_response_normalization[self._station_id] = {}
            self._max_amplification_per_channel[self._station_id] = {}

            for channel_id in range(self._det.get_number_of_channels(self._station_id)):
                ff = np.linspace(0, 0.5 / self._dt, 10000)
                filt = np.ones_like(ff, dtype=complex)
                for i, (name, instance, kwargs) in enumerate(self._evt.iter_modules(self._station_id)):
                    if hasattr(instance, "get_filter"):
                        filt *= instance.get_filter(ff, self._station_id, channel_id, self._det, **kwargs)

                self._max_amplification_per_channel[self._station_id][channel_id] = np.abs(filt).max()

                mean_integrated_response = np.mean(
                    np.abs(filt)[np.abs(filt) > np.abs(filt).max() / 100] ** 2)  # a factor of 100 corresponds to -40 dB in amplitude
                self._integrated_channel_response_normalization[self._station_id][channel_id] = mean_integrated_response

                integrated_channel_response = np.trapz(np.abs(filt) ** 2, ff)
                self._integrated_channel_response[self._station_id][channel_id] = integrated_channel_response

                logger.debug(f"Station.channel {self._station_id}.{channel_id} estimated bandwidth is "
                             f"{integrated_channel_response / mean_integrated_response / units.MHz:.1f} MHz")

        ################################

        self._bandwidth = next(iter(next(iter(self._integrated_channel_response.values())).values()))  # get value of first station/channel key pair
        norm = next(iter(next(iter(self._integrated_channel_response_normalization.values())).values()))
        amplification = next(iter(next(iter(self._max_amplification_per_channel.values())).values()))

        noise_temp = self._cfg['trigger']['noise_temperature']
        Vrms = self._cfg['trigger']['Vrms']

        if noise_temp is not None and Vrms is not None:
            raise AttributeError(f"Specifying noise temperature (set to {noise_temp}) and Vrms (set to {Vrms}) is not allowed).")

        self._Vrms_per_channel = collections.defaultdict(dict)
        self._Vrms_efield_per_channel = collections.defaultdict(dict)

        if noise_temp is not None:

            if noise_temp == "detector":
                logger.status("Use noise temperature from detector description to determine noise Vrms in each channel.")
                self._noise_temp = None  # the noise temperature is defined in the detector description
            else:
                self._noise_temp = float(noise_temp)
                logger.status(f"Use a noise temperature of {noise_temp / units.kelvin:.1f} K for each channel to determine noise Vrms.")

            self._noiseless_channels = collections.defaultdict(list)
            for station_id in self._integrated_channel_response:
                for channel_id in self._integrated_channel_response[station_id]:
                    if self._noise_temp is None:
                        noise_temp_channel = self._det.get_noise_temperature(station_id, channel_id)
                    else:
                        noise_temp_channel = self._noise_temp

                    if self._det.is_channel_noiseless(station_id, channel_id):
                        self._noiseless_channels[station_id].append(channel_id)

                    # Calculation of Vrms. For details see from elog:1566 and https://en.wikipedia.org/wiki/Johnson%E2%80%93Nyquist_noise
                    # (last two Eqs. in "noise voltage and power" section) or our wiki https://nu-radio.github.io/NuRadioMC/NuRadioMC/pages/HDF5_structure.html

                    # Bandwidth, i.e., \Delta f in equation
                    integrated_channel_response = self._integrated_channel_response[station_id][channel_id]
                    max_amplification = self._max_amplification_per_channel[station_id][channel_id]

                    self._Vrms_per_channel[station_id][channel_id] = (noise_temp_channel * 50 * constants.k * integrated_channel_response / units.Hz) ** 0.5
                    self._Vrms_efield_per_channel[station_id][channel_id] = self._Vrms_per_channel[station_id][channel_id] / max_amplification / units.m  # VEL = 1m

                    # for logging
                    mean_integrated_response = self._integrated_channel_response_normalization[self._station_id][channel_id]

                    logger.status(f'Station.channel {station_id}.{channel_id:02d}: noise temperature = {noise_temp_channel} K, '
                                  f'est. bandwidth = {integrated_channel_response / mean_integrated_response / units.MHz:.2f} MHz, '
                                  f'max. filter amplification = {max_amplification:.2e} -> Vrms = '
                                  f'integrated response = {integrated_channel_response / units.MHz:.2e}MHz -> Vrms = '
                                  f'{self._Vrms_per_channel[station_id][channel_id] / units.mV:.2f} mV -> efield Vrms = {self._Vrms_efield_per_channel[station_id][channel_id] / units.V / units.m / units.micro:.2f}muV/m (assuming VEL = 1m) ')

            self._Vrms = next(iter(next(iter(self._Vrms_per_channel.values())).values()))

        elif Vrms is not None:
            self._Vrms = float(Vrms) * units.V
            self._noise_temp = None
            logger.status(f"Use a fix noise Vrms of {self._Vrms / units.mV:.2f} mV in each channel.")

            for station_id in self._integrated_channel_response:

                for channel_id in self._integrated_channel_response[station_id]:
                    max_amplification = self._max_amplification_per_channel[station_id][channel_id]
                    self._Vrms_per_channel[station_id][channel_id] = self._Vrms  # to be stored in the hdf5 file
                    self._Vrms_efield_per_channel[station_id][channel_id] = self._Vrms / max_amplification / units.m  # VEL = 1m

                    # for logging
                    integrated_channel_response = self._integrated_channel_response[station_id][channel_id]
                    mean_integrated_response = self._integrated_channel_response_normalization[self._station_id][channel_id]

                    logger.status(f'Station.channel {station_id}.{channel_id:02d}: '
                                  f'est. bandwidth = {integrated_channel_response / mean_integrated_response / units.MHz:.2f} MHz, '
                                  f'max. filter amplification = {max_amplification:.2e} '
                                  f'integrated response = {integrated_channel_response / units.MHz:.2e}MHz ->'
                                  f'efield Vrms = {self._Vrms_efield_per_channel[station_id][channel_id] / units.V / units.m / units.micro:.2f}muV/m (assuming VEL = 1m) ')

        else:
            raise AttributeError(f"noise temperature and Vrms are both set to None")

        self._Vrms_efield = next(iter(next(iter(self._Vrms_efield_per_channel.values())).values()))
        speed_cut = float(self._cfg['speedup']['min_efield_amplitude'])
        logger.status(f"All signals with less then {speed_cut:.1f} x Vrms_efield will be skipped.")

        self._distance_cut_polynomial = None
        if self._cfg['speedup']['distance_cut']:
            coef = self._cfg['speedup']['distance_cut_coefficients']
            self.__distance_cut_polynomial = np.polynomial.polynomial.Polynomial(coef)

            def get_distance_cut(shower_energy):
                if shower_energy <= 0:
                    return 100 * units.m
                return max(100 * units.m, 10 ** self.__distance_cut_polynomial(np.log10(shower_energy)))

            self._get_distance_cut = get_distance_cut

    def run(self):
        """
        run the NuRadioMC simulation
        """
        if len(self._fin['xx']) == 0:
            logger.status(f"writing empty hdf5 output file")
            self._write_output_file(empty=True)
            logger.status(f"terminating simulation")
            return 0
        logger.status(f"Starting NuRadioMC simulation")
        t_start = time.time()
        t_last_update = t_start

        self._channelSignalReconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
        self._eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
        efieldToVoltageConverterPerEfield = NuRadioReco.modules.efieldToVoltageConverterPerEfield.efieldToVoltageConverterPerEfield()
        efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
        efieldToVoltageConverter.begin(time_resolution=self._cfg['speedup']['time_res_efieldconverter'])
        channelAddCableDelay = NuRadioReco.modules.channelAddCableDelay.channelAddCableDelay()
        channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
        channelGenericNoiseAdder.begin(seed=self._cfg['seed'])
        channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
        electricFieldResampler = NuRadioReco.modules.electricFieldResampler.electricFieldResampler()
        if self._outputfilenameNuRadioReco is not None:
            self._eventWriter.begin(self._outputfilenameNuRadioReco, log_level=self._log_level)
        unique_event_group_ids = np.unique(self._fin['event_group_ids'])
        self._n_showers = len(self._fin['event_group_ids'])
        self._shower_ids = np.array(self._fin['shower_ids'])
        self._shower_index_array = {}  # this array allows to convert the shower id to an index that starts from 0 to be used to access the arrays in the hdf5 file.

        self._raytracer = self._prop(
            self._ice, self._cfg['propagation']['attenuation_model'],
            log_level=self._log_level_ray_propagation,
            n_frequencies_integration=int(self._cfg['propagation']['n_freq']),
            n_reflections=self._n_reflections,
            config=self._cfg,
            detector=self._det
        )
        for shower_index, shower_id in enumerate(self._shower_ids):
            self._shower_index_array[shower_id] = shower_index

        self._create_meta_output_datastructures()

        # check if the same detector was simulated before (then we can save the ray tracing part)
        pre_simulated = self._check_if_was_pre_simulated()

        # Check if vertex_times exists:
        self._check_vertex_times()

        input_time = 0.0
        askaryan_time = 0.0
        rayTracingTime = 0.0
        detSimTime = 0.0
        outputTime = 0.0
        weightTime = 0.0
        distance_cut_time = 0.0

        n_shower_station = len(self._station_ids) * self._n_showers
        iCounter = 0

        # calculate bary centers of station
        self._station_barycenter = np.zeros((len(self._station_ids), 3))
        for iSt, station_id in enumerate(self._station_ids):
            pos = []
            for channel_id in range(self._det.get_number_of_channels(station_id)):
                pos.append(self._det.get_relative_position(station_id, channel_id))
            self._station_barycenter[iSt] = np.mean(np.array(pos), axis=0) + self._det.get_absolute_position(station_id)

        # loop over event groups
        for i_event_group_id, event_group_id in enumerate(unique_event_group_ids):
            logger.debug(f"simulating event group id {event_group_id}")
            if self._event_group_list is not None and event_group_id not in self._event_group_list:
                logger.debug(f"skipping event group {event_group_id} because it is not in the event group list provided to the __init__ function")
                continue
            event_indices = np.atleast_1d(np.squeeze(np.argwhere(self._fin['event_group_ids'] == event_group_id)))

            # the weight calculation is independent of the station, so we do this calculation only once
            # the weight also depends just on the "mother" particle, i.e. the incident neutrino which determines
            # the propability of arriving at our simulation volume. All subsequent showers have the same weight. So
            # we calculate it just once and save it to all subshowers.
            t1 = time.time()

            self._primary_index = event_indices[0]
            # determine if a particle (neutrinos, or a secondary interaction of a neutrino, or surfaec muons) is simulated
            particle_mode = "simulation_mode" not in self._fin_attrs or self._fin_attrs['simulation_mode'] != "emitter"
            self._mout['weights'][event_indices] = np.ones(len(event_indices))  # for a pulser simulation, every event has the same weight
            if particle_mode:
                self._read_input_particle_properties(self._primary_index)  # this sets the self.input_particle for self._primary_index
                # calculate the weight for the primary particle
                self.primary = self.input_particle
                if self._cfg['weights']['weight_mode'] == "existing":
                    if "weights" in self._fin:
                        self._mout['weights'] = self._fin["weights"]
                    else:
                        logger.error("config file specifies to use weights from the input hdf5 file but the input file does not contain this information.")
                elif self._cfg['weights']['weight_mode'] is None:
                    self.primary[simp.weight] = 1.
                else:
                    self.primary[simp.weight] = get_weight(self.primary[simp.zenith],
                                                           self.primary[simp.energy],
                                                           self.primary[simp.flavor],
                                                           mode=self._cfg['weights']['weight_mode'],
                                                           cross_section_type=self._cfg['weights']['cross_section_type'],
                                                           vertex_position=self.primary[simp.vertex],
                                                           phi_nu=self.primary[simp.azimuth])
                # all entries for the event for this primary get the calculated primary's weight
                self._mout['weights'][event_indices] = self.primary[simp.weight]

            weightTime += time.time() - t1
            # skip all events where neutrino weights is zero, i.e., do not
            # simulate neutrino that propagate through the Earth
            if self._mout['weights'][self._primary_index] < self._cfg['speedup']['minimum_weight_cut']:
                logger.debug("neutrino weight is smaller than {}, skipping event".format(self._cfg['speedup']['minimum_weight_cut']))
                continue

            # these quantities get computed to apply the distance cut as a function of shower energies
            # the shower energies of closeby showers will be added as they can constructively interfere
            if self._cfg['speedup']['distance_cut']:
                t_tmp = time.time()
                shower_energies = np.array(self._fin['shower_energies'])[event_indices]
                vertex_positions = np.array([np.array(self._fin['xx'])[event_indices],
                                             np.array(self._fin['yy'])[event_indices],
                                             np.array(self._fin['zz'])[event_indices]]).T
                vertex_distances = np.linalg.norm(vertex_positions - vertex_positions[0], axis=1)
                distance_cut_time += time.time() - t_tmp

            triggered_showers = {}  # this variable tracks which showers triggered a particular station

            # loop over all stations (each station is treated independently)
            for iSt, self._station_id in enumerate(self._station_ids):
                t1 = time.time()
                triggered_showers[self._station_id] = []
                logger.debug(f"simulating station {self._station_id}")

                if self._cfg['speedup']['distance_cut']:
                    # perform a quick cut to reject event group completely if no shower is close enough to the station
                    t_tmp = time.time()
                    vertex_distances_to_station = np.linalg.norm(vertex_positions - self._station_barycenter[iSt], axis=1)
                    distance_cut = self._get_distance_cut(np.sum(shower_energies)) + 100 * units.m  # 100m safety margin is added to account for extent of station around bary center.
                    if vertex_distances_to_station.min() > distance_cut:
                        logger.debug(f"skipping station {self._station_id} because minimal distance {vertex_distances_to_station.min()/units.km:.1f}km > {distance_cut/units.km:.1f}km (shower energy = {shower_energies.max():.2g}eV) bary center of station {self._station_barycenter[iSt]}")
                        distance_cut_time += time.time() - t_tmp
                        iCounter += len(shower_energies)
                        continue
                    distance_cut_time += time.time() - t_tmp

                candidate_station = False
                self._sampling_rate_detector = self._det.get_sampling_frequency(self._station_id, 0)
#                 logger.warning('internal sampling rate is {:.3g}GHz, final detector sampling rate is {:.3g}GHz'.format(self.get_sampling_rate(), self._sampling_rate_detector))
                self._n_samples = self._det.get_number_of_samples(self._station_id, 0) / self._sampling_rate_detector / self._dt
                self._n_samples = int(np.ceil(self._n_samples / 2.) * 2)  # round to nearest even integer
                self._ff = np.fft.rfftfreq(self._n_samples, self._dt)
                self._tt = np.arange(0, self._n_samples * self._dt, self._dt)

                ray_tracing_performed = False
                if 'station_{:d}'.format(self._station_id) in self._fin_stations:
                    ray_tracing_performed = (self._raytracer.get_output_parameters()[0]['name'] in self._fin_stations['station_{:d}'.format(self._station_id)]) and self._was_pre_simulated
                self._evt_tmp = NuRadioReco.framework.event.Event(0, 0)

                if particle_mode:
                    # add the primary particle to the temporary event
                    self._evt_tmp.add_particle(self.primary)

                self._create_sim_station()
                # loop over all showers in event group
                # create output data structure for this channel
                sg = self._create_station_output_structure(len(event_indices), self._det.get_number_of_channels(self._station_id))
                for iSh, self._shower_index in enumerate(event_indices):
                    sg['shower_id'][iSh] = self._shower_ids[self._shower_index]
                    iCounter += 1
#                     if(iCounter % max(1, int(n_shower_station / 100.)) == 0):
                    if (time.time() - t_last_update) > 60 :
                        t_last_update = time.time()
                        eta = pretty_time_delta((time.time() - t_start) * (n_shower_station - iCounter) / iCounter)
                        total_time_sum = input_time + rayTracingTime + detSimTime + outputTime + weightTime + distance_cut_time  # askaryan time is part of the ray tracing time, so it is not counted here.
                        total_time = time.time() - t_start
                        tmp_att = 0
                        if total_time > 0:
                            logger.status(
                                "processing event group {}/{} and shower {}/{} ({} showers triggered) = {:.1f}%, ETA {}, time consumption: ray tracing = {:.0f}%, askaryan = {:.0f}%, detector simulation = {:.0f}% reading input = {:.0f}%, calculating weights = {:.0f}%, distance cut {:.0f}%, unaccounted = {:.0f}% ".format(
                                    i_event_group_id,
                                    len(unique_event_group_ids),
                                    iCounter,
                                    n_shower_station,
                                    np.sum(self._mout['triggered']),
                                    100. * iCounter / n_shower_station,
                                    eta,
                                    100. * (rayTracingTime - askaryan_time) / total_time,
                                    100. * askaryan_time / total_time,
                                    100. * detSimTime / total_time,
                                    100.*input_time / total_time,
                                    100. * weightTime / total_time,
                                    100 * distance_cut_time / total_time,
                                    100 * (total_time - total_time_sum) / total_time))

                    self._read_input_shower_properties()
                    if particle_mode:
                        logger.debug(f"simulating shower {self._shower_index}: {self._fin['shower_type'][self._shower_index]} with E = {self._fin['shower_energies'][self._shower_index]/units.eV:.2g}eV")
                    x1 = self._shower_vertex  # the interaction point

                    if self._cfg['speedup']['distance_cut']:
                        t_tmp = time.time()
                        # calculate the sum of shower energies for all showers within self._cfg['speedup']['distance_cut_sum_length']
                        mask_shower_sum = np.abs(vertex_distances - vertex_distances[iSh]) < self._cfg['speedup']['distance_cut_sum_length']
                        shower_energy_sum = np.sum(shower_energies[mask_shower_sum])
                        # quick speedup cut using barycenter of station as position
                        distance_to_station = np.linalg.norm(x1 - self._station_barycenter[iSt])
                        distance_cut = self._get_distance_cut(shower_energy_sum) + 100 * units.m  # 100m safety margin is added to account for extent of station around bary center.
                        logger.debug(f"calculating distance cut. Current event has energy {self._fin['shower_energies'][self._shower_index]:.4g}, it is event number {iSh} and {np.sum(mask_shower_sum)} are within {self._cfg['speedup']['distance_cut_sum_length']/units.m:.1f}m -> {shower_energy_sum:.4g}")
                        if distance_to_station > distance_cut:
                            logger.debug(f"skipping station {self._station_id} because distance {distance_to_station/units.km:.1f}km > {distance_cut/units.km:.1f}km (shower energy = {self._fin['shower_energies'][self._shower_index]:.2g}eV) between vertex {x1} and bary center of station {self._station_barycenter[iSt]}")
                            distance_cut_time += time.time() - t_tmp
                            continue
                        distance_cut_time += time.time() - t_tmp

                    # skip vertices not in fiducial volume. This is required because 'mother' events are added to the event list
                    # if daugthers (e.g. tau decay) have their vertex in the fiducial volume
                    if not self._is_in_fiducial_volume(self._shower_vertex):
                        logger.debug(f"event is not in fiducial volume, skipping simulation {self._fin['xx'][self._shower_index]}, "
                                     f"{self._fin['yy'][self._shower_index]}, {self._fin['zz'][self._shower_index]}")
                        continue

                    # for special cases where only EM or HAD showers are simulated, skip all events that don't fulfill this criterion
                    if self._cfg['signal']['shower_type'] == "em":
                        if self._fin['shower_type'][self._shower_index] != "em":
                            continue
                    if self._cfg['signal']['shower_type'] == "had":
                        if self._fin['shower_type'][self._shower_index] != "had":
                            continue

                    if particle_mode:
                        self._create_sim_shower()  # create sim shower
                        self._evt_tmp.add_sim_shower(self._sim_shower)
                    else:
                        emitter_obj = NuRadioReco.framework.sim_emitter.SimEmitter(self._shower_index)  # shower_id is equivalent to emitter_id in this case
                        emitter_obj[ep.position] = np.array([self._fin['xx'][self._primary_index], self._fin['yy'][self._primary_index], self._fin['zz'][self._primary_index]])
                        emitter_obj[ep.model] = self._fin['emitter_model'][self._primary_index]
                        emitter_obj[ep.amplitude] = self._fin['emitter_amplitudes'][self._primary_index]
                        for key in ep:
                            if not emitter_obj.has_parameter(key):
                                if 'emitter_' + key.name in self._fin:
                                    emitter_obj[key] = self._fin['emitter_' + key.name][self._primary_index]
                        self._evt_tmp.add_sim_emitter(emitter_obj)

                    # generate unique and increasing event id per station
                    self._event_ids_counter[self._station_id] += 1
                    self._event_id = self._event_ids_counter[self._station_id]

                    # be careful, zenith/azimuth angle always refer to where the neutrino came from,
                    # i.e., opposite to the direction of propagation. We need the propagation direction here,
                    # so we multiply the shower axis with '-1'
                    if 'zeniths' in self._fin:
                        self._shower_axis = -1 * hp.spherical_to_cartesian(self._fin['zeniths'][self._shower_index], self._fin['azimuths'][self._shower_index])
                    else:
                        self._shower_axis = np.array([0, 0, 1])

                    # calculate correct Cherenkov angle for ice density at vertex position
                    n_index = self._ice.get_index_of_refraction(x1)
                    cherenkov_angle = np.arccos(1. / n_index)

                    # first step: perform raytracing to see if solution exists
                    t2 = time.time()
#                     input_time += (time.time() - t1)

                    for channel_id in range(self._det.get_number_of_channels(self._station_id)):
                        x2 = self._det.get_relative_position(self._station_id, channel_id) + self._det.get_absolute_position(self._station_id)
                        logger.debug(f"simulating channel {channel_id} at {x2}")

                        if self._cfg['speedup']['distance_cut']:
                            t_tmp = time.time()
                            distance_cut = self._get_distance_cut(shower_energy_sum)
                            distance = np.linalg.norm(x1 - x2)

                            if distance > distance_cut:
                                logger.debug('A distance speed up cut has been applied')
                                logger.debug('Shower energy: {:.2e} eV'.format(self._fin['shower_energies'][self._shower_index] / units.eV))
                                logger.debug('Distance cut: {:.2f} m'.format(distance_cut / units.m))
                                logger.debug('Distance to vertex: {:.2f} m'.format(distance / units.m))
                                distance_cut_time += time.time() - t_tmp
                                continue
                            distance_cut_time += time.time() - t_tmp

                        self._raytracer.set_start_and_end_point(x1, x2)
                        self._raytracer.use_optional_function('set_shower_axis', self._shower_axis)
                        if pre_simulated and ray_tracing_performed and not self._cfg['speedup']['redo_raytracing']:  # check if raytracing was already performed
                            if self._cfg['propagation']['module'] == 'radiopropa':
                                logger.error('Presimulation can not be used with the radiopropa ray tracer module')
                                raise Exception('Presimulation can not be used with the radiopropa ray tracer module')
                            sg_pre = self._fin_stations["station_{:d}".format(self._station_id)]
                            ray_tracing_solution = {}
                            for output_parameter in self._raytracer.get_output_parameters():
                                ray_tracing_solution[output_parameter['name']] = sg_pre[output_parameter['name']][self._shower_index, channel_id]
                            self._raytracer.set_solution(ray_tracing_solution)
                        else:
                            self._raytracer.find_solutions()

                        if not self._raytracer.has_solution():
                            logger.debug("event {} and station {}, channel {} does not have any ray tracing solution ({} to {})".format(
                                self._event_group_id, self._station_id, channel_id, x1, x2))
                            continue
                        delta_Cs = []
                        viewing_angles = []
                        # loop through all ray tracing solution
                        for iS in range(self._raytracer.get_number_of_solutions()):
                            for key, value in self._raytracer.get_raytracing_output(iS).items():
                                sg[key][iSh, channel_id, iS] = value
                            self._launch_vector = self._raytracer.get_launch_vector(iS)
                            sg['launch_vectors'][iSh, channel_id, iS] = self._launch_vector
                            # calculates angle between shower axis and launch vector
                            viewing_angle = hp.get_angle(self._shower_axis, self._launch_vector)
                            viewing_angles.append(viewing_angle)
                            delta_C = (viewing_angle - cherenkov_angle)
                            logger.debug('solution {} {}: viewing angle {:.1f} = delta_C = {:.1f}'.format(
                                iS, propagation.solution_types[self._raytracer.get_solution_type(iS)], viewing_angle / units.deg, (viewing_angle - cherenkov_angle) / units.deg))
                            delta_Cs.append(delta_C)

                        # discard event if delta_C (angle off cherenkov cone) is too large
                        if min(np.abs(delta_Cs)) > self._cfg['speedup']['delta_C_cut']:
                            logger.debug('delta_C too large, event unlikely to be observed, skipping event')
                            continue

                        n = self._raytracer.get_number_of_solutions()
                        for iS in range(n):  # loop through all ray tracing solution
                            # skip individual channels where the viewing angle difference is too large
                            # discard event if delta_C (angle off cherenkov cone) is too large
                            if np.abs(delta_Cs[iS]) > self._cfg['speedup']['delta_C_cut']:
                                logger.debug('delta_C too large, ray tracing solution unlikely to be observed, skipping event')
                                continue
                            if pre_simulated and ray_tracing_performed and not self._cfg['speedup']['redo_raytracing']:
                                sg_pre = self._fin_stations["station_{:d}".format(self._station_id)]
                                R = sg_pre['travel_distances'][self._shower_index, channel_id, iS]
                                T = sg_pre['travel_times'][self._shower_index, channel_id, iS]
                            else:
                                R = self._raytracer.get_path_length(iS)  # calculate path length
                                T = self._raytracer.get_travel_time(iS)  # calculate travel time
                                if R is None or T is None:
                                    continue
                            sg['travel_distances'][iSh, channel_id, iS] = R
                            sg['travel_times'][iSh, channel_id, iS] = T
                            self._launch_vector = self._raytracer.get_launch_vector(iS)
                            receive_vector = self._raytracer.get_receive_vector(iS)
                            # save receive vector
                            sg['receive_vectors'][iSh, channel_id, iS] = receive_vector
                            zenith, azimuth = hp.cartesian_to_spherical(*receive_vector)

                            # get neutrino pulse from Askaryan module
                            t_ask = time.time()

                            if "simulation_mode" not in self._fin_attrs or self._fin_attrs['simulation_mode'] == "neutrino":
                                # first consider in-ice showers
                                kwargs = {}
                                # if the input file specifies a specific shower realization, use that realization
                                if self._cfg['signal']['model'] in ["ARZ2019", "ARZ2020"] and "shower_realization_ARZ" in self._fin:
                                    kwargs['iN'] = self._fin['shower_realization_ARZ'][self._shower_index]
                                    logger.debug(f"reusing shower {kwargs['iN']} ARZ shower library")
                                elif self._cfg['signal']['model'] == "Alvarez2009" and "shower_realization_Alvarez2009" in self._fin:
                                    kwargs['k_L'] = self._fin['shower_realization_Alvarez2009'][self._shower_index]
                                    logger.debug(f"reusing k_L parameter of Alvarez2009 model of k_L = {kwargs['k_L']:.4g}")
                                else:
                                    # check if the shower was already simulated (e.g. for a different channel or ray tracing solution)
                                    if self._cfg['signal']['model'] in ["ARZ2019", "ARZ2020"]:
                                        if self._sim_shower.has_parameter(shp.charge_excess_profile_id):
                                            kwargs = {'iN': self._sim_shower.get_parameter(shp.charge_excess_profile_id)}
                                    if self._cfg['signal']['model'] == "Alvarez2009":
                                        if self._sim_shower.has_parameter(shp.k_L):
                                            kwargs = {'k_L': self._sim_shower.get_parameter(shp.k_L)}
                                            logger.debug(f"reusing k_L parameter of Alvarez2009 model of k_L = {kwargs['k_L']:.4g}")

                                spectrum, additional_output = askaryan.get_frequency_spectrum(self._fin['shower_energies'][self._shower_index], viewing_angles[iS],
                                                self._n_samples, self._dt, self._fin['shower_type'][self._shower_index], n_index, R,
                                                self._cfg['signal']['model'], seed=self._cfg['seed'], full_output=True, **kwargs)
                                # save shower realization to SimShower and hdf5 file
                                if self._cfg['signal']['model'] in ["ARZ2019", "ARZ2020"]:
                                    if 'shower_realization_ARZ' not in self._mout:
                                        self._mout['shower_realization_ARZ'] = np.zeros(self._n_showers)
                                    if not self._sim_shower.has_parameter(shp.charge_excess_profile_id):
                                        self._sim_shower.set_parameter(shp.charge_excess_profile_id, additional_output['iN'])
                                        self._mout['shower_realization_ARZ'][self._shower_index] = additional_output['iN']
                                        logger.debug(f"setting shower profile for ARZ shower library to i = {additional_output['iN']}")
                                if self._cfg['signal']['model'] == "Alvarez2009":
                                    if 'shower_realization_Alvarez2009' not in self._mout:
                                        self._mout['shower_realization_Alvarez2009'] = np.zeros(self._n_showers)
                                    if not self._sim_shower.has_parameter(shp.k_L):
                                        self._sim_shower.set_parameter(shp.k_L, additional_output['k_L'])
                                        self._mout['shower_realization_Alvarez2009'][self._shower_index] = additional_output['k_L']
                                        logger.debug(f"setting k_L parameter of Alvarez2009 model to k_L = {additional_output['k_L']:.4g}")
                                askaryan_time += (time.time() - t_ask)

                                polarization_direction_onsky = self._calculate_polarization_vector()
                                cs_at_antenna = cstrans.cstrafo(*hp.cartesian_to_spherical(*receive_vector))
                                polarization_direction_at_antenna = cs_at_antenna.transform_from_onsky_to_ground(polarization_direction_onsky)
                                logger.debug('receive zenith {:.0f} azimuth {:.0f} polarization on sky {:.2f} {:.2f} {:.2f}, on ground @ antenna {:.2f} {:.2f} {:.2f}'.format(
                                    zenith / units.deg, azimuth / units.deg, polarization_direction_onsky[0],
                                    polarization_direction_onsky[1], polarization_direction_onsky[2],
                                    *polarization_direction_at_antenna))
                                sg['polarization'][iSh, channel_id, iS] = polarization_direction_at_antenna
                                eR, eTheta, ePhi = np.outer(polarization_direction_onsky, spectrum)

                            elif self._fin_attrs['simulation_mode'] == "emitter":
                                # NuRadioMC also supports the simulation of emitters. In this case, the signal model specifies the electric field polarization
                                amplitude = self._fin['emitter_amplitudes'][self._shower_index]
                                emitter_model = self._fin['emitter_model'][self._shower_index]
                                emitter_kwargs = {}
                                emitter_kwargs["launch_vector"] = self._launch_vector
                                for key in self._fin.keys():
                                    if key not in ['emitter_amplitudes', 'emitter_model']:
                                        if key.startswith("emitter_"):
                                            emitter_kwargs[key[8:]] = self._fin[key][self._shower_index]

                                if emitter_model.startswith("efield_"):
                                    if emitter_model == "efield_idl1_spice":
                                        if "emitter_realization" in self._fin:
                                            emitter_kwargs['iN'] = self._fin['emitter_realization'][self._shower_index]
                                        elif emitter_obj.has_parameter(ep.realization_id):
                                            emitter_kwargs['iN'] = emitter_obj.get_parameter(ep.realization_id)
                                        else:
                                            emitter_kwargs['rnd'] = self._rnd

                                    (eR, eTheta, ePhi), additional_output = emitter.get_frequency_spectrum(amplitude, self._n_samples, self._dt, emitter_model, **emitter_kwargs, full_output=True)
                                    if emitter_model == "efield_idl1_spice":
                                        if 'emitter_realization' not in self._mout:
                                            self._mout['emitter_realization'] = np.zeros(self._n_showers)
                                        if not emitter_obj.has_parameter(ep.realization_id):
                                            emitter_obj.set_parameter(ep.realization_id, additional_output['iN'])
                                            self._mout['emitter_realization'][self._shower_index] = additional_output['iN']
                                            logger.debug(f"setting emitter realization to i = {additional_output['iN']}")
                                else:
                                    # the emitter fuction returns the voltage output of the pulser. We need to convole with the antenna response of the emitting antenna
                                    # to obtain the emitted electric field.
                                    # get emitting antenna properties
                                    antenna_model = self._fin['emitter_antenna_type'][self._shower_index]
                                    antenna_pattern = self._antenna_pattern_provider.load_antenna_pattern(antenna_model)
                                    ori = [self._fin['emitter_orientation_theta'][self._shower_index], self._fin['emitter_orientation_phi'][self._shower_index],
                                           self._fin['emitter_rotation_theta'][self._shower_index], self._fin['emitter_rotation_phi'][self._shower_index]]

                                    # source voltage given to the emitter
                                    voltage_spectrum_emitter = emitter.get_frequency_spectrum(amplitude, self._n_samples, self._dt,
                                                                                              emitter_model, **emitter_kwargs)
                                    # convolve voltage output with antenna response to obtain emitted electric field
                                    frequencies = np.fft.rfftfreq(self._n_samples, d=self._dt)
                                    zenith_emitter, azimuth_emitter = hp.cartesian_to_spherical(*self._launch_vector)
                                    VEL = antenna_pattern.get_antenna_response_vectorized(frequencies, zenith_emitter, azimuth_emitter, *ori)
                                    c = constants.c * units.m / units.s
                                    eTheta = VEL['theta'] * (-1j) * voltage_spectrum_emitter * frequencies * n_index / c
                                    ePhi = VEL['phi'] * (-1j) * voltage_spectrum_emitter * frequencies * n_index / c
                                    eR = np.zeros_like(eTheta)
                                # rescale amplitudes by 1/R, for emitters this is not part of the "SignalGen" class
                                eTheta *= 1 / R
                                ePhi *= 1 / R
                            else:
                                logger.error(f"simulation mode {self._fin_attrs['simulation_mode']} unknown.")
                                raise AttributeError(f"simulation mode {self._fin_attrs['simulation_mode']} unknown.")

                            if self._debug:
                                from matplotlib import pyplot as plt
                                fig, (ax, ax2) = plt.subplots(1, 2)
                                ax.plot(self._ff, np.abs(eTheta) / units.micro / units.V * units.m)
                                ax2.plot(self._tt, fft.freq2time(eTheta, 1. / self._dt) / units.micro / units.V * units.m)
                                ax2.set_ylabel("amplitude [$\mu$V/m]")
                                fig.tight_layout()
                                fig.suptitle("$E_C$ = {:.1g}eV $\Delta \Omega$ = {:.1f}deg, R = {:.0f}m".format(
                                    self._fin['shower_energies'][self._shower_index], viewing_angles[iS], R))
                                fig.subplots_adjust(top=0.9)
                                plt.show()

                            electric_field = NuRadioReco.framework.electric_field.ElectricField([channel_id],
                                                position=self._det.get_relative_position(self._sim_station.get_id(), channel_id),
                                                shower_id=self._shower_ids[self._shower_index], ray_tracing_id=iS)
                            if iS is None:
                                a = 1 / 0
                            electric_field.set_frequency_spectrum(np.array([eR, eTheta, ePhi]), 1. / self._dt)
                            electric_field = self._raytracer.apply_propagation_effects(electric_field, iS)
                            # Trace start time is equal to the interaction time relative to the first
                            # interaction plus the wave travel time.
                            if hasattr(self, '_vertex_time'):
                                trace_start_time = self._vertex_time + T
                            else:
                                trace_start_time = T

                            # We shift the trace start time so that the trace time matches the propagation time.
                            # The centre of the trace corresponds to the instant when the signal from the shower
                            # vertex arrives at the observer. The next line makes sure that the centre time
                            # of the trace is equal to vertex_time + T (wave propagation time)
                            trace_start_time -= 0.5 * electric_field.get_number_of_samples() / electric_field.get_sampling_rate()

                            electric_field.set_trace_start_time(trace_start_time)
                            electric_field[efp.azimuth] = azimuth
                            electric_field[efp.zenith] = zenith
                            electric_field[efp.ray_path_type] = propagation.solution_types[self._raytracer.get_solution_type(iS)]
                            electric_field[efp.nu_vertex_distance] = sg['travel_distances'][iSh, channel_id, iS]
                            electric_field[efp.nu_viewing_angle] = viewing_angles[iS]
                            self._sim_station.add_electric_field(electric_field)

                            # apply a simple threshold cut to speed up the simulation,
                            # application of antenna response will just decrease the
                            # signal amplitude
                            if np.max(np.abs(electric_field.get_trace())) > float(self._cfg['speedup']['min_efield_amplitude']) * self._Vrms_efield_per_channel[self._station_id][channel_id]:
                                candidate_station = True
                        # end of ray tracing solutions loop
                    t3 = time.time()
                    rayTracingTime += t3 - t2
                    # end of channels loop
                # end of showers loop
                # now perform first part of detector simulation -> convert each efield to voltage
                # (i.e. apply antenna response) and apply additional simulation of signal chain (such as cable delays,
                # amp response etc.)
                if not candidate_station:
                    logger.debug("electric field amplitude too small in all channels, skipping to next event")
                    continue
                t1 = time.time()
                self._station = NuRadioReco.framework.station.Station(self._station_id)
                self._station.set_sim_station(self._sim_station)
                self._station.get_sim_station().set_station_time(self._evt_time)

                # convert efields to voltages at digitizer
                if hasattr(self, '_detector_simulation_part1'):
                    # we give the user the opportunity to define a custom detector simulation
                    self._detector_simulation_part1()
                else:
                    efieldToVoltageConverterPerEfield.run(self._evt, self._station, self._det)  # convolve efield with antenna pattern
                    self._detector_simulation_filter_amp(self._evt, self._station.get_sim_station(), self._det)
                    channelAddCableDelay.run(self._evt, self._sim_station, self._det)

                if self._cfg['speedup']['amp_per_ray_solution']:
                    self._channelSignalReconstructor.run(self._evt, self._station.get_sim_station(), self._det)
                    for channel in self._station.get_sim_station().iter_channels():
                        tmp_index = np.argwhere(event_indices == self._get_shower_index(channel.get_shower_id()))[0]
                        sg['max_amp_shower_and_ray'][tmp_index, channel.get_id(), channel.get_ray_tracing_solution_id()] = channel.get_parameter(chp.maximum_amplitude_envelope)
                        sg['time_shower_and_ray'][tmp_index, channel.get_id(), channel.get_ray_tracing_solution_id()] = channel.get_parameter(chp.signal_time)
                start_times = []
                channel_identifiers = []
                for channel in self._sim_station.iter_channels():
                    channel_identifiers.append(channel.get_unique_identifier())
                    start_times.append(channel.get_trace_start_time())
                start_times = np.array(start_times)
                start_times_sort = np.argsort(start_times)
                delta_start_times = start_times[start_times_sort][1:] - start_times[start_times_sort][:-1]  # this array is sorted in time
                split_event_time_diff = float(self._cfg['split_event_time_diff'])
                iSplit = np.atleast_1d(np.squeeze(np.argwhere(delta_start_times > split_event_time_diff)))
#                 print(f"start times {start_times}")
#                 print(f"sort array {start_times_sort}")
#                 print(f"delta times {delta_start_times}")
#                 print(f"split at indices {iSplit}")
                n_sub_events = len(iSplit) + 1
                if n_sub_events > 1:
                    logger.info(f"splitting event group id {self._event_group_id} into {n_sub_events} sub events")

                tmp_station = copy.deepcopy(self._station)
                event_group_has_triggered = False
                for iEvent in range(n_sub_events):
                    iStart = 0
                    iStop = len(channel_identifiers)
                    if n_sub_events > 1:
                        if iEvent > 0:
                            iStart = iSplit[iEvent - 1] + 1
                    if iEvent < n_sub_events - 1:
                        iStop = iSplit[iEvent] + 1
                    indices = start_times_sort[iStart: iStop]
                    if n_sub_events > 1:
                        tmp = ""
                        for start_time in start_times[indices]:
                            tmp += f"{start_time/units.ns:.0f}, "
                        tmp = tmp[:-2] + " ns"
                        logger.info(f"creating event {iEvent} of event group {self._event_group_id} ranging rom {iStart} to {iStop} with indices {indices} corresponding to signal times of {tmp}")
                    self._evt = NuRadioReco.framework.event.Event(self._event_group_id, iEvent)  # create new event

                    if particle_mode:
                        # add MC particles that belong to this (sub) event to event structure
                        # add only primary for now, since full interaction chain is not typically in the input hdf5s
                        self._evt.add_particle(self.primary)
                    # copy over generator information from temporary event to event
                    self._evt._generator_info = self._generator_info

                    self._station = NuRadioReco.framework.station.Station(self._station_id)
                    sim_station = NuRadioReco.framework.sim_station.SimStation(self._station_id)
                    sim_station.set_is_neutrino()
                    tmp_sim_station = tmp_station.get_sim_station()
                    self._shower_ids_of_sub_event = []
                    for iCh in indices:
                        ch_uid = channel_identifiers[iCh]
                        shower_id = ch_uid[1]
                        if shower_id not in self._shower_ids_of_sub_event:
                            self._shower_ids_of_sub_event.append(shower_id)
                        sim_station.add_channel(tmp_sim_station.get_channel(ch_uid))
                        efield_uid = ([ch_uid[0]], ch_uid[1], ch_uid[2])  # the efield unique identifier has as first parameter an array of the channels it is valid for
                        for efield in tmp_sim_station.get_electric_fields():
                            if efield.get_unique_identifier() == efield_uid:
                                sim_station.add_electric_field(efield)

                    if particle_mode:
                        # add showers that contribute to this (sub) event to event structure
                        for shower_id in self._shower_ids_of_sub_event:
                            self._evt.add_sim_shower(self._evt_tmp.get_sim_shower(shower_id))
                    else:
                        for shower_id in self._shower_ids_of_sub_event:
                            self._evt.add_sim_emitter(self._evt_tmp.get_sim_emitter(shower_id))
                    self._station.set_sim_station(sim_station)
                    self._station.set_station_time(self._evt_time)
                    self._evt.set_station(self._station)
                    if bool(self._cfg['signal']['zerosignal']):
                        self._increase_signal(None, 0)

                    logger.debug("performing detector simulation")
                    if hasattr(self, '_detector_simulation_part2'):
                        # we give the user the opportunity to specify a custom detector simulation module sequence
                        # which might be needed for certain analyses
                        self._detector_simulation_part2()
                    else:
                        # start detector simulation
                        efieldToVoltageConverter.run(self._evt, self._station, self._det)  # convolve efield with antenna pattern
                        # downsample trace to internal simulation sampling rate (the efieldToVoltageConverter upsamples the trace to
                        # 20 GHz by default to achive a good time resolution when the two signals from the two signal paths are added)
                        channelResampler.run(self._evt, self._station, self._det, sampling_rate=1. / self._dt)

                        if self._is_simulate_noise():
                            max_freq = 0.5 / self._dt
                            channel_ids = self._det.get_channel_ids(self._station.get_id())
                            Vrms = {}
                            for channel_id in channel_ids:
                                norm = self._integrated_channel_response[self._station.get_id()][channel_id]
                                Vrms[channel_id] = self._Vrms_per_channel[self._station.get_id()][channel_id] / (norm / max_freq) ** 0.5  # normalize noise level to the bandwidth its generated for
                            channelGenericNoiseAdder.run(self._evt, self._station, self._det, amplitude=Vrms, min_freq=0 * units.MHz,
                                                         max_freq=max_freq, type='rayleigh', excluded_channels=self._noiseless_channels[station_id])

                        self._detector_simulation_filter_amp(self._evt, self._station, self._det)

                        self._detector_simulation_trigger(self._evt, self._station, self._det)
                    if not self._station.has_triggered():
                        continue

                    event_group_has_triggered = True
                    triggered_showers[self._station_id].extend(self._get_shower_index(self._shower_ids_of_sub_event))
                    self._calculate_signal_properties()

                    global_shower_indices = self._get_shower_index(self._shower_ids_of_sub_event)
                    local_shower_index = np.atleast_1d(np.squeeze(np.argwhere(np.isin(event_indices, global_shower_indices, assume_unique=True))))
                    self._save_triggers_to_hdf5(sg, local_shower_index, global_shower_indices)
                    if self._outputfilenameNuRadioReco is not None:
                        # downsample traces to detector sampling rate to save file size
                        channelResampler.run(self._evt, self._station, self._det, sampling_rate=self._sampling_rate_detector)
                        channelResampler.run(self._evt, self._station.get_sim_station(), self._det, sampling_rate=self._sampling_rate_detector)
                        electricFieldResampler.run(self._evt, self._station.get_sim_station(), self._det, sampling_rate=self._sampling_rate_detector)

                        output_mode = {'Channels': self._cfg['output']['channel_traces'],
                                       'ElectricFields': self._cfg['output']['electric_field_traces'],
                                       'SimChannels': self._cfg['output']['sim_channel_traces'],
                                       'SimElectricFields': self._cfg['output']['sim_electric_field_traces']}
                        if self.__write_detector:
                            self._eventWriter.run(self._evt, self._det, mode=output_mode)
                        else:
                            self._eventWriter.run(self._evt, mode=output_mode)
                        logger.debug("WRITING EVENT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                # end sub events loop

                # add local sg array to output data structure if any
                if event_group_has_triggered:
                    if self._station_id not in self._mout_groups:
                        self._mout_groups[self._station_id] = {}
                    for key in sg:
                        if key not in self._mout_groups[self._station_id]:
                            self._mout_groups[self._station_id][key] = list(sg[key])
                        else:
                            self._mout_groups[self._station_id][key].extend(sg[key])

                detSimTime += time.time() - t1

            # end station loop

        # end event group loop

        # Create trigger structures if there are no triggering events.
        # This is done to ensure that files with no triggering n_events
        # merge properly.
#         self._create_empty_multiple_triggers()

        # save simulation run in hdf5 format (only triggered events)
        t5 = time.time()
        self._write_output_file()
        if self._outputfilenameNuRadioReco is not None:
            self._eventWriter.end()
            logger.debug("closing nur file")

        if "simulation_mode" not in self._fin_attrs or self._fin_attrs['simulation_mode'] == "neutrino": # only calcualte Veff for neutrino simulations
            try:
                self.calculate_Veff()
            except:
                logger.error("error in calculating effective volume")

        t_total = time.time() - t_start
        outputTime = time.time() - t5

        output_NuRadioRecoTime = "Timing of NuRadioReco modules \n"
        ts = []
        for iM, (name, instance, kwargs) in enumerate(self._evt.iter_modules(self._station.get_id())):
            ts.append(instance.run.time[instance])
        ttot = np.sum(np.array(ts))
        for i, (name, instance, kwargs) in enumerate(self._evt.iter_modules(self._station.get_id())):
            t = pretty_time_delta(ts[i])
            trel = 100.*ts[i] / ttot
            output_NuRadioRecoTime += f"{name}: {t} {trel:.1f}%\n"
        logger.status(output_NuRadioRecoTime)

        logger.status("{:d} events processed in {} = {:.2f}ms/event ({:.1f}% input, {:.1f}% ray tracing, {:.1f}% askaryan, {:.1f}% detector simulation, {:.1f}% output, {:.1f}% weights calculation)".format(self._n_showers,
                                                                                         pretty_time_delta(t_total), 1.e3 * t_total / self._n_showers,
                                                                                         100 * input_time / t_total,
                                                                                         100 * (rayTracingTime - askaryan_time) / t_total,
                                                                                         100 * askaryan_time / t_total,
                                                                                         100 * detSimTime / t_total,
                                                                                         100 * outputTime / t_total,
                                                                                         100 * weightTime / t_total))
        triggered = remove_duplicate_triggers(self._mout['triggered'], self._fin['event_group_ids'])
        n_triggered = np.sum(triggered)
        return n_triggered

    def _calculate_emitter_output(self):
        pass

    def _get_shower_index(self, shower_id):
        if hasattr(shower_id, "__len__"):
            return np.array([self._shower_index_array[x] for x in shower_id])
        else:
            return self._shower_index_array[shower_id]

    def _is_simulate_noise(self):
        """
        returns True if noise should be added
        """
        return bool(self._cfg['noise'])

    def _is_in_fiducial_volume(self, pos):
        """ Checks if pos is in fiducial volume """

        for check_attr in ['fiducial_zmin', 'fiducial_zmax']:
            if not check_attr in self._fin_attrs:
                logger.warning("Fiducial volume not defined. Return True")
                return True

        pos = copy.deepcopy(pos) - np.array([self._fin_attrs.get("x0", 0), self._fin_attrs.get("y0", 0), 0])

        if not (self._fin_attrs["fiducial_zmin"] < pos[2] < self._fin_attrs["fiducial_zmax"]):
            return False

        if "fiducial_rmax" in self._fin_attrs:
            radius = np.sqrt(pos[0] ** 2 + pos[1] ** 2)
            return self._fin_attrs["fiducial_rmin"] < radius < self._fin_attrs["fiducial_rmax"]
        elif "fiducial_xmax" in self._fin_attrs:
            return (self._fin_attrs["fiducial_xmin"] < pos[0] < self._fin_attrs["fiducial_xmax"] and
                    self._fin_attrs["fiducial_ymin"] < pos[1] < self._fin_attrs["fiducial_ymax"])
        else:
            raise ValueError("Could not contruct fiducial volume from input file.")

    def _increase_signal(self, channel_id, factor):
        """
        increase the signal of a simulated station by a factor of x
        this is e.g. used to approximate a phased array concept with a single antenna

        Parameters
        ----------
        channel_id: int or None
            if None, all available channels will be modified
        """
        if channel_id is None:
            for electric_field in self._station.get_sim_station().get_electric_fields():
                electric_field.set_trace(electric_field.get_trace() * factor, sampling_rate=electric_field.get_sampling_rate())

        else:
            sim_channels = self._station.get_sim_station().get_electric_fields_for_channels([channel_id])
            for sim_channel in sim_channels:
                sim_channel.set_trace(sim_channel.get_trace() * factor, sampling_rate=sim_channel.get_sampling_rate())

    def _read_input_hdf5(self):
        """
        reads input file into memory
        """
        fin = h5py.File(self._inputfilename, 'r')
        self._fin = {}
        self._fin_stations = {}
        self._fin_attrs = {}
        for key, value in iteritems(fin):
            if isinstance(value, h5py._hl.group.Group):
                self._fin_stations[key] = {}
                for key2, value2 in iteritems(value):
                    self._fin_stations[key][key2] = np.array(value2)
            else:
                if len(value) and type(value[0]) == bytes:
                    self._fin[key] = np.array(value).astype('U')
                else:
                    self._fin[key] = np.array(value)
        for key, value in iteritems(fin.attrs):
            self._fin_attrs[key] = value

        fin.close()

    def _check_vertex_times(self):

        if 'vertex_times' in self._fin:
            return True
        else:
            warn_msg = 'The input file does not include vertex times. '
            warn_msg += 'Vertices from the same event will not be time-ordered.'
            logger.warning(warn_msg)
            return False

    def _calculate_signal_properties(self):
        if self._station.has_triggered():
            self._channelSignalReconstructor.run(self._evt, self._station, self._det)
            amplitudes = np.zeros(self._station.get_number_of_channels())
            amplitudes_envelope = np.zeros(self._station.get_number_of_channels())
            for channel in self._station.iter_channels():
                amplitudes[channel.get_id()] = channel.get_parameter(chp.maximum_amplitude)
                amplitudes_envelope[channel.get_id()] = channel.get_parameter(chp.maximum_amplitude_envelope)
            self._output_maximum_amplitudes[self._station.get_id()].append(amplitudes)
            self._output_maximum_amplitudes_envelope[self._station.get_id()].append(amplitudes_envelope)

    def _create_empty_multiple_triggers(self):
        if 'trigger_names' not in self._mout_attrs:
            self._mout_attrs['trigger_names'] = np.array([])
            self._mout['multiple_triggers'] = np.zeros((self._n_showers, 1), dtype=bool)
            for station_id in self._station_ids:
                sg = self._mout_groups[station_id]
                n_showers = sg['launch_vectors'].shape[0]
                sg['multiple_triggers'] = np.zeros((n_showers, 1), dtype=bool)
                sg['triggered'] = np.zeros(n_showers, dtype=bool)

    def _create_trigger_structures(self):

        if 'trigger_names' not in self._mout_attrs:
            self._mout_attrs['trigger_names'] = []
        extend_array = False
        for trigger in six.itervalues(self._station.get_triggers()):
            if trigger.get_name() not in self._mout_attrs['trigger_names']:
                self._mout_attrs['trigger_names'].append((trigger.get_name()))
                extend_array = True
        # the 'multiple_triggers' output array is not initialized in the constructor because the number of
        # simulated triggers is unknown at the beginning. So we check if the key already exists and if not,
        # we first create this data structure
        if 'multiple_triggers' not in self._mout:
            self._mout['multiple_triggers'] = np.zeros((self._n_showers, len(self._mout_attrs['trigger_names'])), dtype=bool)
            self._mout['trigger_times'] = np.nan * np.zeros_like(self._mout['multiple_triggers'], dtype=float)
#             for station_id in self._station_ids:
#                 sg = self._mout_groups[station_id]
#                 sg['multiple_triggers'] = np.zeros((self._n_showers, len(self._mout_attrs['trigger_names'])), dtype=bool)
        elif extend_array:
            tmp = np.zeros((self._n_showers, len(self._mout_attrs['trigger_names'])), dtype=bool)
            nx, ny = self._mout['multiple_triggers'].shape
            tmp[:, 0:ny] = self._mout['multiple_triggers']
            self._mout['multiple_triggers'] = tmp
            # repeat for trigger times
            tmp_t = np.nan * np.zeros_like(tmp, dtype=float)
            tmp_t[:, 0:ny] = self._mout['trigger_times']
            self._mout['trigger_times'] = tmp_t
#             for station_id in self._station_ids:
#                 sg = self._mout_groups[station_id]
#                 tmp = np.zeros((self._n_showers, len(self._mout_attrs['trigger_names'])), dtype=bool)
#                 nx, ny = sg['multiple_triggers'].shape
#                 tmp[:, 0:ny] = sg['multiple_triggers']
#                 sg['multiple_triggers'] = tmp
        return extend_array

    def _save_triggers_to_hdf5(self, sg, local_shower_index, global_shower_index):

        extend_array = self._create_trigger_structures()
        # now we also need to create the trigger structure also in the sg (station group) dictionary that contains
        # the information fo the current station and event group
        n_showers = sg['launch_vectors'].shape[0]
        if 'multiple_triggers' not in sg:
            sg['multiple_triggers'] = np.zeros((n_showers, len(self._mout_attrs['trigger_names'])), dtype=bool)
            sg['trigger_times'] = np.nan * np.zeros_like(sg['multiple_triggers'], dtype=float)
        elif extend_array:
            tmp = np.zeros((n_showers, len(self._mout_attrs['trigger_names'])), dtype=bool)
            nx, ny = sg['multiple_triggers'].shape
            tmp[:, 0:ny] = sg['multiple_triggers']
            sg['multiple_triggers'] = tmp
            # repeat for trigger times
            tmp_t = np.nan * np.zeros_like(tmp, dtype=float)
            tmp_t[:, :ny] = sg['trigger_times']
            sg['trigger_times'] = tmp_t

        self._output_event_group_ids[self._station_id].append(self._evt.get_run_number())
        self._output_sub_event_ids[self._station_id].append(self._evt.get_id())
        multiple_triggers = np.zeros(len(self._mout_attrs['trigger_names']), dtype=bool)
        trigger_times = np.nan*np.zeros_like(multiple_triggers)
        for iT, trigger_name in enumerate(self._mout_attrs['trigger_names']):
            if self._station.has_trigger(trigger_name):
                multiple_triggers[iT] = self._station.get_trigger(trigger_name).has_triggered()
                trigger_times[iT] = self._station.get_trigger(trigger_name).get_trigger_time()
                for iSh in local_shower_index:  # now save trigger information per shower of the current station
                    sg['multiple_triggers'][iSh][iT] = self._station.get_trigger(trigger_name).has_triggered()
                    sg['trigger_times'][iSh][iT] = trigger_times[iT]
        for iSh, iSh2 in zip(local_shower_index, global_shower_index):  # now save trigger information per shower of the current station
            sg['triggered'][iSh] = np.any(sg['multiple_triggers'][iSh])
            self._mout['triggered'][iSh2] |= sg['triggered'][iSh]
            self._mout['multiple_triggers'][iSh2] |= sg['multiple_triggers'][iSh]
            self._mout['trigger_times'][iSh2] = np.fmin(
                self._mout['trigger_times'][iSh2], sg['trigger_times'][iSh])
        sg['event_id_per_shower'][local_shower_index] = self._evt.get_id()
        sg['event_group_id_per_shower'][local_shower_index] = self._evt.get_run_number()
        self._output_multiple_triggers_station[self._station_id].append(multiple_triggers)
        self._output_trigger_times_station[self._station_id].append(trigger_times)
        self._output_triggered_station[self._station_id].append(np.any(multiple_triggers))

    def get_Vrms(self):
        return self._Vrms

    def get_sampling_rate(self):
        return 1. / self._dt

    def get_bandwidth(self):
        return self._bandwidth

    def _check_if_was_pre_simulated(self):
        """
        checks if the same detector was simulated before (then we can save the ray tracing part)
        """
        self._was_pre_simulated = False

        if 'detector' in self._fin_attrs:
            if isinstance(self._det, detector.rnog_detector.Detector):
                if self._det.export_as_string() == self._fin_attrs['detector']:
                    self._was_pre_simulated = True
            else:
                with open(self._detectorfile, 'r') as fdet:
                    if fdet.read() == self._fin_attrs['detector']:
                        self._was_pre_simulated = True

        if self._was_pre_simulated:
            logger.status("the simulation was already performed with the same detector")

        return self._was_pre_simulated

    def _create_meta_output_datastructures(self):
        """
        creates the data structures of the parameters that will be saved into the hdf5 output file
        """
        self._mout = {}
        self._mout_attributes = {}
        self._mout['weights'] = np.zeros(self._n_showers)
        self._mout['triggered'] = np.zeros(self._n_showers, dtype=bool)
#         self._mout['multiple_triggers'] = np.zeros((self._n_showers, self._number_of_triggers), dtype=bool)
        self._mout_attributes['trigger_names'] = None
        self._amplitudes = {}
        self._amplitudes_envelope = {}
        self._output_triggered_station = {}
        self._output_event_group_ids = {}
        self._output_sub_event_ids = {}
        self._output_multiple_triggers_station = {}
        self._output_trigger_times_station = {}
        self._output_maximum_amplitudes = {}
        self._output_maximum_amplitudes_envelope = {}

        for station_id in self._station_ids:
            self._mout_groups[station_id] = {}
            sg = self._mout_groups[station_id]
            self._output_event_group_ids[station_id] = []
            self._output_sub_event_ids[station_id] = []
            self._output_triggered_station[station_id] = []
            self._output_multiple_triggers_station[station_id] = []
            self._output_trigger_times_station[station_id] = []
            self._output_maximum_amplitudes[station_id] = []
            self._output_maximum_amplitudes_envelope[station_id] = []

    def _create_station_output_structure(self, n_showers, n_antennas):
        nS = self._raytracer.get_number_of_raytracing_solutions()  # number of possible ray-tracing solutions
        sg = {}
        sg['triggered'] = np.zeros(n_showers, dtype=bool)
        # we need the reference to the shower id to be able to find the correct shower in the upper level hdf5 file
        sg['shower_id'] = np.zeros(n_showers, dtype=int) * -1
        sg['event_id_per_shower'] = np.zeros(n_showers, dtype=int) * -1
        sg['event_group_id_per_shower'] = np.zeros(n_showers, dtype=int) * -1
        sg['launch_vectors'] = np.zeros((n_showers, n_antennas, nS, 3)) * np.nan
        sg['receive_vectors'] = np.zeros((n_showers, n_antennas, nS, 3)) * np.nan
        sg['polarization'] = np.zeros((n_showers, n_antennas, nS, 3)) * np.nan
        sg['travel_times'] = np.zeros((n_showers, n_antennas, nS)) * np.nan
        sg['travel_distances'] = np.zeros((n_showers, n_antennas, nS)) * np.nan
        if self._cfg['speedup']['amp_per_ray_solution']:
            sg['max_amp_shower_and_ray'] = np.zeros((n_showers, n_antennas, nS))
            sg['time_shower_and_ray'] = np.zeros((n_showers, n_antennas, nS))
        for parameter_entry in self._raytracer.get_output_parameters():
            if parameter_entry['ndim'] == 1:
                sg[parameter_entry['name']] = np.zeros((n_showers, n_antennas, nS)) * np.nan
            else:
                sg[parameter_entry['name']] = np.zeros((n_showers, n_antennas, nS, parameter_entry['ndim'])) * np.nan
        return sg

    def _read_input_particle_properties(self, idx=None):
        if idx is None:
            idx = self._primary_index
        self._event_group_id = self._fin['event_group_ids'][idx]

        self.input_particle = NuRadioReco.framework.particle.Particle(0)
        self.input_particle[simp.flavor] = self._fin['flavors'][idx]
        self.input_particle[simp.energy] = self._fin['energies'][idx]
        self.input_particle[simp.interaction_type] = self._fin['interaction_type'][idx]
        self.input_particle[simp.inelasticity] = self._fin['inelasticity'][idx]
        self.input_particle[simp.vertex] = np.array([self._fin['xx'][idx],
                                                     self._fin['yy'][idx],
                                                     self._fin['zz'][idx]])
        self.input_particle[simp.zenith] = self._fin['zeniths'][idx]
        self.input_particle[simp.azimuth] = self._fin['azimuths'][idx]
        self.input_particle[simp.inelasticity] = self._fin['inelasticity'][idx]
        self.input_particle[simp.n_interaction] = self._fin['n_interaction'][idx]
        if self._fin['n_interaction'][idx] <= 1:
            # parents before the neutrino and outgoing daughters without shower are currently not
            # simulated. The parent_id is therefore at the moment only rudimentarily populated.
            self.input_particle[simp.parent_id] = None  # primary does not have a parent

        self.input_particle[simp.vertex_time] = 0
        if 'vertex_times' in self._fin:
            self.input_particle[simp.vertex_time] = self._fin['vertex_times'][idx]

    def _read_input_shower_properties(self):
        """ read in the properties of the shower with index _shower_index from input """
        self._event_group_id = self._fin['event_group_ids'][self._shower_index]

        self._shower_vertex = np.array([self._fin['xx'][self._shower_index],
                                        self._fin['yy'][self._shower_index],
                                        self._fin['zz'][self._shower_index]])

        self._vertex_time = 0
        if 'vertex_times' in self._fin:
            self._vertex_time = self._fin['vertex_times'][self._shower_index]

    def _create_sim_station(self):
        """
        created an empyt sim_station object
        """
        # create NuRadioReco event structure
        self._sim_station = NuRadioReco.framework.sim_station.SimStation(self._station_id)
        self._sim_station.set_is_neutrino()

    def _create_sim_shower(self):
        """
        creates a sim_shower object and saves the meta arguments such as neutrino direction, shower energy and self.input_particle[flavor]
        """
        # create NuRadioReco event structure
        self._sim_shower = NuRadioReco.framework.radio_shower.RadioShower(self._shower_ids[self._shower_index])
        # save relevant neutrino properties
        self._sim_shower[shp.zenith] = self.input_particle[simp.zenith]
        self._sim_shower[shp.azimuth] = self.input_particle[simp.azimuth]
        self._sim_shower[shp.energy] = self._fin['shower_energies'][self._shower_index]
        self._sim_shower[shp.flavor] = self.input_particle[simp.flavor]
        self._sim_shower[shp.interaction_type] = self.input_particle[simp.interaction_type]
        self._sim_shower[shp.vertex] = self.input_particle[simp.vertex]
        self._sim_shower[shp.vertex_time] = self._vertex_time
        self._sim_shower[shp.type] = self._fin['shower_type'][self._shower_index]
        # TODO direct parent does not necessarily need to be the primary in general, but full
        # interaction chain is currently not populated in the input files.
        self._sim_shower[shp.parent_id] = self.primary.get_id()

    def _write_output_file(self, empty=False):
        folder = os.path.dirname(self._outputfilename)
        if not os.path.exists(folder) and folder != '':
            logger.warning(f"output folder {folder} does not exist, creating folder...")
            os.makedirs(folder)
        fout = h5py.File(self._outputfilename, 'w')

        if not empty:
            # here we add the first interaction to the saved events
            # if any of its children triggered

            # Careful! saved should be a copy of the triggered array, and not
            # a reference! saved indicates the interactions to be saved, while
            # triggered should indicate if an interaction has produced a trigger
            saved = np.copy(self._mout['triggered'])
            if 'n_interaction' in self._fin:  # if n_interactions is not specified, there are not parents
                parent_mask = self._fin['n_interaction'] == 1
                for event_id in np.unique(self._fin['event_group_ids']):
                    event_mask = self._fin['event_group_ids'] == event_id
                    if True in self._mout['triggered'][event_mask]:
                        saved[parent_mask & event_mask] = True

            logger.status("start saving events")
            # save data sets
            for (key, value) in iteritems(self._mout):
                fout[key] = value[saved]

            # save all data sets of the station groups
            for (key, value) in iteritems(self._mout_groups):
                sg = fout.create_group("station_{:d}".format(key))
                for (key2, value2) in iteritems(value):
                    sg[key2] = np.array(value2)[np.array(value['triggered'])]

            # save "per event" quantities
            if 'trigger_names' in self._mout_attrs:
                n_triggers = len(self._mout_attrs['trigger_names'])
                for station_id in self._mout_groups:
                    n_events_for_station = len(self._output_triggered_station[station_id])
                    if n_events_for_station > 0:
                        n_channels = self._det.get_number_of_channels(station_id)
                        sg = fout["station_{:d}".format(station_id)]
                        sg['event_group_ids'] = np.array(self._output_event_group_ids[station_id])
                        sg['event_ids'] = np.array(self._output_sub_event_ids[station_id])
                        sg['maximum_amplitudes'] = np.array(self._output_maximum_amplitudes[station_id])
                        sg['maximum_amplitudes_envelope'] = np.array(self._output_maximum_amplitudes_envelope[station_id])
                        sg['triggered_per_event'] = np.array(self._output_triggered_station[station_id])

                        # the multiple triggeres 2d array might have different number of entries per event
                        # because the number of different triggers can increase dynamically
                        # therefore we first create an array with the right size and then fill it
                        tmp = np.zeros((n_events_for_station, n_triggers), dtype=bool)
                        for iE, values in enumerate(self._output_multiple_triggers_station[station_id]):
                            tmp[iE] = values
                        sg['multiple_triggers_per_event'] = tmp
                        tmp_t = np.nan * np.zeros_like(tmp, dtype=float)
                        for iE, values in enumerate(self._output_trigger_times_station[station_id]):
                            tmp_t[iE] = values
                        sg['trigger_times_per_event'] = tmp_t


        # save meta arguments
        for (key, value) in iteritems(self._mout_attrs):
            fout.attrs[key] = value

        if isinstance(self._det, detector.rnog_detector.Detector):
            fout.attrs['detector'] = self._det.export_as_string()
        else:
            with open(self._detectorfile, 'r') as fdet:
                fout.attrs['detector'] = fdet.read()

        if not empty:
            # save antenna position separately to hdf5 output
            for station_id in self._mout_groups:
                n_channels = self._det.get_number_of_channels(station_id)
                positions = np.zeros((n_channels, 3))
                for channel_id in range(n_channels):
                    positions[channel_id] = self._det.get_relative_position(station_id, channel_id) + self._det.get_absolute_position(station_id)
                fout["station_{:d}".format(station_id)].attrs['antenna_positions'] = positions
                fout["station_{:d}".format(station_id)].attrs['Vrms'] = list(self._Vrms_per_channel[station_id].values())
                fout["station_{:d}".format(station_id)].attrs['bandwidth'] = list(self._integrated_channel_response[station_id].values())

            fout.attrs.create("Tnoise", self._noise_temp, dtype=float)
            fout.attrs.create("Vrms", self._Vrms, dtype=float)
            fout.attrs.create("dt", self._dt, dtype=float)
            fout.attrs.create("bandwidth", self._bandwidth, dtype=float)
            fout.attrs['n_samples'] = self._n_samples
        fout.attrs['config'] = yaml.dump(self._cfg)

        # save NuRadioMC and NuRadioReco versions
        from NuRadioReco.utilities import version
        import NuRadioMC
        fout.attrs['NuRadioMC_version'] = NuRadioMC.__version__
        fout.attrs['NuRadioMC_version_hash'] = version.get_NuRadioMC_commit_hash()

        if not empty:
            # now we also save all input parameters back into the out file
            for key in self._fin.keys():
                if key.startswith("station_"):
                    continue
                if not key in fout.keys():  # only save data sets that havn't been recomputed and saved already
                    if np.array(self._fin[key]).dtype.char == 'U':
                        fout[key] = np.array(self._fin[key], dtype=h5py.string_dtype(encoding='utf-8'))[saved]

                    else:
                        fout[key] = np.array(self._fin[key])[saved]

        for key in self._fin_attrs.keys():
            if not key in fout.attrs.keys():  # only save atrributes sets that havn't been recomputed and saved already
                if key not in ["trigger_names", "Tnoise", "Vrms", "bandwidth", "n_samples", "dt", "detector", "config"]:  # don't write trigger names from input to output file, this will lead to problems with incompatible trigger names when merging output files
                    fout.attrs[key] = self._fin_attrs[key]
        fout.close()

    def calculate_Veff(self):
        # calculate effective
        triggered = remove_duplicate_triggers(self._mout['triggered'], self._fin['event_group_ids'])
        n_triggered = np.sum(triggered)
        n_triggered_weighted = np.sum(self._mout['weights'][triggered])
        n_events = self._fin_attrs['n_events']
        logger.status(f'fraction of triggered events = {n_triggered:.0f}/{n_events:.0f} = {n_triggered / self._n_showers:.3f} (sum of weights = {n_triggered_weighted:.2f})')

        V = self._fin_attrs['volume']
        Veff = V * n_triggered_weighted / n_events
        logger.status(f"Veff = {Veff / units.km ** 3:.4g} km^3, Veffsr = {Veff * 4 * np.pi/units.km**3:.4g} km^3 sr")

    def _calculate_polarization_vector(self):
        """ calculates the polarization vector in spherical coordinates (eR, eTheta, ePhi)
        """
        if self._cfg['signal']['polarization'] == 'auto':
            polarization_direction = np.cross(self._launch_vector, np.cross(self._shower_axis, self._launch_vector))
            polarization_direction /= np.linalg.norm(polarization_direction)
            cs = cstrans.cstrafo(*hp.cartesian_to_spherical(*self._launch_vector))
            return cs.transform_from_ground_to_onsky(polarization_direction)
        elif self._cfg['signal']['polarization'] == 'custom':
            ePhi = float(self._cfg['signal']['ePhi'])
            eTheta = (1 - ePhi ** 2) ** 0.5
            v = np.array([0, eTheta, ePhi])
            return v / np.linalg.norm(v)
        else:
            msg = "{} for config.signal.polarization is not a valid option".format(self._cfg['signal']['polarization'])
            logger.error(msg)
            raise ValueError(msg)

    @property
    def _bandwidth_per_channel(self):
        warnings.warn("This variable `_bandwidth_per_channel` is deprecated. "
                      "Please use `integrated_channel_response` instead.", DeprecationWarning)
        return self._integrated_channel_response

    @_bandwidth_per_channel.setter
    def _bandwidth_per_channel(self, value):
        warnings.warn("This variable `_bandwidth_per_channel` is deprecated. "
                        "Please use `integrated_channel_response` instead.", DeprecationWarning)
        self._integrated_channel_response = value

    @property
    def integrated_channel_response(self):
        return self._integrated_channel_response

    @integrated_channel_response.setter
    def integrated_channel_response(self, value):
        self._integrated_channel_response = value
