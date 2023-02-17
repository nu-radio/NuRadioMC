import numpy as np
import datetime
import logging
import os
import yaml
import collections
from six import iteritems
import NuRadioReco.detector.antennapattern
import NuRadioMC.SignalProp.propagation
import NuRadioMC.utilities.medium
import NuRadioReco.detector.detector
import NuRadioReco.detector.generic_detector
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import generatorAttributes as genattrs
from NuRadioReco.framework.parameters import electricFieldParameters as efp
import scipy.constants

STATUS = 31
class NuRadioMCLogger(logging.Logger):

    def status(self, msg, *args, **kwargs):
        if self.isEnabledFor(STATUS):
            self._log(STATUS, msg, args, **kwargs)

logging.setLoggerClass(NuRadioMCLogger)
logging.addLevelName(STATUS, 'STATUS')
logger = logging.getLogger("NuRadioMC")
assert isinstance(logger, NuRadioMCLogger)


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


class simulation_base:
    def __init__(self, inputfilename,
                 outputfilename,
                 detectorfile,
                 outputfilenameNuRadioReco=None,
                 debug=False,
                 evt_time=datetime.datetime(2018, 1, 1),
                 config_file=None,
                 log_level=logging.WARNING,
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
        if 'write_mode' in kwargs.keys():
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
        self._write_detector = write_detector
        logger.status("setting event time to {}".format(evt_time))
        self._event_group_list = event_list
        self._evt = None
        self._antenna_pattern_provider = NuRadioReco.detector.antennapattern.AntennaPatternProvider()

        # initialize propagation module
        self._prop = NuRadioMC.SignalProp.propagation.get_propagation_module(self._cfg['propagation']['module'])

        if self._cfg['propagation']['ice_model'] == "custom":
            if ice_model is None:
                logger.error("ice model is set to 'custom' in config file but no custom ice model is provided.")
                raise AttributeError("ice model is set to 'custom' in config file but no custom ice model is provided.")
            self._ice = ice_model
        else:
            self._ice = NuRadioMC.utilities.medium.get_ice_model(self._cfg['propagation']['ice_model'])

        self._mout = collections.OrderedDict()
        self._mout_groups = collections.OrderedDict()
        self._mout_attrs = collections.OrderedDict()

        # read in detector positions
        logger.status("Detectorfile {}".format(os.path.abspath(self._detectorfile)))
        self._det = None
        if default_detector_station is not None:
            logger.warning(
                'Deprecation warning: Passing the default detector station is deprecated. Default stations and default'
                'channel should be specified in the detector description directly.'
            )
            logger.status(f"Default detector station provided (station {default_detector_station}) -> Using generic detector")
            self._det = NuRadioReco.detector.generic_detector.GenericDetector(json_filename=self._detectorfile, default_station=default_detector_station,
                                                 default_channel=default_detector_channel, antenna_by_depth=False)
        else:
            self._det = NuRadioReco.detector.detector.Detector(json_filename=self._detectorfile, antenna_by_depth=False)

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
        self._particle_mode = "simulation_mode" not in self._fin_attrs or self._fin_attrs['simulation_mode'] != "emitter"

        for enum_entry in genattrs:
            if enum_entry.name in self._fin_attrs:
                self._generator_info[enum_entry] = self._fin_attrs[enum_entry.name]

        # check if the input file contains events, if not save empty output file (for book keeping) and terminate simulation
        if len(self._fin['xx']) == 0:
            logger.status(f"input file {self._inputfilename} is empty")
            return

        self._calculate_noise_rms()


        self._distance_cut_polynomial = None
        if self._cfg['speedup']['distance_cut']:
            coef = self._cfg['speedup']['distance_cut_coefficients']
            self.__distance_cut_polynomial = np.polynomial.polynomial.Polynomial(coef)

            def get_distance_cut(shower_energy):
                if shower_energy <= 0:
                    return 100 * units.m
                return max(100 * units.m, 10 ** self.__distance_cut_polynomial(np.log10(shower_energy)))

            self._get_distance_cut = get_distance_cut

        # initialize parameters to measure run times
        self._input_time = 0.0
        self._askaryan_time = 0.0
        self._rayTracingTime = 0.0
        self._detSimTime = 0.0
        self._outputTime = 0.0
        self._weightTime = 0.0
        self._distance_cut_time = 0.0

    def _calculate_noise_rms(self):
        self._perform_dummy_detector_simulation()
        self._bandwidth = next(iter(next(iter(self._bandwidth_per_channel.values())).values()))
        amplification = next(iter(next(iter(self._amplification_per_channel.values())).values()))
        noise_temp = self._cfg['trigger']['noise_temperature']
        Vrms = self._cfg['trigger']['Vrms']
        if noise_temp is not None and Vrms is not None:
            raise AttributeError(f"Specifying noise temperature (set to {noise_temp}) and Vrms (set to {Vrms} is not allowed.")
        if noise_temp is not None:
            if noise_temp == "detector":
                self._noise_temp = None  # the noise temperature is defined in the detector description
            else:
                self._noise_temp = float(noise_temp)
            self._Vrms_per_channel = {}
            self._noiseless_channels = {}
            for station_id in self._bandwidth_per_channel:
                self._Vrms_per_channel[station_id] = {}
                self._noiseless_channels[station_id] = []
                for channel_id in self._det.get_channel_ids(station_id):
                    if self._noise_temp is None:
                        noise_temp_channel = self._det.get_noise_temperature(station_id, channel_id)
                    else:
                        noise_temp_channel = self._noise_temp
                    if self._det.is_channel_noiseless(station_id, channel_id):
                        self._noiseless_channels[station_id].append(channel_id)

                    self._Vrms_per_channel[station_id][channel_id] = (noise_temp_channel * 50 * scipy.constants.k *
                           self._bandwidth_per_channel[station_id][channel_id] / units.Hz) ** 0.5  # from elog:1566 and https://en.wikipedia.org/wiki/Johnson%E2%80%93Nyquist_noise (last Eq. in "noise voltage and power" section
                    logger.status(f'station {station_id} channel {channel_id} noise temperature = {noise_temp_channel}, bandwidth = {self._bandwidth_per_channel[station_id][channel_id]/ units.MHz:.2f} MHz -> Vrms = {self._Vrms_per_channel[station_id][channel_id]/ units.V / units.micro:.2f} muV')
            self._Vrms = next(iter(next(iter(self._Vrms_per_channel.values())).values()))
            logger.status('(if same bandwidth for all stations/channels is assumed:) noise temperature = {}, bandwidth = {:.2f} MHz -> Vrms = {:.2f} muV'.format(self._noise_temp, self._bandwidth / units.MHz, self._Vrms / units.V / units.micro))
        elif Vrms is not None:
            self._Vrms = float(Vrms) * units.V
            self._noise_temp = None
        else:
            raise AttributeError(f"noise temperature and Vrms are both set to None")

        self._Vrms_efield_per_channel = {}
        for station_id in self._bandwidth_per_channel:
            self._Vrms_efield_per_channel[station_id] = {}
            for channel_id in self._bandwidth_per_channel[station_id]:
                self._Vrms_efield_per_channel[station_id][channel_id] = self._Vrms_per_channel[station_id][channel_id] / self._amplification_per_channel[station_id][channel_id] / units.m
        self._Vrms_efield = next(iter(next(iter(self._Vrms_efield_per_channel.values())).values()))
        tmp_cut = float(self._cfg['speedup']['min_efield_amplitude'])
        logger.status(f"final Vrms {self._Vrms/units.V:.2g}V corresponds to an efield of {self._Vrms_efield/units.V/units.m/units.micro:.2g} muV/m for a VEL = 1m (amplification factor of system is {amplification:.1f}).\n -> all signals with less then {tmp_cut:.1f} x Vrms_efield = {tmp_cut * self._Vrms_efield/units.m/units.V/units.micro:.2g}muV/m will be skipped")

    def _perform_dummy_detector_simulation(self):
        """
        Performs a detector simulation on a dummy station to estimate the bandwidth used to convert from
        noise temperature to noise RMS
        """
        self._bandwidth_per_channel = {}
        self._amplification_per_channel = {}
        self.__noise_adder_normalization = {}

        # first create dummy event and station with channels
        self._Vrms = 1
        for iSt, self._station_id in enumerate(self._station_ids):
            self._shower_index = 0
            self._primary_index = 0
            self._channel_ids = self._det.get_channel_ids(self._station_id)
            first_channel_id = self._channel_ids[0]
            dummy_event = NuRadioReco.framework.event.Event(0, self._primary_index)
            dummy_sim_station = NuRadioReco.framework.sim_station.SimStation(self._station_id)
            self._sampling_rate_detector = self._det.get_sampling_frequency(self._station_id, first_channel_id)
            self._n_samples = self._det.get_number_of_samples(self._station_id, first_channel_id) / self._sampling_rate_detector / self._dt
            self._n_samples = int(np.ceil(self._n_samples / 2.) * 2)  # round to nearest even integer
            self._ff = np.fft.rfftfreq(self._n_samples, self._dt)
            self._tt = np.arange(0, self._n_samples * self._dt, self._dt)

            self._create_sim_station()
            for channel_id in self._channel_ids:
                electric_field = NuRadioReco.framework.electric_field.ElectricField(
                    [channel_id],
                    self._det.get_relative_position(
                        dummy_sim_station.get_id(),
                        channel_id
                    )
                )
                trace = np.zeros_like(self._tt)
                trace[self._n_samples // 2] = 100 * units.V  # set a signal that will satisfy any high/low trigger
                trace[self._n_samples // 2 + 1] = -100 * units.V
                electric_field.set_trace(np.array([np.zeros_like(self._tt), trace, trace]), 1. / self._dt)
                electric_field.set_trace_start_time(0)
                electric_field[efp.azimuth] = 0
                electric_field[efp.zenith] = 100 * units.deg
                electric_field[efp.ray_path_type] = 0
                dummy_sim_station.add_electric_field(electric_field)

            dummy_station = NuRadioReco.framework.station.Station(self._station_id)
            dummy_station.set_sim_station(dummy_sim_station)
            dummy_station.set_station_time(self._evt_time)
            dummy_event.set_station(dummy_station)

            self._detector_simulation_filter_amp(dummy_event, dummy_station, self._det)
            self._bandwidth_per_channel[self._station_id] = {}
            self._amplification_per_channel[self._station_id] = {}
            for channel_id in self._det.get_channel_ids(self._station_id):
                ff = np.linspace(0, 0.5 / self._dt, 10000)
                filt = np.ones_like(ff, dtype=np.complex)
                for i, (name, instance, kwargs) in enumerate(dummy_event.iter_modules(self._station_id)):
                    if hasattr(instance, "get_filter"):
                        filt *= instance.get_filter(ff, self._station_id, channel_id, self._det, **kwargs)

                self._amplification_per_channel[self._station_id][channel_id] = np.abs(filt).max()
                bandwidth = np.trapz(np.abs(filt) ** 2, ff)
                self._bandwidth_per_channel[self._station_id][channel_id] = bandwidth
                logger.status(
                    f"bandwidth of station {self._station_id} channel {channel_id} is {bandwidth / units.MHz:.1f}MHz"
                )

