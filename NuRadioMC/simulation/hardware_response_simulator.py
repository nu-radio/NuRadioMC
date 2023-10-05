import numpy as np
import copy
import astropy.time
import NuRadioReco.framework.station
import NuRadioReco.framework.sim_station
import NuRadioReco.framework.event
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
import NuRadioReco.modules.efieldToVoltageConverterPerEfield
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.channelAddCableDelay
import NuRadioReco.modules.channelSignalReconstructor
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.electricFieldResampler
import NuRadioReco.modules.channelGenericNoiseAdder
from NuRadioReco.utilities import units
import scipy

class hardwareResponseSimulator:
    def __init__(
            self,
            detector,
            config,
            station_ids,
            input_data,
            input_attributes,
            detector_simulation_trigger,
            detector_simulation_filter_amp,
            raytracer,
            event_time,
            time_logger
    ):
        self.__detector = detector
        self.__station_ids = station_ids
        self.__config = config
        self.__input_data = input_data
        self.__input_attributes = input_attributes
        self.__sampling_rate_simulation = self.__config['sampling_rate']
        # some module require passing an event, so we pass this dummy. It doesn't actually do anything
        self.__dummy_event = NuRadioReco.framework.event.Event(-1, -1)
        self.__efield_to_voltage_converter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
        self.__efield_to_voltage_converter.begin(self.__config['speedup']['time_res_efieldconverter'])
        self.__efield_to_voltage_converter.begin(time_resolution=self.__config['speedup']['time_res_efieldconverter'])
        self.__efield_to_voltage_per_efield = NuRadioReco.modules.efieldToVoltageConverterPerEfield.efieldToVoltageConverterPerEfield()
        self.__channel_resampler = NuRadioReco.modules.channelResampler.channelResampler()
        self.__efield_resampler = NuRadioReco.modules.electricFieldResampler.electricFieldResampler()
        self.__channel_generic_noise_adder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
        self.__channel_generic_noise_adder.begin(seed=self.__config['seed'])
        self.__detector_simulation_trigger = detector_simulation_trigger
        self.__detector_simulation_filter_amp = detector_simulation_filter_amp
        self.__raytracer = raytracer
        self.__event_time = event_time
        self.__time_logger = time_logger
        self.__event_group_id = None
        self.__channel_add_cable_delay = NuRadioReco.modules.channelAddCableDelay.channelAddCableDelay()
        self.__channel_signal_reconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
        shower_ids = self.__input_data['shower_ids']
        self.__shower_index_array = {}
        for shower_index, shower_id in enumerate(shower_ids):
            self.__shower_index_array[shower_id] = shower_index
        self.__bandwidth_per_channel = {}
        self.__amplification_per_channel = {}
        self.__noise_adder_normalization = {}
        self.__v_rms_per_channel = {}
        self.__noiseless_channels = {}
        self.__noise_temperature = None
        self.__noise_vrms = None
        self.__perform_dummy_detector_simulation()
        self.__bandwidth = next(iter(next(iter(self.__bandwidth_per_channel.values())).values()))
        self.__calculate_noise_rms()

    def set_event_group(
            self,
            event_group_id
    ):
        self.__event_group_id = event_group_id

    def simulate_detector_response(
            self,
            station_id,
            efield_objects,
            shower_indices
    ):
        self.__time_logger.start_time('detector simulation')
        channel_ids = list(self.__detector.get_channel_ids(station_id))
        sampling_rate_detector = self.__detector.get_sampling_frequency(station_id, channel_ids[0])
        n_samples = self.__detector.get_number_of_samples(station_id, channel_ids[0]) / sampling_rate_detector  * self.__sampling_rate_simulation
        n_samples = int(np.ceil(n_samples / 2.) * 2)  # round to nearest even integer
        ff = np.fft.rfftfreq(n_samples, 1. / self.__sampling_rate_simulation)
        tt = np.arange(0, n_samples / self.__sampling_rate_simulation, 1. / self.__sampling_rate_simulation)
        output_data = self.__get_output_structure(shower_indices, station_id)
        dummy_station = NuRadioReco.framework.station.Station(station_id)
        sim_station = NuRadioReco.framework.sim_station.SimStation(station_id)
        sim_station.set_is_neutrino()
        dummy_station.set_sim_station(sim_station)
        dummy_station = self.__simulate_sim_station_detector_response(
            dummy_station,
            efield_objects
        )
        if dummy_station is None:
            self.__time_logger.stop_time('detector simulation')
            return None, None, None, None, None
        if self.__config['speedup']['amp_per_ray_solution']:
            self.__channel_signal_reconstructor.run(self.__dummy_event, sim_station, self.__detector)
            for channel in sim_station.iter_channels():
                channel_index = channel_ids.index(channel.get_id())
                raytracing_id = channel.get_ray_tracing_solution_id()
                shower_index = self.__shower_index_array[channel.get_shower_id()]
                tmp_index = np.argwhere(shower_indices == shower_index)[0]
                output_data['max_amp_shower_and_ray'][tmp_index, channel_index, raytracing_id] = channel.get_parameter(chp.maximum_amplitude_envelope)
                output_data['time_shower_and_ray'][tmp_index, channel_index, raytracing_id] = channel.get_parameter(chp.signal_time)
        start_times = []
        channel_identifiers = []
        for channel in sim_station.iter_channels():
            channel_identifiers.append(channel.get_unique_identifier())
            start_times.append(channel.get_trace_start_time())
        start_times = np.array(start_times)
        start_times_sort = np.argsort(start_times)
        delta_start_times = start_times[start_times_sort][1:] - start_times[start_times_sort][:-1]
        split_event_time_diff = float(self.__config['split_event_time_diff'])
        split_indices = np.atleast_1d(np.squeeze(np.argwhere(delta_start_times > split_event_time_diff)))
        n_sub_events = len(split_indices) + 1
        station_objects = {}
        event_objects = {}
        station_has_triggered = np.zeros(n_sub_events, dtype=bool)
        sub_event_shower_ids = []
        for i_sub_event in range(n_sub_events):
            i_start = 0
            i_stop = len(channel_identifiers)
            if n_sub_events > 1:
                if i_sub_event > 0:
                    i_start = split_indices[i_sub_event - 1] + 1
            if i_sub_event < n_sub_events - 1:
                i_stop = split_indices[i_sub_event] + 1
            channel_indices = start_times_sort[i_start:i_stop]
            new_sim_station, sub_shower_ids = self.__split_sim_stations(
                dummy_station,
                channel_indices,
                channel_identifiers
            )
            sub_event_shower_ids.append(sub_shower_ids)
            new_station = NuRadioReco.framework.station.Station(station_id)
            new_station.set_sim_station(new_sim_station)
            new_event = NuRadioReco.framework.event.Event(self.__event_group_id, i_sub_event)
            new_station.set_station_time(astropy.time.Time(['2018-01-01T00:00:01.000'], scale='utc', format='isot'))
            new_station.get_sim_station().set_station_time(new_station.get_station_time())
            new_event.set_station(new_station)
            if self.__config['signal']['zerosignal']:
                self.__increase_signal(new_station, None, 0)
            self.__simulate_station_detector_response(new_event, new_station)
            station_has_triggered[i_sub_event] = new_station.has_triggered()
            station_objects[i_sub_event] = new_station
            event_objects[i_sub_event] = new_event
        self.__time_logger.stop_time('detector simulation')
        return event_objects, station_objects, sub_event_shower_ids, station_has_triggered, output_data
    
    def __simulate_station_detector_response(
            self,
            event,
            station
    ):
        self.__efield_to_voltage_converter.run(
            event,
            station,
            self.__detector
        )
        channel_ids = self.__detector.get_channel_ids(station.get_id())
        sampling_rate_detector = self.__detector.get_sampling_frequency(
            station.get_id(),
            channel_ids[0]
        )
        self.__channel_resampler.run(
            event,
            station,
            self.__detector,
            self.__config['sampling_rate']
        )
        if self.__config['noise']:
            max_freq = 0.5 * self.__config['sampling_rate']
            v_rms = {}
            for channel_id in channel_ids:
                norm = self.__bandwidth_per_channel[station.get_id()][channel_id]
                v_rms[channel_id] = self.__v_rms_per_channel[station.get_id()][channel_id] / (
                            norm / max_freq) ** 0.5  # normalize noise level to the bandwidth its generated for
            self.__channel_generic_noise_adder.run(
                event,
                station,
                self.__detector,
                amplitude=v_rms,
                min_freq=0 * units.MHz,
                max_freq=max_freq,
                type='rayleigh',
                excluded_channels=self.__noiseless_channels[station.get_id()]
            )
        self.__detector_simulation_filter_amp(event, station, self.__detector)

        self.__detector_simulation_trigger(event, station, self.__detector)
        if station.has_triggered:
            self.__channel_signal_reconstructor.run(event, station, self.__detector)
    
    def __simulate_sim_station_detector_response(
            self,
            station,
            efield_array
    ):
        n_efields = 0
        for efields_for_shower in efield_array:
            for efields_for_channel in efields_for_shower:
                for efield_object in efields_for_channel:
                    station.get_sim_station().add_electric_field(efield_object)
                    n_efields += 1
        if n_efields == 0:
            return None
        self.__efield_to_voltage_per_efield.run(
            self.__dummy_event,
            station,
            self.__detector
        )
        self.__detector_simulation_filter_amp(self.__dummy_event, station.get_sim_station(), self.__detector)
        self.__channel_add_cable_delay.run(self.__dummy_event, station.get_sim_station(), self.__detector)
        return station

    def __get_output_structure(self, shower_indices, station_id):
        station_output_structure = {}
        n_showers = len(shower_indices)
        n_antennas = len(self.__detector.get_channel_ids(station_id))
        n_raytracing_solutions = self.__raytracer.get_number_of_raytracing_solutions()  # number of possible ray-tracing solutions
        if self.__config['speedup']['amp_per_ray_solution']:
            station_output_structure['max_amp_shower_and_ray'] = np.zeros((n_showers, n_antennas, n_raytracing_solutions))
            station_output_structure['time_shower_and_ray'] = np.zeros((n_showers, n_antennas, n_raytracing_solutions))
        return station_output_structure

    def __split_sim_stations(
            self,
            dummy_station,
            channel_indices,
            channel_identifiers
    ):
        new_sim_station = NuRadioReco.framework.sim_station.SimStation(dummy_station.get_id())
        new_sim_station.set_is_neutrino()
        sub_event_shower_ids = []
        for channel_index in channel_indices:
            channel_identifier = channel_identifiers[channel_index]
            if channel_identifier[1] not in sub_event_shower_ids:
                sub_event_shower_ids.append(channel_identifier[1])
            new_sim_station.add_channel(copy.deepcopy(dummy_station.get_sim_station().get_channel(channel_identifier)))
            efield_identifier = ([channel_identifier[0]], channel_identifier[1], channel_identifier[2])
            for efield in dummy_station.get_sim_station().get_electric_fields():
                if efield.get_unique_identifier() == efield_identifier:
                    new_sim_station.add_electric_field(copy.deepcopy(efield))
        return new_sim_station, sub_event_shower_ids

    def __increase_signal(self, station, channel_id, factor):
        """
        increase the signal of a simulated station by a factor of x
        this is e.g. used to approximate a phased array concept with a single antenna

        Parameters
        ----------
        channel_id: int or None
            if None, all available channels will be modified
        """
        if channel_id is None:
            for electric_field in station.get_sim_station().get_electric_fields():
                electric_field.set_trace(electric_field.get_trace() * factor, sampling_rate=electric_field.get_sampling_rate())

        else:
            sim_channels = station.get_sim_station().get_electric_fields_for_channels([channel_id])
            for sim_channel in sim_channels:
                sim_channel.set_trace(sim_channel.get_trace() * factor, sampling_rate=sim_channel.get_sampling_rate())

    def __perform_dummy_detector_simulation(self):
        """
        Performs a detector simulation on a dummy station to estimate the bandwidth used to convert from
        noise temperature to noise RMS
        """

        # first create dummy event and station with channels
        v_rms = 1
        for iSt, station_id in enumerate(self.__station_ids):
            channel_ids = self.__detector.get_channel_ids(station_id)
            first_channel_id = channel_ids[0]
            dummy_event = NuRadioReco.framework.event.Event(0, 0)
            dummy_sim_station = NuRadioReco.framework.sim_station.SimStation(station_id)
            sampling_rate_detector = self.__detector.get_sampling_frequency(station_id, first_channel_id)
            n_samples = self.__detector.get_number_of_samples(station_id, first_channel_id) / sampling_rate_detector * self.__sampling_rate_simulation
            n_samples = int(np.ceil(n_samples / 2.) * 2)  # round to nearest even integer
            ff = np.fft.rfftfreq(n_samples, 1. / self.__sampling_rate_simulation)
            tt = np.arange(0, n_samples / self.__sampling_rate_simulation, 1. / self.__sampling_rate_simulation)

            for channel_id in channel_ids:
                electric_field = NuRadioReco.framework.electric_field.ElectricField(
                    [channel_id],
                    self.__detector.get_relative_position(
                        dummy_sim_station.get_id(),
                        channel_id
                    )
                )
                trace = np.zeros_like(tt)
                trace[n_samples // 2] = 100 * units.V  # set a signal that will satisfy any high/low trigger
                trace[n_samples // 2 + 1] = -100 * units.V
                electric_field.set_trace(np.array([np.zeros_like(tt), trace, trace]), self.__sampling_rate_simulation)
                electric_field.set_trace_start_time(0)
                electric_field[efp.azimuth] = 0
                electric_field[efp.zenith] = 100 * units.deg
                electric_field[efp.ray_path_type] = 0
                dummy_sim_station.add_electric_field(electric_field)

            dummy_station = NuRadioReco.framework.station.Station(station_id)
            dummy_station.set_sim_station(dummy_sim_station)
            dummy_station.set_station_time(self.__event_time)
            dummy_event.set_station(dummy_station)

            self.__detector_simulation_filter_amp(dummy_event, dummy_station, self.__detector)
            self.__bandwidth_per_channel[station_id] = {}
            self.__amplification_per_channel[station_id] = {}
            for channel_id in self.__detector.get_channel_ids(station_id):
                ff = np.linspace(0, 0.5 * self.__sampling_rate_simulation, 10000)
                filt = np.ones_like(ff, dtype=np.complex)
                for i, (name, instance, kwargs) in enumerate(dummy_event.iter_modules(station_id)):
                    if hasattr(instance, "get_filter"):
                        filt *= instance.get_filter(ff, station_id, channel_id, self.__detector, **kwargs)

                self.__amplification_per_channel[station_id][channel_id] = np.abs(filt).max()
                bandwidth = np.trapz(np.abs(filt) ** 2, ff)
                self.__bandwidth_per_channel[station_id][channel_id] = bandwidth

    def __calculate_noise_rms(self):
        noise_temp = self.__config['trigger']['noise_temperature']
        v_rms = self.__config['trigger']['Vrms']
        if noise_temp is not None and v_rms is not None:
            raise AttributeError(f"Specifying noise temperature (set to {noise_temp}) and Vrms (set to {v_rms} is not allowed.")
        if noise_temp is not None:
            if noise_temp == "detector":
                noise_temp = None  # the noise temperature is defined in the detector description
            else:
                noise_temp = float(noise_temp)
            self.__v_rms_per_channel = {}
            self.__noiseless_channels = {}
            for station_id in self.__bandwidth_per_channel:
                self.__v_rms_per_channel[station_id] = {}
                self.__noiseless_channels[station_id] = []
                for channel_id in self.__detector.get_channel_ids(station_id):
                    if noise_temp is None:
                        noise_temp_channel = self.__detector.get_noise_temperature(station_id, channel_id)
                    else:
                        noise_temp_channel = noise_temp
                    if self.__detector.is_channel_noiseless(station_id, channel_id):
                        self.__noiseless_channels[station_id].append(channel_id)

                    self.__v_rms_per_channel[station_id][channel_id] = (noise_temp_channel * 50 * scipy.constants.k *
                           self.__bandwidth_per_channel[station_id][channel_id] / units.Hz) ** 0.5  # from elog:1566 and https://en.wikipedia.org/wiki/Johnson%E2%80%93Nyquist_noise (last Eq. in "noise voltage and power" section
            self.__noise_vrms = next(iter(next(iter(self.__v_rms_per_channel.values())).values()))
        elif v_rms is not None:
            self.__noise_vrms = float(v_rms) * units.V
        else:
            raise AttributeError(f"noise temperature and Vrms are both set to None")
        self.__noise_temperature = noise_temp
        self.__v_rms_efield_per_channel = {}
        for station_id in self.__bandwidth_per_channel:
            self.__v_rms_efield_per_channel[station_id] = {}
            for channel_id in self.__bandwidth_per_channel[station_id]:
                self.__v_rms_efield_per_channel[station_id][channel_id] = self.__v_rms_per_channel[station_id][channel_id] / self.__amplification_per_channel[station_id][channel_id] / units.m
        self.__v_rms_efield = next(iter(next(iter(self.__v_rms_efield_per_channel.values())).values()))

    def get_noise_temperature(self):
        return self.__noise_temperature

    def get_noise_vrms(self):
        return self.__noise_vrms
    
    def get_noise_vrms_per_channel(self):
        return self.__v_rms_per_channel

    def get_bandwidth(self):
        return self.__bandwidth
    
    def get_efield_v_rms_per_channel(self):
        return self.__v_rms_efield_per_channel