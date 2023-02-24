import NuRadioMC.simulation.simulation_base
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.efieldToVoltageConverterPerEfield
import NuRadioReco.modules.electricFieldResampler
import NuRadioReco.modules.channelAddCableDelay
import NuRadioReco.modules.channelSignalReconstructor
import NuRadioReco.modules.io.eventWriter
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.channelResampler
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import channelParameters as chp
import time
import numpy  as np
import logging
logger = logging.getLogger('NuRadioMC')

class simulation_detector(NuRadioMC.simulation.simulation_base.simulation_base):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super(simulation_detector, self).__init__(*args, **kwargs)
        self._channelSignalReconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
        self._eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
        self._efieldToVoltageConverterPerEfield = NuRadioReco.modules.efieldToVoltageConverterPerEfield.efieldToVoltageConverterPerEfield()
        self._efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
        self._efieldToVoltageConverter.begin(time_resolution=self._cfg['speedup']['time_res_efieldconverter'])
        self._channelAddCableDelay = NuRadioReco.modules.channelAddCableDelay.channelAddCableDelay()
        self._channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
        self._channelGenericNoiseAdder.begin(seed=self._cfg['seed'])
        self._channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
        self._electricFieldResampler = NuRadioReco.modules.electricFieldResampler.electricFieldResampler()
        if self._outputfilenameNuRadioReco is not None:
            self._eventWriter.begin(self._outputfilenameNuRadioReco, log_level=self._log_level)

    def _detector_simulation(
            self,
            event_indices,
            sg
    ):
        t1 = time.time()
        self._station = NuRadioReco.framework.station.Station(self._station_id)
        self._station.set_sim_station(self._sim_station)

        self._simulate_sim_station_detector_response(
            self._evt_tmp,
            self._station
        )

        if self._cfg['speedup']['amp_per_ray_solution']:
            self._channelSignalReconstructor.run(self._evt_tmp, self._station.get_sim_station(), self._det)
            for channel in self._station.get_sim_station().iter_channels():
                tmp_index = np.argwhere(event_indices == self._get_shower_index(channel.get_shower_id()))[0]
                sg['max_amp_shower_and_ray'][tmp_index, self._get_channel_index(
                    channel.get_id()), channel.get_ray_tracing_solution_id()] = channel.get_parameter(
                    chp.maximum_amplitude_envelope)
                sg['time_shower_and_ray'][tmp_index, self._get_channel_index(
                    channel.get_id()), channel.get_ray_tracing_solution_id()] = channel.get_parameter(chp.signal_time)
        start_times = []
        channel_identifiers = []
        for channel in self._sim_station.iter_channels():
            channel_identifiers.append(channel.get_unique_identifier())
            start_times.append(channel.get_trace_start_time())
        start_times = np.array(start_times)
        start_times_sort = np.argsort(start_times)
        delta_start_times = start_times[start_times_sort][1:] - start_times[start_times_sort][
                                                                :-1]  # this array is sorted in time
        split_event_time_diff = float(self._cfg['split_event_time_diff'])
        iSplit = np.atleast_1d(np.squeeze(np.argwhere(delta_start_times > split_event_time_diff)))
        n_sub_events = len(iSplit) + 1
        if n_sub_events > 1:
            logger.info(f"splitting event group id {self._event_group_id} into {n_sub_events} sub events")

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
                    tmp += f"{start_time / units.ns:.0f}, "
                tmp = tmp[:-2] + " ns"
                logger.info(
                    f"creating event {iEvent} of event group {self._event_group_id} ranging rom {iStart} to {iStop} with indices {indices} corresponding to signal times of {tmp}")
            new_event, new_station = self._create_event_structure(
                iEvent,
                indices,
                channel_identifiers
            )
            self._evt = new_event

            if bool(self._cfg['signal']['zerosignal']):
                self._increase_signal(new_station, None, 0)

            logger.debug("performing detector simulation")
            self._simulate_detector_response(
                new_event,
                new_station
            )
            if not new_station.has_triggered():
                continue
            event_group_has_triggered = True
            self._calculate_signal_properties(
                new_event,
                new_station
            )

            global_shower_indices = self._get_shower_index(self._shower_ids_of_sub_event)
            local_shower_index = np.atleast_1d(
                np.squeeze(np.argwhere(np.isin(event_indices, global_shower_indices, assume_unique=True))))
            self._save_triggers_to_hdf5(new_event, new_station, sg, local_shower_index, global_shower_indices)
            if self._outputfilenameNuRadioReco is not None:
                self._write_nur_file(
                    new_event,
                    new_station
                )
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
        # print(self._mout_groups[self._station_id]['travel_times'])
        self._detSimTime += time.time() - t1


    def _simulate_sim_station_detector_response(
            self,
            event,
            station
    ):
        # convert efields to voltages at digitizer
        self._efieldToVoltageConverterPerEfield.run(
            event,
            station,
            self._det)  # convolve efield with antenna pattern
        self._detector_simulation_filter_amp(event, station.get_sim_station(), self._det)
        self._channelAddCableDelay.run(event, station.get_sim_station(), self._det)

    def _simulate_detector_response(
            self,
            event,
            station
    ):
        # start detector simulation
        self._efieldToVoltageConverter.run(
            event,
            station,
           self._det
        )  # convolve efield with antenna pattern
        # downsample trace to internal simulation sampling rate (the efieldToVoltageConverter upsamples the trace to
        # 20 GHz by default to achive a good time resolution when the two signals from the two signal paths are added)
        self._channelResampler.run(
            event,
            station,
            self._det,
            sampling_rate=1. / self._dt)

        if self._is_simulate_noise():
            max_freq = 0.5 / self._dt
            channel_ids = self._det.get_channel_ids(station.get_id())
            Vrms = {}
            for channel_id in channel_ids:
                norm = self._bandwidth_per_channel[station.get_id()][channel_id]
                Vrms[channel_id] = self._Vrms_per_channel[station.get_id()][channel_id] / (
                            norm / max_freq) ** 0.5  # normalize noise level to the bandwidth its generated for
            self._channelGenericNoiseAdder.run(
                event,
                station,
                self._det,
                amplitude=Vrms,
                min_freq=0 * units.MHz,
                max_freq=max_freq,
                type='rayleigh',
                excluded_channels=self._noiseless_channels[station.get_id()]
            )

        self._detector_simulation_filter_amp(event, station, self._det)

        self._detector_simulation_trigger(event, station, self._det)

    def _set_detector_properties(self):
        self._sampling_rate_detector = self._det.get_sampling_frequency(self._station_id, self._channel_ids[0])
        self._n_samples = self._det.get_number_of_samples(self._station_id, self._channel_ids[0]) / self._sampling_rate_detector / self._dt
        self._n_samples = int(np.ceil(self._n_samples / 2.) * 2)  # round to nearest even integer
        self._ff = np.fft.rfftfreq(self._n_samples, self._dt)
        self._tt = np.arange(0, self._n_samples * self._dt, self._dt)
        self._channel_ids = list(self._det.get_channel_ids(self._station_id))