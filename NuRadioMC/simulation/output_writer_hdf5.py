import numpy as np
import h5py
import yaml
import collections
from six import iteritems, itervalues
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.utilities import units


class outputWriterHDF5:
    def __init__(
            self,
            output_filename,
            config,
            detector,
            station_ids,
            raytracer,
            detector_simulator,
            input_filename
    ):
        self.__filename = output_filename
        self.__config = config
        self.__detector = detector
        self.__station_ids = station_ids
        self.__raytracer = raytracer
        self.__output_station = collections.OrderedDict()
        self.__output_groups = collections.OrderedDict()
        self.__meta_output = None
        self.__meta_output_attributes = None
        self.__detector_simulator = detector_simulator
        self.__input_filename = input_filename
        self.__shower_index_array = {}
        input_file = h5py.File(self.__input_filename, 'r')
        self.__input_data = {}
        self.__input_attributes = {}
        for key, value in iteritems(input_file):
            if not isinstance(value, h5py._hl.group.Group):
                if len(value) and type(value[0]) == bytes:
                    self.__input_data[key] = np.array(value).astype('U')
                else:
                    self.__input_data[key] = np.array(value)
        for shower_index, shower_id in enumerate(self.__input_data['shower_ids']):
            self.__shower_index_array[shower_id] = shower_index
        for key, value in iteritems(input_file.attrs):
            self.__input_attributes[key] = value
        self.__create_meta_output_datastructures()

        input_file.close()

    def add_station(
            self,
            station_id,
            event_objects,
            station_objects,
            simulation_results,
            hardware_response_sim_results,
            event_group_id,
            sub_event_shower_ids
    ):
        if station_id not in self.__output_station.keys():
            self.__output_station[station_id] = {}
        event_group_ids = np.full(simulation_results['event_id_per_shower'].shape, event_group_id)
        for key in simulation_results:
            if key not in self.__output_station[station_id]:
                self.__output_station[station_id][key] = list(simulation_results[key])
            else:
                self.__output_station[station_id][key].extend(list(simulation_results[key]))
        for key in hardware_response_sim_results:
            if key not in self.__output_station[station_id]:
                self.__output_station[station_id][key] = list(hardware_response_sim_results[key])
            else:
                self.__output_station[station_id][key].extend(list(hardware_response_sim_results[key]))
        event_indices = np.atleast_1d(np.squeeze(np.argwhere(self.__input_data['event_group_ids'] == event_group_id)))
        for i_station,  station in iteritems(station_objects):
            for trigger_name in station.get_triggers():
                trigger = station.get_trigger(trigger_name)

            self.__add_trigger_to_output(
                event_objects[i_station],
                station,
                sub_event_shower_ids[i_station],
                event_indices,
                simulation_results['launch_vectors'].shape[0]
            )
            if station.has_triggered():
                amplitudes = np.zeros(station.get_number_of_channels())
                amplitudes_envelope = np.zeros(station.get_number_of_channels())
                channel_ids = self.__detector.get_channel_ids(station_id)
                for channel in station.iter_channels():
                    channel_index = channel_ids.index(channel.get_id())
                    amplitudes[channel_index] = channel.get_parameter(chp.maximum_amplitude)
                    amplitudes_envelope[channel_index] = channel.get_parameter(chp.maximum_amplitude_envelope)
                self.__output_maximum_amplitudes[station_id].append(amplitudes)
                self.__output_maximum_amplitudes_envelope[station_id].append(amplitudes_envelope)

    def save_output(self):
        output_file = h5py.File(self.__filename, 'w')

        saved_events_mask = np.copy(self.__meta_output['triggered'])
        if 'n_interactions' in self.__input_data:  # if n_interactions is not specified, there are no parents
            parent_mask = self.__input_data['n_interaction'] == 1
            for event_id in np.unique(self.__input_data['event_group_ids']):
                event_mask = self.__input_data['event_group_ids'] == event_id
                if True in self.__meta_output['triggered'][event_mask]:
                    saved_events_mask[parent_mask & event_mask] = True

        for station_key, val in iteritems(self.__output_station):
            output_group = output_file.create_group('station_{:d}'.format(station_key))
            for key, value in iteritems(val):
                output_group[key] = np.array(value)
        if 'trigger_names' in self.__meta_output_attributes:
            n_triggers = len(self.__meta_output_attributes['trigger_names'])
            for station_id in self.__meta_output_groups:
                n_events_for_station = len(self.__output_triggered_station[station_id])
                if n_events_for_station > 0:
                    station_data = output_file['station_{:d}'.format(station_id)]
                    station_data['event_group_dis'] = np.array(self.__output_event_group_ids[station_id])
                    station_data['event_ids'] = np.array(self.__output_sub_event_ids[station_id])
                    station_data['maximum_amplitudes'] = np.array(self.__output_maximum_amplitudes[station_id])
                    station_data['maximum_amplitudes_envelope'] = np.array(self.__output_maximum_amplitudes_envelope[station_id])
                    station_data['triggered_per_event'] = np.array(self.__output_triggered_station[station_id])
        for (key, value) in iteritems(self.__meta_output):
            output_file.attrs[key] = value[saved_events_mask]

        output_file.attrs.create("Tnoise", self.__detector_simulator.get_noise_temperature(), dtype=np.float)
        output_file.attrs.create("Vrms", self.__detector_simulator.get_noise_vrms(), dtype=np.float)
        output_file.attrs.create("dt", 1. / self.__config['sampling_rate'], dtype=np.float)
        output_file.attrs.create("bandwidth", self.__detector_simulator.get_bandwidth(), dtype=np.float)
        first_channel_id = self.__detector.get_channel_ids(self.__station_ids[0])[0]
        n_samples = self.__detector.get_number_of_samples(
            self.__station_ids[0],
            first_channel_id
        ) / self.__detector.get_sampling_frequency(self.__station_ids[0], first_channel_id) * self.__config['sampling_rate']
        n_samples = int(np.ceil(n_samples / 2.) * 2)  # round to nearest even integer
        output_file.attrs['n_samples'] = n_samples
        output_file.attrs['config'] = yaml.dump(self.__config)

        # copy over data from input file
        for key in self.__input_data.keys():
            if not key.startswith('station_') and not key in output_file.keys():
                if np.array(self.__input_data[key]).dtype.char == 'U':
                    output_file[key] = np.array(self.__input_data[key], dtype=h5py.string_dtype(encoding='utf-8'))[saved_events_mask]

                else:
                    output_file[key] = np.array(self.__input_data[key])[saved_events_mask]

        for key in self.__input_attributes.keys():
            if not key in output_file.attrs.keys():  # only save atrributes sets that havn't been recomputed and saved already
                if key not in ["trigger_names", "Tnoise", "Vrms", "bandwidth", "n_samples", "dt", "detector", "config"]:  # don't write trigger names from input to output file, this will lead to problems with incompatible trigger names when merging output files
                    output_file.attrs[key] = self.__input_attributes[key]

        for key, value in iteritems(self.__meta_output_attributes):
            output_file.attrs[key] = value


        output_file.close()

    def __create_meta_output_datastructures(self):
        """
        creates the data structures of the parameters that will be saved into the hdf5 output file
        """
        self.__meta_output = {}
        self.__meta_output_attributes = {}
        n_showers = len(self.__input_data['event_group_ids'])
        self.__meta_output['weights'] = np.zeros(n_showers)
        self.__meta_output['triggered'] = np.zeros(n_showers, dtype=np.bool)
        #         self._mout['multiple_triggers'] = np.zeros((self._n_showers, self._number_of_triggers), dtype=np.bool)
        self.__meta_output_attributes['trigger_names'] = []
        self.__amplitudes = {}
        self.__amplitudes_envelope = {}
        self.__output_triggered_station = {}
        self.__output_event_group_ids = {}
        self.__output_sub_event_ids = {}
        self.__output_multiple_triggers_station = {}
        self.__output_maximum_amplitudes = {}
        self.__output_maximum_amplitudes_envelope = {}
        self.__output_trigger_times_station = {}
        self.__meta_output_groups = {}
        for station_id in self.__station_ids:
            self.__meta_output_groups[station_id] = {}
            self.__output_event_group_ids[station_id] = []
            self.__output_sub_event_ids[station_id] = []
            self.__output_triggered_station[station_id] = []
            self.__output_multiple_triggers_station[station_id] = []
            self.__output_maximum_amplitudes[station_id] = []
            self.__output_maximum_amplitudes_envelope[station_id] = []
            self.__output_trigger_times_station[station_id] = []

    def __create_station_output_structure(self, station_id):
        self.__output_station[station_id]['shower_id'] = []
        self.__output_station[station_id]['event_id_per_shower'] = []
        self.__output_station[station_id]['event_group_id_per_shower'] = []
        self.__output_station[station_id]['launch_vectors'] = []
        self.__output_station[station_id]['receive_vectors'] = []
        self.__output_station[station_id]['polarization'] = []
        self.__output_station[station_id]['travel_times'] = []
        self.__output_station[station_id]['travel_distances'] = []

    def __get_shower_index(self, shower_id):
        if hasattr(shower_id, "__len__"):
            return np.array([self.__shower_index_array[x] for x in shower_id])
        else:
            return self.__shower_index_array[shower_id]

    def __add_trigger_to_output(
            self,
            event_object,
            station,
            sub_event_shower_ids,
            event_indices,
            n_showers
    ):
        global_shower_indices = self.__get_shower_index(sub_event_shower_ids)
        local_shower_indices = np.atleast_1d(np.squeeze(np.argwhere(np.isin(event_indices, global_shower_indices, assume_unique=True))))
        station_id = station.get_id()
        extend_array = False
        for trigger in itervalues(station.get_triggers()):
            if trigger.get_name() not in self.__meta_output_attributes['trigger_names']:
                self.__meta_output_attributes['trigger_names'].append(trigger.get_name())
                extend_array = True
        if 'multiple_triggers' not in self.__meta_output:
            self.__meta_output['multiple_triggers'] = np.zeros((len(self.__input_data['event_group_ids']), len(self.__meta_output_attributes['trigger_names'])), dtype=np.bool)
            self.__meta_output['trigger_times'] = np.nan * np.zeros_like(self.__meta_output['multiple_triggers'], dtype=float)
        elif extend_array:
            tmp = np.zeros((len(self.__input_data['event_group_ids']), len(self.__meta_output_attributes['trigger_names'])), dtype=np.bool)
            nx, ny = self.__meta_output['multiple_triggers'].shape
            tmp[:, 0:ny] = self.__meta_output['multiple_triggers']
            self.__meta_output['multiple_triggers'] = tmp
        trigger_data = {
            'multiple_triggers': np.zeros((n_showers, len(self.__meta_output_attributes['trigger_names'])), dtype=np.bool),
            'trigger_times': np.full((n_showers, len(self.__meta_output_attributes['trigger_names'])), np.nan),
            'triggered': np.zeros(n_showers, dtype=bool)
        }
        self.__output_station[station_id]['trigger_times'] = np.full((n_showers, len(self.__meta_output_attributes['trigger_names'])), np.nan)
        self.__output_event_group_ids[station_id].append(event_object.get_run_number())
        self.__output_sub_event_ids[station_id].append(event_object.get_id())
        multiple_triggers = np.zeros(len(self.__meta_output_attributes['trigger_names']), dtype=np.bool)
        trigger_times = np.nan * np.zeros_like(multiple_triggers)
        for i_trigger, trigger_name in enumerate(self.__meta_output_attributes['trigger_names']):
            if station.has_trigger(trigger_name):
                multiple_triggers[i_trigger] = station.get_trigger(trigger_name).has_triggered()
                trigger_times[i_trigger] = station.get_trigger(trigger_name).get_trigger_time()
                for local_shower_index in local_shower_indices:  # now save trigger information per shower of the current station
                    trigger_data['multiple_triggers'][local_shower_index][i_trigger] = station.get_trigger(trigger_name).has_triggered()
                    trigger_data['trigger_times'][local_shower_index][i_trigger] = trigger_times[i_trigger]
                    self.__output_station[station_id]['trigger_times'][local_shower_index][i_trigger] = trigger_times[i_trigger]

        for local_index, global_index in zip(local_shower_indices, global_shower_indices):  # now save trigger information per shower of the current station
            trigger_data['triggered'][local_index] = np.any(trigger_data['multiple_triggers'][local_index])
            self.__meta_output['triggered'][global_index] |= trigger_data['triggered'][local_index]
            self.__meta_output['multiple_triggers'][global_index] |= trigger_data['multiple_triggers'][local_index]
            self.__meta_output['trigger_times'][global_index] = np.fmin(
                self.__meta_output['trigger_times'][global_index],
                trigger_data['trigger_times'][local_index]
            )

        self.__output_multiple_triggers_station[station_id].append(multiple_triggers)
        self.__output_trigger_times_station[station_id].append(trigger_times)
        self.__output_triggered_station[station_id].append(np.any(multiple_triggers))




