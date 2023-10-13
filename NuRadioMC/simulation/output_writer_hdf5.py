import numpy as np
import h5py
import yaml
import collections
from six import iteritems, itervalues
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.utilities import units


class outputWriterHDF5:
    """
    Class to write output HDF5 files
    """
    def __init__(
            self,
            output_filename,
            config,
            detector,
            station_ids,
            raytracer,
            detector_simulator,
            input_filename,
            particle_mode
    ):
        """
        Initialize class

        Parameters
        ----------
        output_filename: string
            Name of the output file
        config: dict
            Dictionary containing the contents of the configuration .yaml file
        detector: NuRadioReco.detector.detector.Detector or NuRadioReco.detector.generic_detector.GenericDetector object
            Object conaining the detector description
        station_ids: list of integers
            List containing the IDs of all stations that are simulated
        raytracer: NuRadioMC.SignalProp.analyticraytracing.ray_tracing object or similar
            Object that can perform the propagation fromt the shower to the detector. Has to follow the template
            specified in NuRadioMC.SignalProp.propagation_base_class. Options are the analytic raytracer, RadioPropa of the 
            direct line raytracer.
        detector_simulator: NuRadioMC.simulation.hardware_response_simulator.hardwareResponseSimulator object
        input_filename: string
            The name of the file containing the simulation input.
        particle_mode: boolean
            Specifies if the events simulated are from particles or other sources (e.g. pulsers)
        """
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
        self.__particle_mode = particle_mode
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
        self.__output_attributes = {}
        self.__output = {}
        self.__create_output_data_structure()
        self.__output_triggered_station = {}
        self.__station_property_names = [
            'focusing_factor',
            'launch_vectors',
            'polarization',
            'ray_tracing_C0',
            'ray_tracing_C1',
            'ray_tracing_reflection',
            'ray_tracing_reflection_case',
            'ray_tracing_solution_type',
            'receive_vectors',
            'travel_distances',
            'travel_times'
        ]
        self.__hardware_property_names = [
            'max_amp_shower_and_ray',
            'time_shower_and_ray'
        ]

        input_file.close()

    def add_station(
            self,
            station_id,
            event_objects,
            station_objects,
            simulation_results,
            hardware_response_sim_results,
            event_group_id,
            sub_event_shower_ids,
            station_has_triggered
    ):
        """
        Writes simulation results for a station into the output data structure.
        This method writes properties that are stored per triggered event.

        Parameters:
        -----------
        station_id: integer
            The ID of the simulated station
        event_objects: dict 
            A dictionary containing the sub-events that are written into the output structure
        station_objects: dict
            A dictionary containing the station that are written into the output structure.
        simulation_results: dict
            A dictionary of arrays containing results of the event simulation.
            Most arrays are 3-dimensional, with the first dimension specifying the sub-event, the second
            the channel and the third the raytracing solution.
        hardware_response_sim_results: dict
            A dictionary of arrays containing results of the detector response simulation.
            Arrays are 3-dimensional, with the first dimension specifying the sub-event, the second
            the channel and the third the raytracing solution.
        event_group_id: integer
            The ID of the event group
        sub_event_shower_ids: List of integers with shape (n_sub_events, n_showers)
            A list containing the IDs of the showers that are parts of the events.
        station_has_triggered: List of booleans with shape (n_sub_events)
            Specifies which sub-events have triggered
        """
        trigger_indices = np.where(station_has_triggered)[0]
        if station_id not in self.__output_station.keys():
            self.__output_station[station_id] = {}
            self.__create_station_output_structure(station_id)
        elif 'shower_id' not in self.__output_station[station_id].keys():
            self.__create_station_output_structure(station_id)
        event_indices = np.atleast_1d(np.squeeze(np.argwhere(self.__input_data['event_group_ids'] == event_group_id)))
        for event_key in event_objects.keys():
            if 'trigger_names' not in self.__output_attributes.keys():
                self.__write_trigger_names(station_objects[event_key])
                self.__create_output_data_structure_for_triggers()
            if station_objects[event_key].has_triggered():
                self.__output_station[station_id]['event_group_ids'].append(event_group_id)
                self.__output_station[station_id]['event_ids'].append(event_objects[event_key].get_id())
                self.__add_trigger_to_output(
                    station_objects[trigger_indices[0]],
                    sub_event_shower_ids[trigger_indices[0]],
                    event_indices,
                    station_has_triggered
                )
                maximum_amplitudes = np.zeros(len(station_objects[event_key].get_channel_ids()))
                maximum_amplitudes_envelope = np.zeros(len(station_objects[event_key].get_channel_ids()))
                for i_channel, channel_id in enumerate(station_objects[event_key].get_channel_ids()):
                    maximum_amplitudes[i_channel] = station_objects[event_key].get_channel(channel_id).get_parameter(chp.maximum_amplitude)
                    maximum_amplitudes_envelope[i_channel] = station_objects[event_key].get_channel(channel_id).get_parameter(chp.maximum_amplitude_envelope)
                self.__output_station[station_id]['maximum_amplitudes'].append(maximum_amplitudes)
                self.__output_station[station_id]['maximum_amplitudes_envelope'].append(maximum_amplitudes_envelope)
        
    def add_station_per_shower(
        self,
        station_id,
        event_objects,
        station_objects, 
        simulation_results,
        hardware_response_sim_results,
        event_group_id,
        sub_event_shower_id
    ):
        """
        Writes simulation results for a station into the output data structure.
        This method writes properties that are stored per shower.

        Parameters:
        -----------
        station_id: integer
            The ID of the simulated station
        event_objects: dict 
            A dictionary containing the sub-events that are written into the output structure
        station_objects: dict
            A dictionary containing the station that are written into the output structure.
        simulation_results: dict
            A dictionary of arrays containing results of the event simulation.
            Most arrays are 3-dimensional, with the first dimension specifying the sub-event, the second
            the channel and the third the raytracing solution.
        hardware_response_sim_results: dict
            A dictionary of arrays containing results of the detector response simulation.
            Arrays are 3-dimensional, with the first dimension specifying the sub-event, the second
            the channel and the third the raytracing solution.
        event_group_id: integer
            The ID of the event group
        sub_event_shower_ids: List of integers with shape (n_sub_events, n_showers)
            A list containing the IDs of the showers that are parts of the events.
        """
        if station_id not in self.__output_station.keys():
            self.__create_station_output_structure()
        # If there are multiple triggers, only the trigger times of the last station that triggered is stored.
        trigger_station = None
        for station_key in station_objects.keys():
            stn = station_objects[station_key]
            if stn.has_triggered():
                trigger_station = stn
        if trigger_station is None:
            trigger_station = station_objects[station_objects.keys()[-1]]
        for i_sub_shower in range(simulation_results['shower_id'].shape[0]):
            if 'event_group_id_per_shower' not in self.__output_station[station_id].keys():
                self.__output_station[station_id]['event_group_id_per_shower'] = [event_group_id]
            else:
                self.__output_station[station_id]['event_group_id_per_shower'].append(event_group_id)
            if 'event_id_per_shower' not in self.__output_station[station_id].keys():
                self.__output_station[station_id]['event_id_per_shower'] = [simulation_results['event_id_per_shower'][i_sub_shower]]
            else:
                self.__output_station[station_id]['event_id_per_shower'].append(simulation_results['event_id_per_shower'][i_sub_shower])
            for station_property_name in self.__station_property_names:
                if station_property_name not in self.__output_station[station_id].keys():
                    self.__output_station[station_id][station_property_name] = [simulation_results[station_property_name][i_sub_shower]]
                else:
                    self.__output_station[station_id][station_property_name].append(simulation_results[station_property_name][i_sub_shower])
            for property_name in self.__hardware_property_names:
                if property_name not in self.__output_station[station_id].keys():
                    self.__output_station[station_id][property_name] = [hardware_response_sim_results[property_name][i_sub_shower]]
                else:
                    self.__output_station[station_id][property_name].append(hardware_response_sim_results[property_name][i_sub_shower])
            self.__add_trigger_to_output_per_shower(
                trigger_station,
            )

    def store_event_group_weight(
        self,
        weight,
        event_indices
    ):
        """
        Stores the event weight on the output data structure.

        Parameters
        ----------
        weight: float
            The weight of the event
        event_indices: numpy.array of integers
            The indices of the events whose weights are stored, i.e. their position in the input data
        """
        self.__output['weights'][event_indices] = weight

    def save_output(self):
        """
        Writes the data in the output data structure into an HDF5 file.
        """
        output_file = h5py.File(self.__filename, 'w')
        saved_events_mask = np.copy(self.__output['triggered'])
        if 'n_interactions' in self.__input_data:  # if n_interactions is not specified, there are no parents
            parent_mask = self.__input_data['n_interaction'] == 1
            for event_id in np.unique(self.__input_data['event_group_ids']):
                event_mask = self.__input_data['event_group_ids'] == event_id
                if True in self.__output['triggered'][event_mask]:
                    saved_events_mask[parent_mask & event_mask] = True

        for station_key, val in iteritems(self.__output_station):
            output_group = output_file.create_group('station_{:d}'.format(station_key))
            for key, value in iteritems(val):
                if key in ['event_group_id_per_shower', 'event_id_per_shower', 'event_ids', 'event_group_ids']:
                    output_group[key] = np.array(value, dtype=int)
                elif key in ['multiple_triggers', 'multiple_triggers_per_event']:
                    output_group[key] = np.array(value, dtype=np.bool)
                else:
                    output_group[key] = np.array(value, dtype=float)
        if 'trigger_names' in self.__output_attributes:
            n_triggers = len(self.__output_attributes['trigger_names'])
            for station_id in self.__output_groups:
                n_events_for_station = len(self.__output_triggered_station[station_id])
                if n_events_for_station > 0:
                    station_data = output_file['station_{:d}'.format(station_id)]
                    # station_data['triggered_per_event'] = np.array(self.__output_triggered_station[station_id])
        for (key, value) in iteritems(self.__output):
            if value.dtype.char == 'U':
                output_file[key] = np.array(value[saved_events_mask], dtype=h5py.string_dtype(encoding='utf-8'))
            else:
                output_file[key] = value[saved_events_mask]

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

        for key in self.__input_attributes.keys():
            if not key in output_file.attrs.keys():  # only save atrributes sets that havn't been recomputed and saved already
                if key not in ["trigger_names", "Tnoise", "Vrms", "bandwidth", "n_samples", "dt", "detector", "config"]:  # don't write trigger names from input to output file, this will lead to problems with incompatible trigger names when merging output files
                    output_file.attrs[key] = self.__input_attributes[key]

        for key, value in iteritems(self.__output_attributes):
            output_file.attrs[key] = value

        output_file.close()

    def __create_output_data_structure_for_triggers(self):
        """
        Creates empty arrays into which trigger simulation results can be written.
        """
        self.__output['multiple_triggers'] = np.zeros((self.__input_data['shower_ids'].shape[0], len(self.__output_attributes['trigger_names'])), dtype=bool)
        self.__output['trigger_times'] = np.zeros((self.__input_data['shower_ids'].shape[0], len(self.__output_attributes['trigger_names'])))

    def __create_output_data_structure(self):
        """
        Creates the data structure into which the simulation results at event level can be written.
        """
        if self.__particle_mode:
            self.__output['azimuths'] = np.array(self.__input_data['azimuths'])
            self.__output['energies'] = np.array(self.__input_data['energies'])
            self.__output['flavors'] = np.array(self.__input_data['flavors'])
            self.__output['inelasticity'] = np.array(self.__input_data['inelasticity'])
            self.__output['interaction_type'] = np.array(self.__input_data['interaction_type'])
            self.__output['n_interaction'] = np.array(self.__input_data['n_interaction'])
            self.__output['shower_energies'] = np.array(self.__input_data['shower_energies'])
            self.__output['zeniths'] = np.array(self.__input_data['zeniths'])
            self.__output['shower_type'] = np.array(self.__input_data['shower_type'])
            self.__output['vertex_times'] = np.array(self.__input_data['vertex_times'])
        else:
            self.__output['emitter_amplitudes'] = np.array(self.__input_data['emitter_amplitudes'])
            self.__output['emitter_antenna_type'] = np.array(self.__input_data['emitter_antenna_type'])
            self.__output['emitter_frequency'] = np.array(self.__input_data['emitter_frequency'])
            self.__output['emitter_half_width'] = np.array(self.__input_data['emitter_half_width'])
            self.__output['emitter_model'] = np.array(self.__input_data['emitter_model'])
            self.__output['emitter_orientation_phi'] = np.array(self.__input_data['emitter_orientation_phi'])
            self.__output['emitter_orientation_theta'] = np.array(self.__input_data['emitter_orientation_theta'])
            self.__output['emitter_rotation_phi'] = np.array(self.__input_data['emitter_rotation_phi'])
            self.__output['emitter_rotation_theta'] = np.array(self.__input_data['emitter_rotation_theta'])
        self.__output['event_group_ids'] = np.array(self.__input_data['event_group_ids'])
        self.__output['shower_ids'] = np.array(self.__input_data['shower_ids'])
        self.__output['triggered'] = np.zeros(self.__input_data['shower_ids'].shape, dtype=bool)
        self.__output['weights'] = np.zeros(self.__input_data['shower_ids'].shape)
        self.__output['xx'] = np.array(self.__input_data['xx'])
        self.__output['yy'] = np.array(self.__input_data['yy'])
        self.__output['zz'] = np.array(self.__input_data['zz'])
        
    def __create_station_output_structure(self, station_id):
        """
        Creates the data structure into which the simulation results for a specific station can be written.

        Parameters
        ----------
        station_id: integer
            The ID of the station for which the output strucure is created.
        """
        self.__output_station[station_id]['shower_id'] = []
        self.__output_station[station_id]['event_ids'] = []
        self.__output_station[station_id]['event_id_per_shower'] = []
        self.__output_station[station_id]['event_group_ids'] = []
        self.__output_station[station_id]['event_group_id_per_shower'] = []
        self.__output_station[station_id]['multiple_triggers'] = []
        self.__output_station[station_id]['multiple_triggers_per_event'] = []
        self.__output_station[station_id]['maximum_amplitudes'] = []
        self.__output_station[station_id]['maximum_amplitudes_envelope'] = []
        self.__output_station[station_id]['launch_vectors'] = []
        self.__output_station[station_id]['receive_vectors'] = []
        self.__output_station[station_id]['polarization'] = []
        self.__output_station[station_id]['travel_times'] = []
        self.__output_station[station_id]['travel_distances'] = []
        self.__output_station[station_id]['trigger_times'] = []
        self.__output_station[station_id]['trigger_times_per_event'] = []
        self.__output_triggered_station[station_id] = []

    def __write_trigger_names(self, station_object):
        """
        Writes the names of all triggers in the station into the output attributes

        Parameters
        ----------
        station_object: NuRadioReco.framework.station.Station object
            The station whose triggers are stored
        """
        self.__output_attributes['trigger_names'] = []
        for trigger_name in station_object.get_triggers():
            self.__output_attributes['trigger_names'].append(trigger_name)
        
    def __get_shower_index(self, shower_id):
        """
        Finds the index (i.e. its position in the input data) of a shower Id

        Parameters
        ----------
        shower_id: integer or numpy.array of integers
            The shower ID (or IDs) whose index is returned
        """
        if hasattr(shower_id, "__len__"):
            return np.array([self.__shower_index_array[x] for x in shower_id])
        else:
            return self.__shower_index_array[shower_id]

    def __add_trigger_to_output(
        self,
        station,
        sub_event_shower_ids,
        event_indices,
        has_triggered
    ):
        """
        Writes results of the trigger simulation into the output structure

        Parameters
        ----------
        station: NuRadioReco.framework.station.Station object
            The station holding the triggers that are added to the output
        sub_event_shower_ids: list of integers
            IDs of the showers belonging to the events whose triggers are saved.
        event_indices: list of integers
            The indices (i.e. their positions in the input file) of the events whose triggers are saved
        has_triggered: List of booleans
            Specifies of the events have triggered.
        """
        global_shower_indices = self.__get_shower_index(sub_event_shower_ids)
        self.__output['triggered'][global_shower_indices] = np.any(has_triggered) or self.__output['triggered'][global_shower_indices]
        for trigger in itervalues(station.get_triggers()):
            if trigger.get_name() not in self.__output_attributes['trigger_names']:
                self.__output_attributes['trigger_names'].append(trigger.get_name())
        self.__output_triggered_station[station.get_id()].append(np.any(has_triggered))
        multiple_triggers = np.zeros(len(self.__output_attributes['trigger_names']), dtype=np.bool)
        trigger_times = np.zeros(len(self.__output_attributes['trigger_names']))
        for i_trigger, trigger_name in enumerate(self.__output_attributes['trigger_names']):
            if station.has_trigger(trigger_name):
                multiple_triggers[i_trigger] = station.get_trigger(trigger_name).has_triggered()
                trigger_times[i_trigger] = station.get_trigger(trigger_name).get_trigger_time()
        self.__output_station[station.get_id()]['multiple_triggers_per_event'].append(multiple_triggers)
        self.__output_station[station.get_id()]['trigger_times_per_event'].append(trigger_times)
        self.__output['multiple_triggers'][global_shower_indices] = multiple_triggers
        self.__output['trigger_times'][global_shower_indices] = trigger_times

    def __add_trigger_to_output_per_shower(
        self,
        station
    ):
        """
        Stores results of the trigger simulation that are stored for every shower

        Parameters
        ----------
        station: NuRadioReco.framework.station.Station object
            The station holding the triggers to be stored.
        """
        multiple_triggers = np.zeros(len(self.__output_attributes['trigger_names']), dtype=np.bool)
        trigger_times = np.zeros(len(self.__output_attributes['trigger_names']))
        for i_trigger, trigger_name in enumerate(self.__output_attributes['trigger_names']):
            if station.has_trigger(trigger_name):
                multiple_triggers[i_trigger] = station.get_trigger(trigger_name).has_triggered()
                trigger_times[i_trigger] = station.get_trigger(trigger_name).get_trigger_time()
        self.__output_station[station.get_id()]['multiple_triggers'].append(multiple_triggers)
        self.__output_station[station.get_id()]['trigger_times'].append(trigger_times)

    def get_trigger_status(self):
        return self.__output['triggered']

    def get_weights(self):
        return self.__output['weights']

