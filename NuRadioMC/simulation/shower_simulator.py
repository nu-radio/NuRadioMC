import numpy as np
import radiotools.helper
from NuRadioReco.utilities import units

class showerSimulator:
    def __init__(
            self,
            detector,
            channel_ids,
            config,
            input_data,
            input_attributes,
            channel_efield_simulator,
            n_raytracing_solutions
    ):
        self.__detector = detector
        self.__channel_ids = channel_ids
        self.__config = config
        self.__input_data = input_data
        self.__input_attributes = input_attributes
        self.__n_raytracing_solutions = n_raytracing_solutions
        self.__station_id = None
        self.__event_group_id = None
        self.__i_event_group = None
        self.__event_indices = None
        self.__event_group_vertices = None
        self.__event_group_vertex_distances = None
        self.__event_group_shower_energies = None
        self.__station_barycenter = None
        self.__channel_efield_simulator = channel_efield_simulator
        if self.__config['speedup']['distance_cut']:
            self.__distance_cut_polynomial = np.polynomial.polynomial.Polynomial(self.__config['speedup']['distance_cut_coefficients'])
        else:
            self.__distance_cut_polynomial = None


    def set_station(
            self,
            station_id
    ):
        self.__station_id = station_id
        self.__station_barycenter = np.zeros(3)
        for channel_id in self.__detector.get_channel_ids(station_id):
            self.__station_barycenter += self.__detector.get_relative_position(station_id, channel_id)
        self.__station_barycenter /= len(self.__detector.get_channel_ids(station_id))
        self.__station_barycenter += self.__detector.get_absolute_position(station_id)


    def set_event_group(
            self,
            i_event_group,
            event_group_id,
            event_indices,
            particle_mode
    ):
        self.__i_event_group = i_event_group
        self.__event_group_id = event_group_id
        self.__event_indices = event_indices
        self.__event_group_vertices = np.array([
            np.array(self.__input_data['xx'])[event_indices],
            np.array(self.__input_data['yy'])[event_indices],
            np.array(self.__input_data['zz'])[event_indices]
        ]).T
        self.__event_group_vertex_distances = np.linalg.norm(
            self.__event_group_vertices - self.__event_group_vertices[0], axis=1
        )
        if particle_mode:
            self.__event_group_shower_energies = self.__input_data['energies'][event_indices]
        self.__channel_efield_simulator.set_event_group(np.sum(self.__event_group_shower_energies))

    def simulate_shower(
            self,
            shower_id,
            shower_index,
            pre_simulated,
            ray_tracing_performed

    ):
        self.__channel_efield_simulator.set_shower(
            self.__station_id,
            shower_id,
            shower_index,
            pre_simulated,
            ray_tracing_performed
        )
        shower_vertex = np.array([
            self.__input_data['xx'][shower_index],
            self.__input_data['yy'][shower_index],
            self.__input_data['zz'][shower_index]
        ])
        if not self.__distance_cut(shower_vertex):
            return [], [], [], [], [], [], [], []
        if not self.__in_fiducial_volume(shower_vertex):
            return [], [], [], [], [], [], [], []
        if self.__config['signal']['shower_type'] == "em":
            if self.__config['shower_type'][shower_index] != "em":
                return [], [], [], [], [], [], [], []
        if self.__config['signal']['shower_type'] == "had":
            if self.__config['shower_type'][shower_index] != "had":
                return [], [], [], [], [], [], [], []
        n_channels = len(self.__channel_ids)
        efield_list = []
        launch_vector_list = np.full((n_channels, self.__n_raytracing_solutions, 3), np.nan)
        receive_vector_list =  np.full((n_channels, self.__n_raytracing_solutions, 3), np.nan)
        travel_time_list = np.full((n_channels, self.__n_raytracing_solutions), np.nan)
        path_length_list = np.full((n_channels, self.__n_raytracing_solutions), np.nan)
        polarization_direction_list = np.full((n_channels, self.__n_raytracing_solutions, 3), np.nan)
        efield_amplitude_list = np.full((n_channels, self.__n_raytracing_solutions), np.nan)
        raytracing_output_list = {}
        for i_channel, channel_id in enumerate(self.__channel_ids):
            efield_objects, launch_vectors, receive_vectors, travel_times, path_lenghts, polarization_directions, \
                efield_amplitudes, raytracing_output = self.__channel_efield_simulator.simulate_efield_at_channel(
                channel_id
            )
            n_efield_entries = path_lenghts.shape[0]
            efield_list.append(efield_objects)
            launch_vector_list[i_channel, :n_efield_entries] = launch_vectors
            receive_vector_list[i_channel, :n_efield_entries] = receive_vectors
            travel_time_list[i_channel, :n_efield_entries] = travel_times
            path_length_list[i_channel, :n_efield_entries] = path_lenghts
            polarization_direction_list[i_channel, :n_efield_entries] = polarization_directions
            efield_amplitude_list[i_channel, :n_efield_entries] = efield_amplitudes
            for key in raytracing_output:
                if key not in raytracing_output_list:
                    raytracing_output_list[key] = np.full((n_channels, self.__n_raytracing_solutions), np.nan)
                raytracing_output_list[key][i_channel, :n_efield_entries] = raytracing_output[key]
        return efield_list, launch_vector_list, receive_vector_list, travel_time_list, path_length_list, \
                polarization_direction_list, efield_amplitude_list, raytracing_output_list

    def __distance_cut(
            self,
            shower_vertex
    ):
        if not self.__config['speedup']['distance_cut']:
            return True

        mask_shower_sum = np.abs(
            self.__event_group_vertex_distances - self.__event_group_vertex_distances[0]
        ) < self.__config['speedup']['distance_cut_sum_length']
        shower_energy_sum = np.sum(self.__event_group_shower_energies[mask_shower_sum])
        distance_to_station = np.linalg.norm(shower_vertex - self.__station_barycenter)
        if shower_energy_sum <= 0:
            distance_cut_value = 200. * units.m
        else:
            distance_cut_value = np.max([
                100. * units.m,
                10. ** self.__distance_cut_polynomial(np.log10(shower_energy_sum))
            ]) + 100. * units.m
        return distance_to_station <= distance_cut_value

    def __in_fiducial_volume(
            self,
            shower_vertex
    ):
        """
        checks whether a vertex is in the fiducial volume

        if the fiducial volume is not specified in the input file, True is returned (this is required
        for the simulation of pulser calibration measuremens)
        """
        parameter_names = ['fiducial_rmin', 'fiducial_rmax', 'fiducial_zmin', 'fiducial_zmax']
        for parameter_name in parameter_names:
            if parameter_name not in self.__input_attributes:
                return True

        r = (shower_vertex[0] ** 2 + shower_vertex[1] ** 2) ** 0.5
        if self.__input_attributes['fiducial_rmin'] <= r <= self.__input_attributes['fiducial_rmax']:
            if self.__input_attributes['fiducial_zmin'] <= shower_vertex[2] <= self.__input_attributes['fiducial_zmax']:
                return True
        return False