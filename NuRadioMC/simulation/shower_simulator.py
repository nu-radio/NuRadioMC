import numpy as np
from NuRadioReco.utilities import units

class showerSimulator:
    def __init__(
            self,
            detector,
            channel_ids,
            config,
            input_data,
            channel_efield_simulator
    ):
        self.__detector = detector
        self.__channel_ids = channel_ids
        self.__config = config
        self.__input_data = input_data
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
            self.__distance_cut_polynomial = np.polynomial.polynomial(self.__config['speedup']['distance_cut_coefficients'])
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
            event_indices
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
            self.__event_group_vertex_distances - self.__event_group_vertex_distances[0], axis=1
        )
        self.__event_group_shower_energies = self.__input_data['energies'][event_indices]

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
        vertex_time = self.__input_data['vertex_times'][shower_index]
        if not self.__distance_cut(shower_vertex):
            return

    def __distance_cut(
            self,
            shower_vertex
    ):
        if not self.__config['speedup']['distance_cut']:
            return True

        mask_shower_sum = np.abs(
            self.__event_group_vertex_distances - self.__event_group_vertex_distances[iSh]
        ) < self.__config['speedup']['distance_cut_sum_length']
        shower_energy_sum = np.sum(shower_energies[mask_shower_sum])
        distance_to_station = np.linalg.norm(shower_vertex - self.__station_barycenter)
        if shower_energy_sum <= 0:
            distance_cut_value = 200. * units.m
        else:
            distance_cut_value = np.max(
                100. * units.m,
                10. ** self.__distance_cut_polynomial(np.log10(shower_energy_sum))
            ) + 100. * units.m
        return distance_to_station <= distance_cut_value