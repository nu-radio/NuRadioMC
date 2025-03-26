import numpy as np
from NuRadioReco.utilities import units
from scipy.constants import c as speed_of_light


class TravelTimeSimulator:
    def __init__(
            self,
            ice_model
    ):
        self.__ice_model = ice_model

    def get_time_differences(self, ch_1_pos, ch_2_pos, pulser_pos):
        d_1 = pulser_pos - ch_1_pos
        d_2 = pulser_pos - ch_2_pos
        if ch_1_pos.ndim > 1:
            length_1 = np.sqrt(np.sum(d_1 ** 2, axis=1))
            length_2 = np.sqrt(np.sum(d_2 ** 2))
            ior1 = np.zeros_like(ch_1_pos[:,0])
            for ip, pos in enumerate(ch_1_pos):
                ior1[ip] = self.__ice_model.get_average_index_of_refraction(pos,pulser_pos)
            ior2 = self.__ice_model.get_average_index_of_refraction(ch_2_pos,pulser_pos)
            t_1 = length_1 * ior1 / (speed_of_light * units.m / units.s)
            t_2 = length_2 * ior2 / (speed_of_light * units.m / units.s)
            return t_1 - t_2

        else:
            length_1 = np.sqrt(np.sum(d_1 ** 2))
            length_2 = np.sqrt(np.sum(d_2 ** 2))
            ior1 = self.__ice_model.get_average_index_of_refraction(ch_1_pos,pulser_pos)
            ior2 = self.__ice_model.get_average_index_of_refraction(ch_2_pos,pulser_pos)
            t_1 = length_1 * ior1 / (speed_of_light * units.m / units.s)
            t_2 = length_2 * ior2 / (speed_of_light * units.m / units.s)
            return t_1 - t_2

    def get_travel_time(self, channel_pos, pulser_pos):
        travel_dist = np.sqrt(np.sum((channel_pos - pulser_pos)**2))
        ior = self.__ice_model.get_average_index_of_refraction(channel_pos, pulser_pos)
        # if np.isnan(ior):
        #     print(channel_pos, pulser_pos)
        return travel_dist * ior / (speed_of_light * units.m / units.s)
