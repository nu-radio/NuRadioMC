import numpy as np
#import NuRadioMC.SignalProp.radioproparaytracing
from NuRadioMC.SignalProp import analyticraytracing
import matplotlib.pyplot as plt
import NuRadioMC.utilities.medium
import NuRadioMC.utilities.medium_base
#import AntPosCal.ice_model.icemodel
from NuRadioReco.utilities import units
#import radiopropa
from NuRadioMC.utilities.medium_base import IceModelSimple

class TravelTimeSimulatorRadioPropa:
    def __init__(
            self,
            ice_model
    ):

        self.__ice_model = ice_model
        if isinstance(ice_model, IceModelSimple):
            self.__raytracer = analyticraytracing.ray_tracing(ice_model)
        else:
            self.__raytracer = radioproparaytracing.radiopropa_ray_tracing(ice_model)
            self.__raytracer.set_iterative_sphere_sizes(np.array([25, 2., .5, .2]))
            self.__raytracer.set_iterative_step_sizes(np.array([.5, .05, .005, .001]) * units.deg)

    def get_travel_time(self, start, end):
        self.__raytracer.set_start_and_end_point(list(start), list(end))
        self.__raytracer.find_solutions()
        t_prop = np.nan
        for i_result, result in enumerate(self.__raytracer.get_results()):
            t_solution = self.__raytracer.get_travel_time(i_result)

            if np.isnan(t_prop) or np.min(t_solution) < t_prop:
                t_prop = np.min(t_solution)
        return t_prop

    def get_time_differences(
            self,
            ch_1_pos,
            ch_2_pos,
            pulser_pos
    ):
        t_2 = self.get_travel_time(pulser_pos, ch_2_pos)
        if ch_1_pos.ndim > 1:
            time_differences = np.zeros(ch_1_pos.shape[0])
            for i_pos, pos in enumerate(ch_1_pos):
                time_differences[i_pos] = self.get_travel_time(pulser_pos, pos) - t_2
        else:
            time_differences = self.get_travel_time(pulser_pos, ch_1_pos) - t_2
        return time_differences
