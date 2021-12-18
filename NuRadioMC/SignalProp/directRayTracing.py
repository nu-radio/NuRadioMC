import scipy.constants
import numpy as np
from NuRadioReco.utilities import units
from NuRadioMC.SignalProp.propagation_base_class import ray_tracing_base
from NuRadioMC.SignalProp.propagation import solution_types_revert

import logging
logging.basicConfig()



speed_of_light = scipy.constants.c * units.m / units.s



class direct_ray_tracing(ray_tracing_base):
        
    def find_solutions(self):
        results = []
        for iS in range(self.get_number_of_solutions()):
            results.append({'type': self.get_solution_type(iS), 'reflection':0})
        self._results = results
        return results    
    
    def get_launch_vector(self, iS):
        launch_vector = self._X2 - self._X1  
        return launch_vector 
    
    def get_number_of_solutions(self):
        return 1
    
    def get_solution_type(self, iS):
        return solution_types_revert['direct']
    
    def get_path(self, iS, n_points = 1000):
        delta_x =(self._X2-self._X1)/(n_points-1)
        path = self._X1[None] + np.arange(n_points)[:, None] * delta_x[None]
        return path
    
    def get_receive_vector(self, iS):
        receive_vector = self._X1 - self._X2
        return receive_vector
    
    def get_path_length(self, iS):
        path_length = np.linalg.norm(self._X2 - self._X1)
        return path_length 
    
    def get_travel_time(self, iS):
        traveltime = 0
        path = self.get_path(iS)
        segment = [path[0][1]-path[0][0],path[1][1]-path[1][0],path[2][1]-path[2][0]]
        r = np.linalg.norm(segment)
        for i in range(len(path[0])-1):
            xx = np.array([path[0][i], path[1][i], path[2][i]])
            yy = np.array([path[0][i+1], path[1][i+1], path[2][i+1]])
            x = (xx+yy)/2
            n = self._medium.get_index_of_refraction(x)
            traveltime += r/(speed_of_light/n)
        return traveltime
    
    
    def get_reflection_angle(self):
        return None 
    
    def apply_propagation_effects(self, efield, iS):
        return efield

    def get_output_parameters(self):
        return [
            {'name': 'ray_tracing_solution_type', 'ndim': 1}
        ]

    def get_raytracing_output(self, i_solution):
        return {
            'ray_tracing_solution_type': self.get_solution_type(i_solution)
        }