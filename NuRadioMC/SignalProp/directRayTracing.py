import scipy.constants
import numpy as np
from NuRadioReco.utilities import units




solution_types = {1: 'direct',
                      2: 'refracted',
                      3: 'reflected'}
    
speed_of_light = scipy.constants.c * units.m / units.s



class directRayTracing():

    def __init__(self, attenuation_model = "SP1", medium):
        self._medium = medium
        self._attenuation_model = attenuation_model
        pass
       
    def set_start_and_end_point(self, x1, x2):
        x1 = np.array(x1, dtype =np.float)
        x2 = np.array(x2, dtype = np.float)
        self._x1 = x1
        self._x2 = x2

        
    def find_solutions(self):
        results = []
        solution_type = 1
        results.append({'type': solution_type})
        self._results = results
        return results
    
    def has_solution(self):
        return len(self._results) > 0    
    
    
    
    def get_launch_vector(self, iS):
        launch_vector = self._x2 - self._x1  
        return launch_vector 
    
    
    def get_number_of_solutions(self):
        return len(self._results)
    

    def get_results(self):
        return self._results
    
    def get_solution_type(self, iS):
        return 1
    
    def get_path(self, iS, n_points = 1000):
        delta_x = self._x2-self._x1/n_points
        path = [[],[],[]]
        for i in range(n_points+1):
            for j in range(3):
                path[j].append(self._x1[j] + i*delta_x[j])
                
        return path
    
    def get_receive_vector(self, iS):
        receive_vector = self._x1 - self._x2
        return receive_vector
    
    def get_path_length(self):
        path_length = np.linalg.norm(self._x2 - self._x1)
        return path_length 
    
    def get_travel_time(self):
        traveltime = 0
        path = get_path(iS)
        segment = [path[0][1]-path[0][0],path[1][1]-path[1][0],path[2][1]-path[2][0]]
        r = np.linalg.norm(segment)
        for i in range(len(path[0])-1):
            xx = [path[0][i], path[1][i], path[2][i]]
            yy = [path[0][i+1], path[1][i+1], path[2][i+1]]
            x = (xx+yy)/2
            n = self._medium.get_index_of_refraction(x)
            traveltime += r/(speed_of_light/n)
        return traveltime
    
    
    def get_reflection_angle(self):
        return None 
    
    
    def apply_propagation_effects(self, iS, frequency, efield):
        mask = frequency > 0
        attenuation = np.ones_like(frequency)
        attenuation[mask] = np.interp(frequency[mask], freqs, tmp)
        return efield
    
    
    def get_number_of_solutions(self):
        return len(self._results)
    
    def check_if_presimulated(self):
        return 
    
    
    def create_output_data_structure(self, dictionary, n_showers, n_antennas):
        nS = self.get_number_of_solutions()
        dictionary['ray_tracing_solution_type'] = np.ones((n_showers, n_antennas, nS), dtype=np.int) * -1
        
    def write_raytracing_output(self, dictionary, i_shower, channel_id, i_solution):
        dictionary['ray_tracing_solution_type'][i_shower, channel_id, i_solution] = self.get_solution_type(i_solution)

    