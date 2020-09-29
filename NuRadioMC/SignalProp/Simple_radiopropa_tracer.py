from __future__ import absolute_import, division, print_function
import numpy as np
from radiotools import helper as hp
import logging
from NuRadioMC.utilities import attenuation as attenuation_util
from scipy import interpolate
import NuRadioReco.utilities.geometryUtilities
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import electricFieldParameters as efp
import radiopropa
logging.basicConfig()
import radiopropa

"""
STILL TO DO: 
- compare timing, distances, receive and launch angles wtr analytic raytracer
- compare attenuation wtr analytic raytracer 
- proper implementatin of the icemodel 
- Add warnings when the icemodel is not implemented in radiopropa
- Timing reduction; everything takes too long now
- compare waveforms wtr analytic raytracer
"""

def getPathCandidate(Candidate):
    pathx = np.fromstring(Candidate.getPathX()[1:-1],sep=',')
    pathy = np.fromstring(Candidate.getPathY()[1:-1],sep=',')
    pathz = np.fromstring(Candidate.getPathZ()[1:-1],sep=',')
    return np.stack([pathx,pathy,pathz], axis=1)

def getReflectionAnglesCandidate(Candidate):
    return np.fromstring(Candidate.getReflectionAngles()[1:-1],sep=',')


class ray_tracing:
    
    """ Numerical raytracing using Radiopropa. Currently this only works for icemodels that have only changing refractive index in z. """    
    
    solution_types = {1: 'direct',
                  2: 'refracted',
                  3: 'reflected'}
	
	
    def __init__(self, medium, attenuation_model="GL1", log_level=logging.WARNING,
                 n_frequencies_integration=100,
                 n_reflections=0, config=None, detector = None, shower_dir = None):
        
        """
        class initilization

        Parameters
        ----------
        x1: np.array of shape (1,3)
            start point of the ray
        x2: np.array of shape (1,3)
            stop point of the ray
        medium: medium class
            class describing the index-of-refraction profile
        attenuation_model: string
            signal attenuation model (so far only "SP1" is implemented)
        log_level: logging object
            specify the log level of the ray tracing class
            * logging.ERROR
            * logging.WARNING
            * logging.INFO
            * logging.DEBUG
            default is WARNING
        n_frequencies_integration: int
            the number of frequencies for which the frequency dependent attenuation
            length is being calculated. The attenuation length for all other frequencies
            is obtained via linear interpolation.
        shower_dir: np.array of shape (1,2)
                    zenith and azimuth of direction of shower in radians

        """
        self._airBoundary = radiopropa.Discontinuity(radiopropa.Plane(radiopropa.Vector3d(0,0,0), radiopropa.Vector3d(0,0,1)), 1.3, 1)
        self._medium = medium
        self._attenuation_model = attenuation_model
        self._results = None
        self._n_reflections = n_reflections
        self._shower_dir = shower_dir ## this is given so we cn limit the rays that are checked around the cherenkov angle 
        self._cut_viewing_angle = 20 #degrees wrt cherenkov angle
        self._iceModel = radiopropa.GreenlandIceModel() ## we need to figure out how to do this properly
        self._config = config
        self._n_frequencies_integration = n_frequencies_integration

        self._max_detector_frequency = None
        self._detector = detector
        if self._detector is not None:
            for station_id in self._detector.get_station_ids():
                sampling_frequency = self._detector.get_sampling_frequency(station_id, 0)
                if self._max_detector_frequency is None or sampling_frequency * .5 > self._max_detector_frequency:
                    self._max_detector_frequency = sampling_frequency * .5
        
       

   
        
    
    def set_start_and_end_point(self, x1, x2):
        x1 = np.array(x1, dtype =np.float)
        x2 = np.array(x2, dtype = np.float)
        self._x1 = x1
        self._x2 = x2


    def set_cut_viewing_angle(self,cut):
        self._cut_viewing_angle = cut
      

    
    
    def RadioPropa_raytracer(self, x1, x2 ):
        """
        uses RadioPropa to find the numerical ray tracing solutions for x1 x2 and returns the Candidates for all the possible solutions 
        """
        candidates = []
        ##define ice-air boundary
        
        ##define module list for simulation
        sim = radiopropa.ModuleList()
        sim.add(radiopropa.PropagationCK(self._iceModel, 1E-8, .001, 1.)) ## add propagation to module list
        sim.add(self._airBoundary)
        sim.add(radiopropa.MaximumTrajectoryLength(5000*radiopropa.meter))
        ## define observer (channel)
        obs = radiopropa.Observer()
        obs.setDeactivateOnDetection(True)
        channel = radiopropa.ObserverSurface(radiopropa.Sphere(radiopropa.Vector3d(self._x2[0], self._x2[1], self._x2[2]), 2 * radiopropa.meter)) ## when making the radius larger than 2 meters, somethimes three solution times are found
        obs.add(channel)
        sim.add(obs) ## add observer to module list

        phi_direct, theta = hp.cartesian_to_spherical(*(np.array(self._x2)-np.array(self._x1))) ## zenith and azimuth for the direct ray solution
        phi_direct = np.rad2deg(phi_direct) + 5 #the median solution is taken, meaning that we need to add some degrees in case the good solution is near phi_direct
        theta = np.rad2deg(theta)
         
        step = .05
        for phi in reversed(np.arange(0,phi_direct + step, step)): # in degrees
            if len(candidates) == 2: break
            x = hp.spherical_to_cartesian(self._shower_dir[0], self._shower_dir[1])
            y = hp.spherical_to_cartesian(np.deg2rad(phi), np.rad2deg(theta))
            delta = np.arccos(np.dot(x, y))

            cherenkov_angle = 56
            if 1:#(abs(np.rad2deg(delta) - cherenkov_angle) < self._cut_viewing_angle): #only include rays with angle wrt cherenkov angle smaller than 20 degrees ## if we add this, we need to make sure that the solution is not near the boundary, because we're taking the median solution now.
                source = radiopropa.Source()
                source.add(radiopropa.SourcePosition(radiopropa.Vector3d(self._x1[0], self._x1[1], self._x1[2])))
                x,y,z = hp.spherical_to_cartesian(phi * radiopropa.deg ,theta * radiopropa.deg)
                source.add(radiopropa.SourceDirection(radiopropa.Vector3d(x, y , z)))
                sim.setShowProgress(True)
                candidate = source.getCandidate()
                sim.run(candidate, True)
                trajectory_length = candidate.getTrajectoryLength()
                Candidate = candidate.get() #candidate is a pointer to the object Candidate
                detection = channel.checkDetection(Candidate) # check if the detection status of the channel
                if detection == 0: #check if the channel is reached
                    candidates.append(candidate)

        self._candidates = candidates 

    
    

    def find_solutions(self, reflection = 0):
        """
        find all solutions between x1 and x2
        """
        results = []
        
        launch_angles = []
        solution_types = []
        iSs = []
   
        
        self.RadioPropa_raytracer(self._x1, self._x2)
        num = len(self._candidates)
        candidates = np.copy(self._candidates)
        for iS, candidate in enumerate(self._candidates):
            solution_type = self.get_solution_type(iS)
            launch_vector = [self._candidates[iS].getLaunchVector().x, self._candidates[iS].getLaunchVector().y, self._candidates[iS].getLaunchVector().z] 
            launch_angles.append(hp.cartesian_to_spherical(launch_vector[0], launch_vector[1], launch_vector[2])[0])
        #    print("luanch angles {}, solutiontype {}".format( np.rad2deg(hp.cartesian_to_spherical(launch_vector[0], launch_vector[1], launch_vector[2])[0]), solution_type))
            solution_types.append(solution_type)
        mask = (np.array(solution_types) ==1 )
        index = 1
        self._candidates = []
        if mask.any():
            index = int(np.median(np.array(np.arange(0, num, 1))[mask]))
            self._candidates.append(candidates[index])
            results.append({'type':1, 'reflection':reflection})

        mask = (np.array(solution_types) ==2 )
        if mask.any():
            index = int(np.median(np.array(np.arange(0, num, 1))[mask]))
            self._candidates.append(candidates[index])
            results.append({'type':2, 'reflection':reflection})


        mask = (np.array(solution_types) ==3 )
        if mask.any():
            index = int(np.median(np.array(np.arange(0, num, 1))[mask]))
            self._candidates.append(candidates[index])
            results.append({'type':3, 'reflection':reflection})

                    
        
        
        self._results = results

    def has_solution(self):
        """
        checks if ray tracing solution exists
        """
        return len(self._results) > 0

    def get_number_of_solutions(self):
        """
        returns the number of solutions
        """
        return len(self._results)

    def get_results(self):
        """

        """
        return self._results
    
    
    def get_path(self, iS, n_points=1000):
        """
        helper function that returns the 3D ray tracing path of solution iS

        Parameters
        ----------
        iS: int
            ray tracing solution
        n_points: int
            number of points of path
        """
        self._path = getPathCandidate(self._candidates[iS].get())
        
        return self._path

    def get_solution_type(self, iS):
        """ returns the type of the solution

        Parameters
        ----------
        iS: int
            choose for which solution to compute the launch vector, counting
            starts at zero

        Returns
        -------
        solution_type: int
            * 1: 'direct'
            * 2: 'refracted'
            * 3: 'reflected
        """
        self._path = getPathCandidate(self._candidates[iS].get())
        pathz = self._path[:, 2]
        if len(getReflectionAnglesCandidate(self._candidates[iS].get())) != 0:
            solution_type = 3
        
        elif(pathz[-1] < max(pathz)):
            solution_type = 2
        else:
            solution_type = 1        
        
        return solution_type
        
    
        

    def get_launch_vector(self, iS):
        """
        calculates the launch vector (in 3D) of solution iS

        Parameters
        ----------
        iS: int
            choose for which solution to compute the launch vector, counting
            starts at zero

        Returns
        -------
        launch_vector: 3dim np.array
            the launch vector
            
        """
        launch_vector = [self._candidates[iS].getLaunchVector().x, self._candidates[iS].getLaunchVector().y, self._candidates[iS].getLaunchVector().z] 
        return np.array(launch_vector)

    def get_receive_vector(self, iS):
        """
        calculates the receive vector (in 3D) of solution iS

        Parameters
        ----------
        iS: int
            choose for which solution to compute the launch vector, counting
            starts at zero

        Returns
        -------
        receive_vector: 3dim np.array
            the receive vector
            
        """
        receive_vector = [self._candidates[iS].getReceiveVector().x, self._candidates[iS].getReceiveVector().y, self._candidates[iS].getReceiveVector().z]
        return np.array(receive_vector)

    def get_reflection_angle(self, iS):
        """
        calculates the angle of reflection at the surface (in case of a reflected ray)

        Parameters
        ----------
        iS: int
            choose for which solution to compute the launch vector, counting
            starts at zero

        Returns
        -------
        reflection_angle: float or None
            the reflection angle (for reflected rays) or None for direct and refracted rays
        """
        reflection_angles = getReflectionAnglesCandidate(self._candidates[iS])
        return reflection_angles

    def get_path_length(self, iS, analytic=True):
        """
        calculates the path length of solution iS

        Parameters
        ----------
        iS: int
            choose for which solution to compute the launch vector, counting
            starts at zero

        analytic: bool
            If True the analytic solution is used. If False, a numerical integration is used. (default: True)

        Returns
        -------
        distance: float
            distance from x1 to x2 along the ray path
        """
        path_length = self._candidates[iS].getTrajectoryLength()
        return path_length

    def get_travel_time(self, iS, analytic=True):
        """
        calculates the travel time of solution iS

        Parameters
        ----------
        iS: int
            choose for which solution to compute the launch vector, counting
            starts at zero

        analytic: bool
            If True the analytic solution is used. If False, a numerical integration is used. (default: True)

        Returns
        -------
        time: float
            travel time
        """
        travel_time = self._candidates[iS].getPropagationTime()
        return travel_time
    
    
    def get_frequencies_for_attenuation(self, frequency, max_detector_freq):
        mask = frequency > 0
        nfreqs = min(self._n_frequencies_integration, np.sum(mask))
        freq = np.linspace(frequency[mask].min(), frequency[mask].max(), nfreqs)
        if(nfreqs < np.sum(mask) and max_detector_freq is not None):
            mask2 = frequency <= max_detector_freq
            nfreqs2 = min(self._n_frequencies_integration, np.sum(mask2 & mask))
            freqs = np.linspace(frequency[mask2 & mask].min(), frequency[mask2 & mask].max(), nfreqs2)
            if(np.sum(~mask2)>1):
                freqs = np.append(freqs, np.linspace(frequency[~mask2].min(), frequency[~mask2].max(), nfreqs // 2))
            return freqs
        
        
    def get_attenuation(self, iS, frequency, max_detector_freq=None):
        """
        calculates the signal attenuation due to attenuation in the medium (ice)

        Parameters
        ----------
        iS: int
            choose for which solution to compute the launch vector, counting
            starts at zero

        frequency: array of floats
            the frequencies for which the attenuation is calculated

        max_detector_freq: float or None
            the maximum frequency of the final detector sampling
            (the simulation is internally run with a higher sampling rate, but the relevant part of the attenuation length
            calculation is the frequency interval visible by the detector, hence a finer calculation is more important)

        Returns
        -------
        attenuation: array of floats
            the fraction of the signal that reaches the observer
            (only ice attenuation, the 1/R signal falloff not considered here)
        
        """
        self.get_path(iS)
        
        mask = frequency > 0
        freqs = self.get_frequencies_for_attenuation(frequency, self._max_detector_frequency)
        #freqs = frequency
        
        integral = np.zeros(len(freqs))
        def dt(depth, freqs):
            ds = np.sqrt((self._path[:, 0][depth] - self._path[:, 0][depth+1])**2 + (self._path[:, 1][depth] - self._path[:, 1][depth+1])**2 + (self._path[:, 2][depth] - self._path[:, 2][depth+1])**2) # get step size
            return ds / attenuation_util.get_attenuation_length(self._path[2][depth], freqs, self._attenuation_model)
        for z_position in range(len(self._path[2]-1)):
            integral += dt(z_position, freqs)
        att_func = interpolate.interp1d(freqs, integral)
        tmp = att_func(frequency[mask])
        attenuation = np.ones_like(frequency)
        attenuation[mask] = np.exp(-1 * tmp)
        return attenuation

    def apply_propagation_effects(self, efield, i_solution):
        spec = efield.get_frequency_spectrum()
        ## aply attenuation 
        if self._config is None:
            apply_attenuation = True
        else: 
            apply_attenuation = self._config['propagation']['attenuate_ice']
        if apply_attenuation:
            if self._max_detector_frequency is None:
                max_freq = np.max(efield.get_frequencies())
            else:
                max_freq = self._max_detector_frequency
            attenuation = self.get_attenuation(i_solution, efield.get_frequencies(), max_freq)
            spec *= attenuation
        ## apply reflections
        i_reflections = self.get_results()[i_solution]['reflection']
        zenith_reflections = np.atleast_1d(self.get_reflection_angle(i_solution))
        for zenith_reflection in zenith_reflections:
            if (zenith_reflection is None):
                continue
            r_theta = NuRadioReco.utilities.geometryUtilities.get_fresnel_r_p(
                zenith_reflection, n_2=1., n_1=self._medium.get_index_of_refraction([self._x2[0], self._x2[1], -1 * units.cm]))
            r_phi = NuRadioReco.utilities.geometryUtilities.get_fresnel_r_s(
                zenith_reflection, n_2=1., n_1=self._medium.get_index_of_refraction([self._x2[0], self._x2[1], -1 * units.cm]))
            efield[efp.reflection_coefficient_theta] = r_theta
            efield[efp.reflection_coefficient_phi] = r_phi

            spec[1] *= r_theta
            spec[2] *= r_phi
           
            
            
        if (i_reflections > 0):  # take into account possible bottom reflections
            # each reflection lowers the amplitude by the reflection coefficient and introduces a phase shift
            reflection_coefficient = self._medium.reflection_coefficient ** i_reflections
            phase_shift = (i_reflections * self._medium.reflection_phase_shift) % (2 * np.pi)
            # we assume that both efield components are equally affected
            spec[1] *= reflection_coefficient * np.exp(1j * phase_shift)
            spec[2] *= reflection_coefficient * np.exp(1j * phase_shift)
            
            
        ## apply focussing effect
        if self._config['propagation']['focusing']:
            dZRec = -0.01 * units.m
            focusing = self.get_focusing(i_solution, dZRec, float(self._config['propagation']['focusing_limit']))
            spec[1:] *= focusing

        efield.set_frequency_spectrum(spec, efield.get_sampling_rate())
        return efield
    

    def create_output_data_structure(self, dictionary, n_showers, n_antennas):
        nS = self.get_number_of_raytracing_solutions()
        dictionary['ray_tracing_solution_type'] = np.ones((n_showers, n_antennas, nS), dtype=np.int) * -1


    def write_raytracing_output(self, dictionary, i_shower, channel_id, i_solution):
        dictionary['ray_tracing_solution_type'][i_shower, channel_id, i_solution] = self.get_solution_type(i_solution)


    def get_number_of_raytracing_solutions(self):
        return 2 + 4 * self._n_reflections 
