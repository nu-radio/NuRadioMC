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
import scipy.constants 

"""
STILL TO DO:
- compare timing, distances, receive and launch angles wtr analytic raytracer
- compare attenuation wtr analytic raytracer
- proper implementatin of the icemodel
- Add warnings when the icemodel is not implemented in radiopropa
- Timing reduction; everything takes too long now
- compare waveforms wtr analytic raytracer
"""


class ray_tracing:

    """ Numerical raytracing using Radiopropa. Currently this only works for icemodels that have only changing refractive index in z. """

    solution_types = {1: 'direct',
                  2: 'refracted',
                  3: 'reflected'}


    def __init__(self, medium, attenuation_model="GL1", log_level=logging.WARNING,
                 n_frequencies_integration=100,
                 n_reflections=0, config=None, detector = None):

        """
        class initilization

        Parameters
        ----------
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
        shower_axis: np.array of shape (3,)
                    x, y and z of direction of shower in radians

        """
        self._airBoundary = radiopropa.Discontinuity(radiopropa.Plane(radiopropa.Vector3d(0,0,0), radiopropa.Vector3d(0,0,1)), 1.3, 1)
        self._medium = medium
        self._attenuation_model = attenuation_model
        self._results = None
        self._n_reflections = n_reflections
        self._cut_viewing_angle = 40 * units.degree #degrees wrt cherenkov angle
        self._max_traj_length = 5000 * units.meter
        self._iceModel = radiopropa.GreenlandIceModel() ## we need to figure out how to do this properly
        self._config = config
        self._n_frequencies_integration = n_frequencies_integration

        self._sphere_sizes = np.array([25.,2.,.5]) * units.meter ## iteration from big to small observer around channel
        self._step_sizes = np.array([.5,.05,.0125]) * units.degree ## step for theta corresponding to the sphere size, should have same lenght as _sphere_sizes
        self._x1 = None
        self._x2 = None
        self._shower_axis = None ## this is given so we cn limit the rays that are checked around the cherenkov angle
        self._max_detector_frequency = None
        self._detector = detector
        if self._detector is not None:
            for station_id in self._detector.get_station_ids():
                sampling_frequency = self._detector.get_sampling_frequency(station_id, 0)
                if self._max_detector_frequency is None or sampling_frequency * .5 > self._max_detector_frequency:
                    self._max_detector_frequency = sampling_frequency * .5





    def get_start_and_end_point(self ):
        return self._x1, self._x2

    def set_start_and_end_point(self, x1=None, x2=None):
        """
        Parameters
        ----------
        x1: np.array of shape (3,), unit is meter
            start point of the ray
        x2: np.array of shape (3,), unit is meter
            stop point of the ray
        """ 
        x1 = np.array(x1, dtype =np.float)
        self._x1 = x1 * units.meter
        x2 = np.array(x2, dtype = np.float)
        self._x2 = x2 * units.meter

    def get_shower_axis(self):
        return self._shower_axis

    def set_shower_axis(self,shower_axis=None):
        """
        Set the the shower axis. This is oposite to the neutrino arrival direction

        Parameters
        ----------
        shower_axis: np.array of shape (3,), unit not relevant (preferably meter)
            the direction of where the shower is moving towards to in cartesian coordinates
        """ 
        self._shower_axis = shower_axis


    def set_cut_viewing_angle(self,cut):
        """
        Parameters
        ----------
        cut: float, unit is degree
             range around the cherenkov angle
             rays with a viewing angle out of this range will be to dim and won't be seen --> limiting computing time
        """
        self._cut_viewing_angle = cut * units.degree

    def set_maximum_trajectory_length(self,max_traj_length):
        """
        Parameters
        ----------
        max_traj_length: float, unit is meter
                         maxmimal length to trace a ray. tracing aborted when reached
        """
        self._max_traj_length = max_traj_length * units.meter

    def get_cut_viewing_angle(self):
        """
        Returns
        ----------
        cut_viewing_angle: float
                           range around the cherenkov angle
                           rays with a viewing angle out of this range will be to dim and won't be seen --> limiting computing time
        """
        return self._cut_viewing_angle

    def get_maximum_trajectory_length(self,max_traj_length):
        """
        Returns
        ----------
        max_traj_length: float
                         maxmimal length to trace a ray, tracing aborted when reached.
        """
        return self._max_traj_length

    def RadioPropa_raytracer(self ):
        """
        uses RadioPropa to find the numerical ray tracing solutions for x1 x2 and returns the Candidates for all the possible solutions
        """
        try:
            x1 = self._x1  * (radiopropa.meter/units.meter)
            x2 = self._x2  * (radiopropa.meter/units.meter)
        except TypeError: 
            print('NoneType: start or endpoint not initialized')
            TypeError

        theta_direct, phi = hp.cartesian_to_spherical(*(np.array(self._x2)-np.array(self._x1))) *units.radian ## zenith and azimuth for the direct linear ray solution (radians)
        theta_direct += 5*units.degree ##the median solution is taken, meaning that we need to add some degrees in case the good solution is near theta_direct

        launch_lower = [0]
        launch_upper = [theta_direct] ##below theta_direct no solutions are possible without upward reflections
        previous_candidates = None
        for s,sphere_size in enumerate(self._sphere_sizes):
            current_candidates = []

            ##define module list for simulation
            sim = radiopropa.ModuleList()
            sim.add(radiopropa.PropagationCK(self._iceModel, 1E-8, .001, 1.)) ## add propagation to module list
            sim.add(self._airBoundary)
            sim.add(radiopropa.MaximumTrajectoryLength(self._max_traj_length*(radiopropa.meter/units.meter)))

            ## define observer for detection (channel)            
            obs = radiopropa.Observer()
            obs.setDeactivateOnDetection(True)
            channel = radiopropa.ObserverSurface(radiopropa.Sphere(radiopropa.Vector3d(*x2), sphere_size*(radiopropa.meter/units.meter))) ## when making the radius larger than 2 meters, somethimes three solution times are found
            obs.add(channel)
            sim.add(obs)

            ## define observer for stopping simulation (boundaries)
            obs2 = radiopropa.Observer()
            obs2.setDeactivateOnDetection(True)
            v = (x2-x1)
            v[2]=0
            v = (v/np.linalg.norm(v)) * 2*sphere_size*(radiopropa.meter/units.meter)
            boundary_behind_channel = radiopropa.ObserverSurface(radiopropa.Plane(radiopropa.Vector3d(*(x2+v)), radiopropa.Vector3d(*v)))
            obs2.add(boundary_behind_channel)
            boundary_above_surface = radiopropa.ObserverSurface(radiopropa.Plane(radiopropa.Vector3d(0,0,1*radiopropa.meter), radiopropa.Vector3d(0,0,1)))
            obs2.add(boundary_above_surface)
            sim.add(obs2)

            #loop over previous candidates to find the upper and lower theta of each bundle of rays
            #uses step, but because step is initialized after this loop this ios the previous step size as intented
            if previous_candidates == None:
                pass
            elif len(previous_candidates)>0:
                launch_lower.clear()
                launch_upper.clear()
                for iPC,PC in enumerate(previous_candidates):
                    launch_theta = PC.getLaunchVector().getTheta()
                    if iPC == (len(previous_candidates)-1) or iPC == 0:
                        if iPC == 0: 
                            launch_lower.append(launch_theta-step)
                        if iPC == (len(previous_candidates)-1): 
                            launch_upper.append(launch_theta+step)
                    elif abs(launch_theta - launch_theta_prev)>1.1*step: ##take 1.1 times the step to be sure the next ray is not in the bundle of the previous one
                        launch_upper.append(launch_theta_prev+step)
                        launch_lower.append(launch_theta-step)
                    else:
                        pass
                    launch_theta_prev = launch_theta
            else:
                #if previous_candidates is empthy, no solutions where found and the tracer is terminated
                break

            
            #create total scanning range from the upper and lower thetas of the bundles
            step = self._step_sizes[s]
            theta_scanning_range = np.array([])
            for iL in range(len(launch_lower)):
                new_scanning_range = np.arange(launch_lower[iL],launch_upper[iL]+step,step)
                theta_scanning_range = np.concatenate((theta_scanning_range,new_scanning_range))

            cherenkov_angle = 56 *units.degree

            for theta in theta_scanning_range: 
                ray = hp.spherical_to_cartesian(theta/units.radian,phi/units.radian)
                viewing = np.arccos(np.dot(self._shower_axis, ray)) * units.radian


                delta = viewing - cherenkov_angle
                #only include rays with angle wrt cherenkov angle smaller than 20 degrees 
                if (abs(delta) < self._cut_viewing_angle): ## if we add this, we need to make sure that the solution is not near the boundary, because we're taking the median solution now.
                    source = radiopropa.Source()
                    source.add(radiopropa.SourcePosition(radiopropa.Vector3d(*x1)))
                    source.add(radiopropa.SourceDirection(radiopropa.Vector3d(*ray)))
                    sim.setShowProgress(True)
                    candidate = source.getCandidate()
                    sim.run(candidate, True)
                    Candidate = candidate.get() #candidate is a pointer to the object Candidate
                    detection = channel.checkDetection(Candidate) #the detection status of the channel
                    if detection == 0: #check if the channel is reached
                        current_candidates.append(candidate)

            previous_candidates = current_candidates

        self._candidates = current_candidates




    def find_solutions(self, reflection = 0):
        """
        find all solutions
        """
        results = []

        self.RadioPropa_raytracer()
        num = len(self._candidates)

        launch_zeniths = []
        receive_zeniths = []
        solution_types = []
        ray_endpoints = []
        iSs = np.array(np.arange(0, num, 1))

        for iS, candidate in enumerate(self._candidates):
            solution_type = self.get_solution_type(iS)
            launch_vector = self.get_launch_vector(iS)
            receive_vector = self.get_receive_vector(iS)
            ray_endpoint = self.get_path(iS)[-1]
            launch_zeniths.append(hp.cartesian_to_spherical(launch_vector[0], launch_vector[1], launch_vector[2])[0])
            receive_zeniths.append(hp.cartesian_to_spherical(receive_vector[0], receive_vector[1], receive_vector[2])[0])
            solution_types.append(solution_type)
            ray_endpoints.append(ray_endpoint)
        
        candidates = np.copy(self._candidates)
        self._candidates = []
        channel_pos = self._x2

        mask = {i:(np.array(solution_types) == i ) for i in range(1,4)}       
        for i in range(1,4):
            if mask[i].any():
                '''
                index = int(np.median(np.array(np.arange(0, num, 1))[mask]))
                self._candidates.append(candidates[index])
                results.append({'type':1, 'reflection':reflection})
                '''
                final_candidate = None
                delta_min = np.deg2rad(90)
                for iS in iSs[mask[i]]: #index o candidates with solution type i
                    vector = ray_endpoints[iS] - self._x2 #position of the receive vector on the sphere around the channel
                    vector_zenith = hp.cartesian_to_spherical(vector[0],vector[1],vector[2])[0]
                    delta = abs(vector_zenith-receive_zeniths[iS])
                    if delta < delta_min:
                        final_candidate = candidates[iS]
                        delta_min = delta
                self._candidates.append(final_candidate)
                results.append({'type':int(i), 'reflection':reflection})
                '''
                delta_min = self._sphere_size
                for candidate in candidates[mask[i]]:
                    ray_endpoint = candidate.get().getEndPosition()*(units.meter/radiopropa.meter) #endpoint on sphere
                    ray_endpoint -=(candidate.get().getReceiveVector().getUnitVector()*(units.meter/radiopropa.meter) *self._sphere_size) #endpoint in sphere after extrapolation
                    ray_endpoint = np.array([ray_endpoint.getX(),ray_endpoint.getY(),ray_endpoint.getZ()]) #transform in numpy array
                    delta = np.linalg.norm(ray_endpoint-channel_pos)
                    if delta < delta_min: 
                        final_candidate = candidate
                        delta_min = min(delta_min,delta)
                self._candidates.append(candidate)
                results.append({'type':i, 'reflection':reflection})
                '''

        self._results = results

    def has_solution(self):
        """
        checks if ray tracing solution exists
        """
        return len(self._results) > 0

    def get_number_of_solutions(self):
        """
        Returns
        -------
        the number of solutions: int
        """
        return len(self._results)

    def get_results(self):
        """
        Returns
        -------
        the results of the tracing: dictionary
        """
        return self._results


    def get_path(self, iS, n_points=1000):#n_points still need to be fixed
        """
        helper function that returns the 3D ray tracing path of solution iS

        Parameters
        ----------
        iS: int
            ray tracing solution
        n_points: int
            number of points of path

        Returns
        -------
        path: 2dim np.array of shape (n_points,3)
              x, y, z coordinates along second axis
        """
        Candidate = self._candidates[iS].get()
        pathx = np.fromstring(Candidate.getPathX()[1:-1],sep=',')*(units.meter/radiopropa.meter)
        pathy = np.fromstring(Candidate.getPathY()[1:-1],sep=',')*(units.meter/radiopropa.meter)
        pathz = np.fromstring(Candidate.getPathZ()[1:-1],sep=',')*(units.meter/radiopropa.meter)

        return np.stack([pathx,pathy,pathz], axis=1)

    def get_solution_type(self, iS):
        """ returns the type of the solution

        Parameters
        ----------
        iS: int
            choose for which solution to compute the solution type, 
            counting starts at zero

        Returns
        -------
        solution_type: int
            * 1: 'direct'
            * 2: 'refracted'
            * 3: 'reflected
        """

        pathz = self.get_path(iS)[:, 2]
        if self.get_reflection_angle(iS) != None:
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
            choose for which solution to compute the launch vector, 
            counting starts at zero

        Returns
        -------
        launch_vector: np.array of shape (3,)
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
            choose for which solution to compute the receive vector, 
            counting starts at zero

        Returns
        -------
        receive_vector: np.array of shape (3,)
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
            choose for which solution to compute the reflection angle, 
            counting starts at zero

        Returns
        -------
        reflection_angle: 1dim np.array
            the reflection angle (for reflected rays) or None for direct and refracted rays
        """
        Candidate = self._candidates[iS].get()
        reflection_angles = np.fromstring(Candidate.getReflectionAngles()[1:-1],sep=',') *(units.degree/radiopropa.deg)
        if len(reflection_angles)==0:
            return None
        else:
            return reflection_angles[0]

    def get_correction_path_length(self, iS):
        """
        calculates the correction of the path length of solution iS 
        due to the sphere around the channel

        Parameters
        ----------
        iS: int
            choose for which solution to compute the path length correction, 
            counting starts at zero

        Returns
        -------
        distance: float
            distance that should be added to the path length
        """
        end_of_path = self.get_path(iS)[-1] #position of the receive vector on the sphere around the channel in detector coordinates
        receive_vector = self.get_receive_vector(iS)
        
        vector = end_of_path - self._x2 #position of the receive vector on the sphere around the channel
        vector_zen,vector_az = hp.cartesian_to_spherical(vector[0],vector[1],vector[2])
        receive_zen,receive_az = hp.cartesian_to_spherical(receive_vector[0],receive_vector[1],receive_vector[2])

        path_correction_arrival_direction = abs(np.cos(receive_zen-vector_zen))*self._sphere_sizes[-1]
        
        if abs(receive_az-vector_az) > np.deg2rad(90): 
            path_correction_overshoot = np.linalg.norm(vector[0:2])*abs(np.cos(receive_az-vector_az))
        else: 
            path_correction_overshoot = 0
        
        return path_correction_arrival_direction - path_correction_overshoot

    def get_correction_travel_time(self, iS):
        """
        calculates the correction of the travel time of solution iS 
        due to the sphere around the channel

        Parameters
        ----------
        iS: int
            choose for which solution to compute the travel time correction, 
            counting starts at zero

        Returns
        -------
        distance: float
            distance that should be added to the path length
        """
        refrac_index = self._medium.get_index_of_refraction(self._x2)
        return self.get_correction_path_length(iS) / ((scipy.constants.c*units.meter/units.second)/refrac_index)


    def get_path_length(self, iS):
        """
        calculates the path length of solution iS

        Parameters
        ----------
        iS: int
            choose for which solution to compute the path length, 
            counting starts at zero

        analytic: bool
            If True the analytic solution is used. If False, a numerical integration is used. (default: True)

        Returns
        -------
        distance: float
            distance from x1 to x2 along the ray path
        """
        path_length = self._candidates[iS].getTrajectoryLength() *(units.meter/radiopropa.meter)
        return path_length + self.get_correction_path_length(iS)

    def get_travel_time(self, iS):
        """
        calculates the travel time of solution iS

        Parameters
        ----------
        iS: int
            choose for which solution to compute the travel time, 
            counting starts at zero

        analytic: bool
            If True the analytic solution is used. If False, a numerical integration is used. (default: True)

        Returns
        -------
        time: float
            travel time
        """

        travel_time = self._candidates[iS].getPropagationTime() *(units.second/radiopropa.second)
        return travel_time + self.get_correction_travel_time(iS)


    def get_frequencies_for_attenuation(self, frequency, max_detector_freq):
        mask = frequency > 0
        nfreqs = min(self._n_frequencies_integration, np.sum(mask))
        freqs = np.linspace(frequency[mask].min(), frequency[mask].max(), nfreqs)
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
            choose for which solution to compute the attenuation, 
            counting starts at zero

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
        path = self.get_path(iS)

        mask = frequency > 0
        freqs = self.get_frequencies_for_attenuation(frequency, self._max_detector_frequency)
        integral = np.zeros(len(freqs))
        def dt(depth, freqs):
            ds = np.sqrt((path[:, 0][depth] - path[:, 0][depth+1])**2 + (path[:, 1][depth] - path[:, 1][depth+1])**2 + (path[:, 2][depth] - path[:, 2][depth+1])**2) # get step size
            return ds / attenuation_util.get_attenuation_length(path[:, 2][depth], freqs, self._attenuation_model)
        for z_position in range(len(path[:, 2]) - 1):
            integral += dt(z_position, freqs)
        att_func = interpolate.interp1d(freqs, integral)
        tmp = att_func(frequency[mask])
        attenuation = np.ones_like(frequency)
        tmp = np.exp(-1 * tmp)
        attenuation[mask] = tmp
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
