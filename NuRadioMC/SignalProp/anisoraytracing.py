from __future__ import absolute_import, division, print_function
import logging
logging.basicConfig()

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, curve_fit, root, root_scalar, OptimizeResult, least_squares
from scipy.constants import speed_of_light
from scipy.linalg import null_space
from scipy.interpolate import interp1d
from multiprocessing import Process, Queue

from NuRadioMC.SignalProp import analyticraytracing
from NuRadioMC.utilities import medium
from radiotools import helper as hp
from NuRadioMC.utilities import attenuation as attenuation_util
from NuRadioReco.utilities import units
import NuRadioReco


solution_types = {1: 'direct',
                  2: 'refracted',
                  3: 'reflected'}

reflection_case = {1: 'upwards launch vector',
                  2: 'downward launch vector'}


class aniso_ray_tracing:
    """
    Raytracer class for NuRMC. Most of the implementations are written in the ray 
    class, and this class acts as a wrapper.
    """
    
    solution_types = {1: 'direct',
                  2: 'refracted',
                  3: 'reflected'}

    def __init__(self, medium, attenuation_model="SP1", log_level=logging.WARNING,
                 n_frequencies_integration=6, config=None, n_reflections=0, detector=None,
                 guess_medium=medium.get_ice_model('ARAsim_southpole')):
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

        """
        self.__config = config
        self.__anisorays = rays(medium, guess_medium=guess_medium, par=self.__config['propagation']['par'])
        
        self.__attenuation_model = attenuation_model
        self.__n_frequencies_integration = n_frequencies_integration
        self.__max_detector_frequency = None
        self.__detector = detector
        if self.__detector is not None:
            for station_id in self.__detector.get_station_ids():
                sampling_frequency = self.__detector.get_sampling_frequency(station_id, 0)
                if self.__max_detector_frequency is None or sampling_frequency * .5 > self.__max_detector_frequency:
                    self.__max_detector_frequency = sampling_frequency * .5

    def set_start_and_end_point(self, x1, x2):
        """
        Set the start and end points between which raytracing solutions shall be found
        It is recommended to also reset the solutions from any previous raytracing to avoid
        confusing them with the current solution

        Parameters:
        --------------
        x1: 3D array
            Start point of the ray
        x2: 3D array
            End point of the ray
        """
        self.__anisorays.set_start_and_end_point(x1, x2)
        
    def _index_to_tuple(self, a):
        """function to change NRMC indexing into indexing used by rays class"""
        if(a == 0):
            return [0,0]
        elif(a==1):
            return [0,1]
        elif(a==2):
            return [1,0]
        elif(a==3):
            return [1,1]
        else:
            return None
    
    def _tuple_to_index(self, a):
        """function to change rays class indexing to NRMC indexing"""
        if(a==[0,0]):
            return 0
        elif(a==[0,1]):
            return 1
        elif(a==[1,0]):
            return 2
        elif(a==[1,1]):
            return 3
        else:
            return None

    def use_optional_function(self, function_name, *args, **kwargs):
        """
        Use optional function which may be different for each ray tracer.
        If the name of the function is not present for the ray tracer this function does nothing.

        Parameters
        ----------
        function_name: string
                       name of the function to use
        *args: type of the argument required by function
               all the neseccary arguments for the function separated by a comma
        **kwargs: type of keyword argument of function
                  all all the neseccary keyword arguments for the function in the
                  form of key=argument and separated by a comma

        Example
        -------
        use_optional_function('set_shower_axis',np.array([0,0,1]))
        use_optional_function('set_iterative_sphere_sizes',sphere_sizes=np.aray([3,1,.5]))
        """
        if not hasattr(self,function_name):
            pass
        else:
            getattr(self,function_name)(*args,**kwargs)

    def find_solutions(self):
        """
        find all solutions between x1 and x2
        """
        self.__anisorays.get_rays()
        
        results = []
        for i in range(4):
            results.append(self.get_raytracing_output(i))
        
        self.__results = results

    def has_solution(self):
        """
        checks if ray tracing solution exists
        """
        return len(self.__results) > 0

    def get_number_of_solutions(self):
        """
        returns the number of solutions
        """
        return len(self.__results)

    def get_results(self):
        """

        """

        return self.__results

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
        return self.__anisorays.get_solution_type(*self._index_to_tuple(iS))

    def get_path(self, iS):
        """
        helper function that returns the 3D ray tracing path of solution iS

        Parameters
        ----------
        iS: int
            ray tracing solution
        """
        return self.__anisorays.get_path(*self._index_to_tuple(iS))


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
        return self.__anisorays.get_launch_vector(*self._index_to_tuple(iS))

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
        return self.__anisorays.get_receive_vector(*self._index_to_tuple(iS))

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
        return self.__anisorays.get_reflection_angle(*self._index_to_tuple(iS))

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
        return self.__anisorays.get_path_length(*self._index_to_tuple(iS))

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
        return self.__anisorays.get_travel_time(*self._index_to_tuple(iS))

    def get_frequencies_for_attenuation(self, frequency, max_detector_freq):
        """
        helper function to get the frequencies for applying attenuation
        Parameters
        ----------
        frequency: array of float of dim (n,)
            frequencies of the signal
        max_detector_freq: float or None
            the maximum frequency of the final detector sampling
            (the simulation is internally run with a higher sampling rate, but the relevant part of the attenuation length
            calculation is the frequency interval visible by the detector, hence a finer calculation is more important)
        Returns
        -------
        freqs: array of float of dim (m,)
             the frequencies for which the attenuation is calculated
        """
        mask = frequency > 0
        nfreqs = min(self.__n_frequencies_integration, np.sum(mask))
        freqs = np.linspace(frequency[mask].min(), frequency[mask].max(), nfreqs)
        if(nfreqs < np.sum(mask) and max_detector_freq is not None):
            mask2 = frequency <= max_detector_freq
            nfreqs2 = min(self.__n_frequencies_integration, np.sum(mask2 & mask))
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
        
        # bob oeyen's attenuation code

        n = self.get_number_of_solutions()
        if(iS >= n):
            self.__logger.error("solution number {:d} requested but only {:d} solutions exist".format(iS + 1, n))
            raise IndexError

        path = self.get_path(iS)

        mask = frequency > 0
        freqs = self.get_frequencies_for_attenuation(frequency, self.__max_detector_frequency)
        integral = np.zeros(len(freqs))
        
        def dt(depth, freqs):
            ds = np.sqrt((path.y[:, 0][depth] - path.y[:, 0][depth+1])**2 + (path.y[:, 1][depth] - path.y[:, 1][depth+1])**2 + (path.y[:, 2][depth] - path.y[:, 2][depth+1])**2) # get step size
            return ds / attenuation_util.get_attenuation_length(path.y[:, 2][depth], freqs, self.__attenuation_model)
        
        for z_position in range(len(path.y[:, 2]) - 1):
            integral += dt(z_position, freqs)
        
        att_func = interp1d(freqs, integral)
        tmp = att_func(frequency[mask])
        attenuation = np.ones_like(frequency)
        tmp = np.exp(-1 * tmp)
        attenuation[mask] = tmp
        
        return attenuation

    def apply_propagation_effects(self, efield, i_solution):
        """
        Apply propagation effects to the electric field
        Note that the 1/r weakening of the electric field is already accounted for in the signal generation

        Parameters:
        ----------------
        efield: ElectricField object
            The electric field that the effects should be applied to
        i_solution: int
            Index of the raytracing solution the propagation effects should be based on

        Returns
        -------------
        efield: ElectricField object
            The modified ElectricField object
        """
        
        #apply attenuation using isotropic approximation
        spec = efield.get_frequency_spectrum()
        if self.__config is None:
            apply_attenuation = True
        else:
            apply_attenuation = self.__config['propagation']['attenuate_ice']
        if apply_attenuation:
            if self.__max_detector_frequency is None:
                max_freq = np.max(efield.get_frequencies())
            else:
                max_freq = self.__max_detector_frequency
            attenuation = self.get_attenuation(i_solution, efield.get_frequencies(), max_freq)
            spec *= attenuation

        zenith_reflections = np.atleast_1d(self.get_reflection_angle(i_solution))  # lets handle the general case of multiple reflections off the surface (possible if also a reflective bottom layer exists)
        for zenith_reflection in zenith_reflections:  # loop through all possible reflections
            if (zenith_reflection is None):  # skip all ray segments where not reflection at surface happens
                continue
            r_theta = NuRadioReco.utilities.geometryUtilities.get_fresnel_r_p(
                zenith_reflection, n_2=1., n_1=self.__medium.get_index_of_refraction([self.__X2[0], self.__X2[1], -1 * units.cm]))
            r_phi = NuRadioReco.utilities.geometryUtilities.get_fresnel_r_s(
                zenith_reflection, n_2=1., n_1=self.__medium.get_index_of_refraction([self.__X2[0], self.__X2[1], -1 * units.cm]))
            efield[efp.reflection_coefficient_theta] = r_theta
            efield[efp.reflection_coefficient_phi] = r_phi

            spec[1] *= r_theta
            spec[2] *= r_phi
            self.__logger.debug(
                "ray hits the surface at an angle {:.2f}deg -> reflection coefficient is r_theta = {:.2f}, r_phi = {:.2f}".format(
                    zenith_reflection / units.deg,
                    r_theta, r_phi))
        
        # apply the focusing effect
        if self.__config['propagation']['focusing']:
            focusing = self.get_focusing(i_solution, limit=float(self.__config['propagation']['focusing_limit']))
            spec[1:] *= focusing
        
        #apply anisotropic propagation effects (i.e. eigenpols)
        epol_cartesian = self.__anisorays.get_final_E_pol(*self._index_to_tuple(i_solution))
        theta, phi = hp.cartesian_to_spherical(*epol_cartesian)
        rho = 1 #epol vectors are normalized
        epol_spherical = np.array([rho, theta, phi]/np.linalg.norm([rho, theta, phi]))

        trace = efield.get_trace()
        for i in range(spec.shape[0]):
            trace[i, :] *= epol_spherical[i]         
        
        efield.set_frequency_spectrum(spec, efield.get_sampling_rate())
        efield.set_trace(trace, efield.get_sampling_rate())
        return efield

    def get_output_parameters(self):
        """
        Returns a list with information about parameters to include in the output data structure that are specific
        to this raytracer

        ! be sure that the first entry is specific to your raytracer !

        Returns:
        -----------------
        list with entries of form [{'name': str, 'ndim': int}]
            ! be sure that the first entry is specific to your raytracer !
            'name': Name of the new parameter to include in the data structure
            'ndim': Dimension of the data structure for the parameter
        """
        return [
            {'name': 'speed_type','ndim': 1},
            {'name': 'launch_vector', 'ndim': 3},
            {'name': 'focusing_factor', 'ndim': 1},
            {'name': 'ray_tracing_reflection', 'ndim': 1},
            {'name': 'ray_tracing_reflection_case', 'ndim': 1},
            {'name': 'ray_tracing_solution_type', 'ndim': 1}
        ]

    def get_raytracing_output(self, i_solution):
        """
        Write parameters that are specific to this raytracer into the output data.
#
        Parameters:
        ---------------
        i_solution: int
            The index of the raytracing solution

        Returns:
        ---------------
        dictionary with the keys matching the parameter names specified in get_output_parameters and the values being
        the results from the raytracing
        """
        
        # focusing is not implemented
        # bottom reflections are not implemented

        output_dict = {
            'speed_type': self._index_to_tuple(i_solution)[1] + 1,
            'launch_vector': self.get_launch_vector(i_solution),
            'focusing_factor': 1,
            'ray_tracing_reflection': 1.0 if self.get_reflection_angle(i_solution) is not None else 0.0,
            'ray_tracing_reflection_case': 1.0,
            'ray_tracing_solution_type': self.get_solution_type(i_solution)            
        }
        return output_dict

    def get_number_of_raytracing_solutions(self):
        """
        Function that returns the maximum number of raytracing solutions that can exist between each given
        pair of start and end points
        """
        return 4

    def get_config(self):
        """
        Function that returns the configuration currently used by the raytracer
        """
        return self.__config

    def set_config(self, config):
        """
        Function to change the configuration file used by the raytracer

        Parameters:
        ----------------
        config: dict
            The new configuration settings
        """
        self.__config = config

class ray:
    '''
        utility class for 1 ray

        rays are defined by start and end points, type of 'index of refraction' used, and the type of ray (direct of refracted/reflected)

        object attributes:
            * x0, y0, x0 : initial ray coordinates
            * xf, yf, zf : final ray coordinates
            * ray : solution of ODE integrator
                ray.t : discrete arclength values along the ray
                ray.y : discrete ray position, momentum and (unscaled) time values along the ray
            * travel_time : final travel time for ray, in nanoseconds
            * launch_vector : unit vector for the ray's launch direction
            * receive_vector : unit vector for the ray's receive direction 
            * initial_wavefront : unit vector for the initial p vector
            * final_wavefront : unit vector for the final p vector
            * initial/final_E_pol : unit vector for the initial/final electric field polarization
            * initial/final_B_pol : unit vector for the intial/final magnetic flux density polarization
                NOTE: B and H are parallel since the permeability = 1, generally this is not true
    '''

    def __init__(self, x0, y0, z0, xf, yf, zf, ntype, raytype, medium, label=None):
        '''
            initializes object
            
            x0, y0, z0 : intial cartesian coordinates of the ray

            xf, yf, zf : final cartesian coordinates fo the ray

            ntype : selects which root to use for |p| calculation, can be either 1 or 2
                * in uniaxial media, 1 = extraordinary and 2 = ordinary root

            raytype : defines which solution to search for, can be either 1 or 2
                * 1 = directish, in NRMC this is the direct ray
                * 2 = refractedish, in NRMC this can be either the reflected or refracted ray

            eps : function handle for the material's permittivity, must be a function of position only
                  if you want to use NRMC raytracing for initial guesses, it must be an exponential profile for the scalar index of refraction approximation
        '''

        self.label = label

        self.x0, self.y0, self.z0 = x0, y0, z0
        self.xf, self.yf, self.zf = xf, yf, zf
        
        if raytype == 1 or raytype == 2:
            self.raytype = raytype
        else:
            print(raytype)
            raise RuntimeError('Please enter a valid ray type (1 or 2)')
        
        if ntype == 1 or ntype == 2:
            self.ntype = ntype
        else:
            print(ntype)
            raise RuntimeError('Please enter a valid index of refraction type (1 or 2)')
        
        self.medium = medium

        self.isaniso = hasattr(self.medium, 'aniso')
        if self.isaniso:
            self.isaniso = self.medium.aniso

        if self.isaniso:
            self.eps = self.medium.get_permittivity_tensor
        else:
            self.eps = lambda r : (self.medium.get_index_of_refraction(r))**2

        self._travel_time = 0.
        self._path_length = 0.
        self._ray = OptimizeResult(t=[], y=[])
        self._launch_vector = []
        self._receive_vector = []
        self._initial_E_pol = []
        self._initial_B_pol = []
        self._final_E_pol = []
        self._final_B_pol = []
        self._xsign = np.sign(xf - x0)
        self._ysign = np.sign(yf - y0)
        self._reflected = False
        self._solution_type = 0
        self._reflect_angle = None

    def copy_ray(self, odesol):
        if odesol != (None, None):
            self._ray = odesol
            self._ray.t *= self._ray.t * units.meter
            self._ray.y *= self._ray.y * units.meter
            self._travel_time = 1e9*odesol.y[-1,-1]/speed_of_light * units.nanosecond
            self._path_length = odesol.t[-1]
            self._launch_vector = self._unitvect(np.array(self._ray.y[0:3, 1] - self._ray.y[0:3, 0])) * units.radian
            self._receive_vector = self._unitvect(np.array(self._ray.y[0:3, -1] - self._ray.y[0:3,-2])) * units.radian
            
            if self._reflected:
                idxtmp = np.where(self._ray.y[2, :] == 0.)[0][0]
                #print('reflected index', idxtmp)
                self._reflect_angle = np.arccos(self._unitvect(np.array(self._ray.y[0:3, idxtmp+1] - self._ray.y[0:3, idxtmp])))

            self.initial_wavefront = self._unitvect(np.array(self._ray.y[3:6, 0]))
            self.final_wavefront = self._unitvect(np.array(self._ray.y[3:6, -1]))
            
            self._initial_E_pol, self._initial_B_pol = self._get_pols(0)
            self._final_E_pol, self._final_B_pol = self._get_pols(-1)
            
            if self._reflected:
                self._solution_type = 3
            else:
                if self.zf < np.max(self._ray.y[2, :]):
                    self._solution_type = 2
                else:
                    self._solution_type = 1

    def get_ray_parallel(self, q, sguess, phiguess, thetaguess):
        q.put(self.get_path(sguess, phiguess, thetaguess))

    def hit_top(self, s, u):
        return u[2]
    
    hit_top.terminal = True

    def hit_bot(t, y):
        return np.abs(y[1]) - 2800
    
    def ode(self, s, u):
        '''
            RHS of q and p ODE system, with travel time ODE

            takes in values and converts derivatives w.r.t. arclength (NuRMC uses pathlength)
        '''
        
        q = u[:3]
        p = u[3:-1]
        phat = p/np.linalg.norm(p, 2)
        qdot = self.DpH(q, p)
        qdn = np.linalg.norm(qdot, 2)
        pdot = -self.DqH(q, p)

        stmp = self._sign(qdot)
        qdot = stmp*qdot/qdn
        pdot = stmp*pdot/qdn

        if np.dot(phat, qdot) > 1:
            cosang = 1
        elif np.dot(phat, qdot) < -1: 
            cosang = -1
        else:
            cosang = np.dot(phat, qdot)

        return np.array([qdot[0], qdot[1], qdot[2], pdot[0], pdot[1], pdot[2], np.dot(p, qdot)])
    
    def _sign(self, q):
        x, y, z = q
        
        if self._xsign == 0 and np.abs(x) <= 1e-8:
            if self._ysign == np.sign(y):
                return 1
            else:
                return -1
        elif self._ysign == 0 and np.abs(y) <= 1e-8: 
            if self._xsign == np.sign(x):
                return 1
            else:
                return -1
        else: 
            if self._xsign != 0 and self._ysign == 0:
                if np.sign(x) == self._xsign:
                    return 1
                else:
                    return -1
            elif self._xsign == 0 and self._ysign !=0:
                if np.sign(y) == self._ysign:
                    return 1
                else:
                    return -1
            else:
                if np.sign(x) == self._xsign and np.sign(y) == self._ysign:
                    return 1
                else:
                    return -1

            
    def HMat(self, q, p):
        D2 = np.array([[-p[1]**2 - p[2]**2, p[0]*p[1], p[0]*p[2]],
            [p[0]*p[1], -p[0]**2-p[2]**2, p[1]*p[2]],
            [p[2]*p[0], p[2]*p[1], -p[0]**2-p[1]**2]])
        eps = self.eps(q)
        return self.eps(q)+D2
    
    def det(self, A):
        #return A[0,0]*(A[1,1]*A[2,2] - A[1,2]*A[2,1]) - A[0,1]*(A[0,1]*A[2,2] - A[1,2]*A[2,0]) + A[0,2]*(A[1,0]*A[2,1] - A[1,1]*A[2,0])
        return np.linalg.det(A)

    def H(self, q, p):
        return np.linalg.det(self.HMat(q, p))

    def DqH(self, q, p):
        q = np.array(q)
        p = np.array(p)
        macheps = 7./3 - 4./3 -1
        h = (macheps)**(2/3)*np.max(np.abs(q))
        DxH = (self.H([q[0] + h, q[1], q[2]], p) - self.H([q[0] - h, q[1], q[2]], p))/(2*h)
        DyH = (self.H([q[0], q[1] + h, q[2]], p) - self.H([q[0], q[1]- h, q[2]], p))/(2*h)
        DzH = (self.H([q[0], q[1], q[2] + h], p) - self.H([q[0], q[1], q[2] - h], p))/(2*h)
        return np.array([DxH, DyH, DzH])

    def DpH(self, q, p):
        q = np.array(q)
        p = np.array(p)
        macheps = 7./3 - 4./3 -1
        h = (macheps)**(2/3)*np.max(np.abs(p))
        Dp1H = (self.H(q, [p[0] + h, p[1], p[2]]) - self.H(q, [p[0] - h, p[1], p[2]]))/(2*h)
        Dp2H = (self.H(q, [p[0], p[1] + h, p[2]]) - self.H(q, [p[0], p[1] - h, p[2]]))/(2*h)
        Dp3H = (self.H(q, [p[0], p[1], p[2] + h]) - self.H(q, [p[0], p[1], p[2] - h]))/(2*h)
        return np.array([Dp1H, Dp2H, Dp3H])
        
    def adj(self, A):
        '''computes the adjugate for a 3x3 matrix'''
        return 0.5*np.eye(3)*(np.trace(A)**2 - np.trace(A@A)) - np.trace(A)*A + A@A
    
    def n(self, r, rdot):
        '''
            chooses index of refraction based on 

            r : position
            rdot : p direction
        '''
        
        if self.isaniso:
            n1, n2 = self.medium.get_index_of_refraction(r, rdot)
            if self.ntype == 1:
                return n1
            if self.ntype == 2:
                return n2
        else:
            return self.medium.get_index_of_refraction(r)


    def shoot_ray(self, sf, phi0, theta0, solver='RK45'): 
        '''
            solves the ray ODEs given arclength and intial angles for p

            sf : final arclength value for integration
            phi0 : azimuth angle for p
            theta0 : zenith angle for p
        '''
        
        idir = np.array([np.cos(phi0)*np.sin(theta0), np.sin(phi0)*np.sin(theta0), np.cos(theta0)])
        dx0, dy0, dz0 = self.n([self.x0, self.y0, self.z0], idir)*idir
        
        self._pxsign = np.sign(dx0)
        self._pysign = np.sign(dy0)
        
        self._reflected = False

        mstep = max(np.abs(sf), np.sqrt((self.xf - self.x0)**2+(self.yf-self.y0)**2 + (self.zf-self.z0)**2))/30
        sol=solve_ivp(self.ode, [0, sf], [self.x0, self.y0, self.z0, dx0, dy0, dz0, 0], method=solver, events=self.hit_top, max_step=mstep) 
        if len(sol.t_events[0]) == 0:
            sol.t = np.abs(sol.t)
            sol.y[-1,:] = np.abs(sol.y[-1,:])
            return OptimizeResult(t=sol.t, y=sol.y)
        else:
            sinit = sol.t_events[0][0]
            evnt = sol.y_events[0][0]
            sol2 = solve_ivp(self.ode, [sinit, sf], [evnt[0], evnt[1], 0, evnt[3], evnt[4], -evnt[5], evnt[6]], method=solver, max_step=mstep)
            tvals = np.hstack((sol.t[:-1], sol2.t))
            yvals = np.hstack((sol.y[:, :len(sol.t[:-1])], sol2.y))
            tvals = np.abs(tvals)
            yvals[-1,:] = np.abs(yvals[-1,:])
            self._reflected = True
            return OptimizeResult(t=tvals, y=yvals)

    def _rootfn(self, args):
        '''
            function for rootfinder, returns absolute distance of x, y, z components
        
            args: tuple of arclength, azimuth and zenith angles
        '''
        sol = self.shoot_ray(args[0], args[1], args[2])
        return np.array([sol.y[0, -1] - self.xf, sol.y[1, -1] - self.yf, sol.y[2, -1] - self.zf])

    def _distsq(self, args):
        '''
            function for rootfinder, returns absolute distance (scalar)
        
            args: tuple of arclength, azimuth and zenith angles
        '''
        sol = self.shoot_ray(args[0], args[1], args[2])
        return (sol.y[0, -1] - self.xf)**2 +  (sol.y[1, -1] - self.yf)**2 + (sol.y[2, -1] - self.zf)**2
    
    def _unitvect(self, v):
        '''returns unit vector in the direction of v'''
        return v/np.linalg.norm(v)
    
    def _get_pols(self, idx):
        '''
            given an array index idx, computes E and B fields at position[idx] and p[idx] along the ray using formulas in Chen
        
            idx : array index
        '''

        q = self._ray.y[0:3, idx]
        p = self._ray.y[3:6, idx]
        #n = np.linalg.norm(p)
        n = self.n(q, p)

        det = lambda a : np.linalg.det(a)
        
        e = np.zeros(3)
        b = np.zeros(3)
        
        
        J = self.eps(q) - n**2*np.eye(3)
        J[np.abs(J) < 1e-5] = 0
        dJ = np.abs(det(J))
            
        #print(idx, '\n-------------')

        if dJ >= 1e-7:
            #print(self.ntype, self.raytype, 'J is nonsingular', dJ)
            #print(J)
            e = self.adj(J) @ p
            e = e / np.linalg.norm(e)
            b = np.cross(p, e)
            b = b / np.linalg.norm(b)
            #return e, b
        else:
            #print(self.ntype, self.raytype, 'J is singular')
            deps = det(self.eps(q))
            M2 = n**2*(np.dot(p, np.dot(self.eps(q),p)))
            #print(self.ntype, self.raytype, np.abs(deps - M2))
            if np.abs(deps - M2) < 1e-3:
                #print('isotropic')
                #local plane of incidence defined by p cross grad(n)
                localgrad = -self.DqH(q, p)
                localgrad = self._unitvect(localgrad)
                planenorm = self._unitvect(np.cross(p, localgrad))

                if self.ntype == 2:
                    #force ntype 2 to be a TE wave
                    e = planenorm
                    
                    b = np.cross(p, e)
                    b = b/np.linalg.norm(b)
                    
                    #return e, b
                elif self.ntype == 1:
                    #force ntype 1 to be a TM wave
                    e = np.cross(p, planenorm)
                    e = e/np.linalg.norm(e)

                    b = np.cross(p, e)
                    b = b/np.linalg.norm(b)

                    #return e, b
                else:
                    return ValueError('not a valid ntype')
            else:
                #print('anisotropic')
                vals, vects = np.linalg.eig(self.eps(q))
                #print(self.ntype, self.raytype, n**2 - vals)
                idx = np.where(np.abs(n**2 - vals) == np.abs(n**2 - vals).min())[0]
                #print(idx)
                u = np.zeros((3, len(idx)))

                for i in range(len(idx)):
                    u[:, i] = vects[:, idx[i]]

                keps = self.eps(q)@self._unitvect(p)
                #print(u.T@keps)
                #print(u)
                #print(np.dot(u.T, np.dot(self.eps(q), self._unitvect(p))))
                u = u[:, np.argmin(np.abs((self._unitvect(p)@self.eps(q))@u))]
                #print(u)
                b = np.cross(u, p)
                b = b / np.linalg.norm(b)
                e = -np.linalg.inv(self.eps(q)) @ np.cross(p, b)
                e = e / np.linalg.norm(e)
                
        #print(self.ntype, self.raytype, e, b)
        #print(np.dot(p, self.eps(q) @ e), np.dot(p, b))
        
        return e, b

    def _get_ray(self, sf, phi, theta):
        '''
            finds ray using rootfinding via the shooting method for BVPs, then computes ray attributes

            sf : initial guess for ray's arclength
            phi : intial guess for azimuth angle
            theta : intial guess for zenith angle
        '''
        
        if self._ray.t == []:
            minsol = root(self._rootfn, [sf, phi, theta], method='lm', options={'ftol':1e-10, 'xtol':1e-12, 'maxiter':100, 'eps':1e-7, 'diag':[1/1e3, 1/1e-8, 1], 'factor':10})
            #print(self.ntype, self.raytype, minsol.success, minsol.message)
            self.copy_ray(self.shoot_ray(*minsol.x))
            if self._reflected:
                self._solution_type = 3
            else:
                if self.zf < np.max(self._ray.y[2, :]):
                    self._solution_type = 2
                else:
                    self._solution_type = 1
            
            return self._ray
        else:
            return self._ray
     
    def get_path(self, sg, phig, thetag):
        if self._ray.t == []:     
            self._get_ray(sg, phig, thetag)
            print(self.ntype, self.raytype, np.sqrt((self.xf - self._ray.y[0, -1])**2 + (self.yf - self._ray.y[1,-1])**2 + (self.zf - self._ray.y[2,-1])**2))
            return self._ray
        else:
            return self._ray
    
    def get_ray(self):
        return self._ray

    def get_travel_time(self):
        return self._travel_time

    def get_path_length(self):
        return self._path_length

    def get_initial_E_pol(self):
        return self._initial_E_pol

    def get_final_E_pol(self):
        return self._final_E_pol

    def get_initial_B_pol(self):
        return self._initial_B_pol

    def get_final_B_pol(self):
        return self._final_B_pol

    def get_solution_type(self):
        return self._solution_type

    def get_launch_vector(self):
        return self._launch_vector

    def get_receive_vector(self):
        return self._receive_vector

    def get_reflection_angle(self):
        return self._reflect_angle

class rays(ray):
    '''
        wrapper for ray class

        finds all 4 solutions: 1 refracted and 1 direct for each of the 2 ntype calculations
    '''

    def __init__(self, medium, dr=None, par=False, guess_medium = medium.get_ice_model('ARAsim_southpole')):
        #naming convention is r_ik, i = ntype, k = raytype
        self.__medium = medium
        self.__par = par
        self.__guess_medium = guess_medium
    
    def set_start_and_end_point(self, x1, x2):
        self.__x0, self.__y0, self.__z0 = x1
        self.__xf, self.__yf, self.__zf = x2
        
        self.r = np.array([[None, None], [None, None]])
        for i in [1,2]:
            for k in [1,2]:
                self.r[i-1, k-1] = ray(self.__x0, self.__y0, self.__z0, 
                        self.__xf, self.__yf, self.__zf, i, k, self.__medium)

    def set_guess(self):
        g = analyticraytracing.ray_tracing(self.__guess_medium, n_frequencies_integration = 1)
        g.set_start_and_end_point(np.array([self.__x0, self.__y0, self.__z0]), np.array([self.__xf, self.__yf, self.__zf]))
        g.find_solutions()
        self.sg1, self.sg2 = g.get_path_length(0), g.get_path_length(1)
                
        lv1, lv2 = g.get_launch_vector(0), g.get_launch_vector(1)
        lv1, lv2 = lv1/np.linalg.norm(lv1), lv2/np.linalg.norm(lv2)
        
        self.phig = np.arctan2((self.__yf - self.__y0), (self.__xf - self.__x0))
        
        self.thetag1, self.thetag2 = np.arccos(lv1[2]), np.arccos(lv2[2])
    
    def get_guess(self, raytype):
        if raytype == 1:
            return (self.sg1, self.phig, self.thetag1)
        if raytype == 2:
            return (self.sg2, self.phig, self.thetag2)

    def get_rays(self):
        self.set_guess()

        if self.__par == True:
            q = np.array([[Queue() for k in [1,2]] for i in [1,2]])

            p = np.array([[Process(target=self.r[i-1,k-1].get_ray_parallel, args=(q[i-1, k-1], *self.get_guess(k))) for k in [1,2]] for i in [1,2]])

            [[p[i,k].start() for i in range(2)] for k in range(2)]
            [[p[i,k].join() for i in range(2)] for k in range(2)]

            for i in [0,1]:
                for k in [0,1]:
                    self.r[i,k].copy_ray(q[i,k].get())
            
        else:
            for i in [0,1]:
                for k in [0,1]:
                    self.r[i, k].get_path(*self.get_guess(k+1))
        
        # check if the order of the rays is right
        for i in [0,1]:
            if max(self.r[i,0].get_ray().y[2,:]) > max(self.r[i,1].get_ray().y[2,:]):
                tmp = self.r[i,0]
                self.r[i,0] = self.r[i,1]
                self.r[i,1] = tmp
                del tmp

    def get_path(self, i, k):
        return self.r[i,k].get_path(*self.get_guess(k+1))

    def get_travel_time(self, i, k):
        return self.r[i,k].get_travel_time()

    def get_path_length(self, i, k):
        return self.r[i,k].get_path_length()
    
    def get_initial_E_pol(self, i, k):
        return self.r[i,k].get_initial_E_pol()

    def get_final_E_pol(self, i, k):
        return self.r[i,k].get_final_E_pol()

    def get_initial_B_pol(self, i, k):
        return self.r[i,k].get_initial_B_pol()

    def get_final_B_pol(self, i, k):
        return self.r[i,k].get_final_B_pol()

    def get_solution_type(self, i, k):
        return self.r[i,k].get_solution_type()

    def get_launch_vector(self, i, k):
        return self.r[i,k].get_launch_vector()

    def get_receive_vector(self, i, k):
        return self.r[i, k].get_receive_vector()

    def get_reflection_angle(self, i, k):
        return self.r[i,k].get_reflection_angle()
