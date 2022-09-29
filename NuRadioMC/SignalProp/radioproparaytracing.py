from __future__ import absolute_import, division, print_function
import numpy as np
from radiotools import helper as hp
import logging
from NuRadioMC.utilities import attenuation as attenuation_util
from NuRadioMC.utilities import medium_base
from scipy import interpolate, optimize
import NuRadioReco.utilities.geometryUtilities
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioMC.SignalProp.propagation_base_class import ray_tracing_base
from NuRadioMC.SignalProp.propagation import solution_types, solution_types_revert
import radiopropa
import scipy.constants 
import copy
import logging
import time
from matplotlib import pyplot as plt
logging.basicConfig()

"""
RadioPropa is a C++ module dedicated for ray tracing. It is a seperate module and
it has its own unit system. However, all object within NuRadio ecosystem are in the
NuRadio uit system. Therefore, when passing argument from NuRadio to RadioPropa, or
when receiving object from RadioPropa into NuRadio the units of object needed to be 
converted to the right unit system. Below is an example given for an object 'distance'

- from NuRadio to RadioPropa:
    distance_in_meter = distance_in_nuradio / units.meter
    --> this converts the distance from NuRadio units into SI unit meter  
    distance_in_radiopropa = distance_in_meter * radiopropa.meter
    --> this converts the distance from SI unit meter into RadioPropa units

- from RadioPropa to NuRadio:
    distance_in_meter = distance_in_radiopropa / radiopropa.meter
    --> this converts the distance from RadioPropa units into SI unit meter  
    distance_in_nuradio = distance_in_meter * units.meter
    --> this converts the distance from SI unit meter into NuRadio units
"""


class radiopropa_ray_tracing(ray_tracing_base):

    """ Numerical raytracing using Radiopropa. Currently this only works for icemodels 
    that have only changing refractive index in z. More information on RadioPropa and
    how to install it can be found at https://github.com/nu-radio/RadioPropa"""

    def __init__(self, medium, attenuation_model="SP1", log_level=logging.WARNING,
                 n_frequencies_integration=100, n_reflections=0, config=None, detector=None):

        """
        class initilization

        Parameters
        ----------
        medium: medium class
            class describing the index-of-refraction profile
        attenuation_model: string
            signal attenuation model
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
        n_reflections: int (default 0)
            in case of a medium with a reflective layer at the bottom, how many reflections should be considered
        config: dict or None
            a dictionary with the optional config settings. If None, the default config is used:
                config['propagation']['attenuate_ice'] = True
                config['propagation']['focusing_limit'] = 2
                config['propagation']['focusing'] = False
                config['speedup']['delta_C_cut'] = 40 * units.degree
                config['propagation']['radiopropa']['mode'] = 'iterative'
                config['propagation']['radiopropa']['max_traj_length'] = 10000
                config['propagation']['radiopropa']['iter_steps_channel'] = [25., 2., .5]
                config['propagation']['radiopropa']['auto_step_size'] = False
                config['propagation']['radiopropa']['iter_steps_zenith'] = [.5, .05, .005]
        detector: detector object
        """
        self.__logger = logging.getLogger('radiopropa_ray_tracing')
        self.__logger.setLevel(log_level)

        try:
            import radiopropa
        except ImportError:
            self.__logger.error('ImportError: This raytracer depends on radiopropa which could not be imported. '+
                               'Check wether all dependencies are installed correctly. More information on '+
                               'https://github.com/nu-radio/RadioPropa')
            raise ImportError('This raytracer depends on radiopropa which could not be imported. Check wether all dependencies are installed correctly. More information on https://github.com/nu-radio/RadioPropa')

        super().__init__(medium=medium, 
                         attenuation_model=attenuation_model,
                         log_level=log_level,
                         n_frequencies_integration=n_frequencies_integration, 
                         n_reflections=n_reflections,
                         config=config, 
                         detector=detector)

        self.set_config(config=config)
        self._ice_model = self._medium.get_ice_model_radiopropa()

        self.set_minimizer_tolerance()
        self._shower_axis = None ## this is given so we can limit the rays that are checked around the cherenkov angle
        self._rays = None


    def reset_solutions(self):
        """
        Resets the raytracing solutions back to None. This is useful to do 
        in the loop before a new raytracing is prepared.
        """
      
        super().reset_solutions()
        self._shower_axis = None
        self._rays = None

    def set_start_and_end_point(self, x1, x2):
        """
        Set the start and end points of the raytracing

        Parameters
        ----------
        x1: 3dim np.array
            start point of the ray
        x2: 3dim np.array
            stop point of the ray
        """
        super().set_start_and_end_point(x1, x2)
        self.set_iterative_step_sizes(step_zeniths=self._step_zeniths) #if auto is on this set the automated step size, otherwise nothing happens

    def set_shower_axis(self, shower_axis):
        """
        Set the the shower axis. This is oposite to the neutrino arrival direction

        Parameters
        ----------
        shower_axis: np.array of shape (3,), default unit
            the direction of the shower in cartesian coordinates
        """ 
        self._shower_axis = shower_axis / np.linalg.norm(shower_axis)

    def set_iterative_sphere_sizes(self, sphere_sizes=None):
        """
        Set the sphere_sizes for the iterative ray tracer

        Parameters
        ----------
        sphere_sizes: np.array of size (n,), default unit
            the sphere size used by the iterative ray tracer
            iteration from big to small observer around channel
        """
        if sphere_sizes is None:
            sphere_sizes = np.array(self._config['propagation']['radiopropa']['iter_steps_channel']) * units.meter

        if (sphere_sizes.ndim == 1):
            self._sphere_sizes = sphere_sizes
        else:
            self.__logger.error('sphere_sizes array should be 1 dimensional')
            raise ValueError('sphere_sizes array should be 1 dimensional')

    def set_iterative_step_sizes(self, step_zeniths= None):
        """
        Set the steps_sizes for the iterative ray tracer

        Parameters
        ----------
        sphere_sizes: np.array of size (n,), default unit
            the sphere size used by the iterative ray tracer
            iteration from big to small observer around channel
        step_zeniths: np.array size (n,), default unit
            the step size for theta used by the iterative ray tracer
            corresponding to the sphere size, should have same lenght as _sphere_sizes
        auto_step:  boolean
            defines whether or not an automatic step_size should be calculated for each
            sphere_size depending on the horizontal distance of the event
        """   
        if step_zeniths is None:
            step_zeniths = np.array(self._config['propagation']['radiopropa']['iter_steps_zenith']) * units.degree

        if self._auto_step:
            if (self._X1 != None) and (self._X2 != None):
                for s, sphere_size in enumerate(self._sphere_sizes):
                    self._step_zeniths[s] = min(abs(self.delta_theta_reflective(dz=sphere_size, n_bottom_reflections=self._n_reflections)),
                                               self._step_zeniths[s])
            else:
                return
        else:
            if (self._sphere_sizes.shape == step_zeniths.shape):      
                self._step_zeniths = step_zeniths
            else:
                self.__logger.error('sphere_sizes array and step_zeniths array should have the same dimensions')
                raise ValueError('sphere_sizes array and step_zeniths array should have the same dimensions')

    def activate_auto_step_size(self):
        self._auto_step = True
        self.set_iterative_step_sizes()

    def deactivate_auto_step_size(self):
        self._auto_step = False


    def set_cut_viewing_angle(self, cut):
        """
        Set a cut on the viewing angle around the cherenkov angle. Rays with a viewing 
        angle out of this range will be to dim and won't be seen --> limiting computing time

        Parameters
        ----------
        cut: float, default unit
             range around the cherenkov angle
        """
        self._cut_viewing_angle = cut

    def set_maximum_trajectory_length(self, max_traj_length):
        """
        Set a cut on the trajectory length. Otherwise computing infinite may be possible

        Parameters
        ----------
        max_traj_length: float, default units
            maxmimal length to trace a ray. tracing aborted when reached
        """
        self._max_traj_length = max_traj_length

    def raytracer_iterative(self, n_reflections=0):
        """
        Uses RadioPropa to find all the numerical ray tracing solutions between sphere X1 and X2.
        If reflections is bigger than 0, also bottom reflected rays are searched for with a max
        of n_reflections of the bottom
        """
        try:
            X1 = self._X1 * (radiopropa.meter/units.meter)
            X2 = self._X2 * (radiopropa.meter/units.meter)
        except TypeError: 
            self.__logger.error('NoneType: start or endpoint not initialized')
            raise TypeError('NoneType: start or endpoint not initialized')

      
        v = (self._X2 - self._X1)
        u = copy.deepcopy(v)
        u[2] = 0
        theta_direct, phi_direct = hp.cartesian_to_spherical(*v) # zenith and azimuth for the direct linear ray solution (radians)
        cherenkov_angle = np.arccos(1. / self._medium.get_index_of_refraction(self._X1))
        
        if np.linalg.norm(u) != 0:
            delta_theta = 2*abs(self.delta_theta_direct(dz=self._sphere_sizes[0]))
        else:
            delta_theta = self._step_sizes[0]/units.radian
            
        ## regions of theta with posible solutions (radians)
        launch_lower = [0]
        launch_upper = [theta_direct + delta_theta] # below theta_direct no solutions are possible without upward reflections

        if n_reflections > 0:
            if self.medium.reflection is None:
                self.__logger.error("a solution for {:d} reflection(s) off the bottom reflective layer is requested,"
                                    +"but ice model does not specify a reflective layer".format(n_reflections))
                raise AttributeError("a solution for {:d} reflection(s) off the bottom reflective layer is requested,"
                                    +"but ice model does not specify a reflective layer".format(n_reflections))
            else:
                z_refl = self._medium.reflection
                rho_channel = np.linalg.norm(u)
                if self._X2[2] > self._X1[2]: 
                    z_up = self._X2[2]
                    z_down = self._X1[2]
                else:
                    z_up = self._X1[2]
                    z_down = self._X2[2]
                rho_bottom = (rho_channel * (z_refl - z_down)) / (2*z_refl - z_up - z_down)
                alpha = np.arctan((z_down - z_refl)/rho_bottom)
                ## when reflection on the bottom are allowed, a initial region for theta from 180-alpha to 180 degrees is added
                launch_lower.append(((np.pi/2 + alpha) - 2*abs(self.delta_theta_bottom(dz=self._sphere_sizes[0], z_refl=z_refl) / units.radian)))
                launch_upper.append(np.pi)
        
        for s,sphere_size in enumerate(self._sphere_sizes):

            sphere_size = sphere_size * (radiopropa.meter/units.meter)
            detected_rays = []
            results = []

            ##define module list for simulation
            sim = radiopropa.ModuleList()
            sim.add(radiopropa.PropagationCK(self._ice_model.get_scalar_field(), 1E-8, .001, 1.)) ## add propagation to module list
            for module in self._ice_model.get_modules().values(): 
                sim.add(module)
            sim.add(radiopropa.MaximumTrajectoryLength(self._max_traj_length * (radiopropa.meter/units.meter)))

            ## define observer for detection (channel)            
            obs = radiopropa.Observer()
            obs.setDeactivateOnDetection(True)
            channel = radiopropa.ObserverSurface(radiopropa.Sphere(radiopropa.Vector3d(*X2), sphere_size)) ## when making the radius larger than 2 meters, somethimes three solution times are found
             
            obs.add(channel)
            sim.add(obs)

            ## define observer for stopping simulation (boundaries)
            obs2 = radiopropa.Observer()
            obs2.setDeactivateOnDetection(True)
            w = (u / np.linalg.norm(u)) * 2*sphere_size
            boundary_behind_channel = radiopropa.ObserverSurface(radiopropa.Plane(radiopropa.Vector3d(*(X2 + w)), radiopropa.Vector3d(*w)))
            obs2.add(boundary_behind_channel)
            boundary_above_surface = radiopropa.ObserverSurface(radiopropa.Plane(radiopropa.Vector3d(0, 0, 1*radiopropa.meter), radiopropa.Vector3d(0, 0, 1)))
            obs2.add(boundary_above_surface)
            sim.add(obs2)
            
            #create total scanning range from the upper and lower thetas of the bundles
            step = self._step_zeniths[s] / units.radian
            theta_scanning_range = np.array([])
            for iL in range(len(launch_lower)):
                new_scanning_range = np.arange(launch_lower[iL], launch_upper[iL]+step, step)
                theta_scanning_range = np.concatenate((theta_scanning_range, new_scanning_range))

            for theta in theta_scanning_range:
                ray_dir = hp.spherical_to_cartesian(theta, phi_direct)
                
                def delta(ray_dir,shower_dir):
                    viewing = np.arccos(np.dot(shower_dir, ray_dir)) * units.radian
                    return viewing - cherenkov_angle

                if (self._shower_axis is None) or (abs(delta(ray_dir,self._shower_axis)) < self._cut_viewing_angle):
                    source = radiopropa.Source()
                    source.add(radiopropa.SourcePosition(radiopropa.Vector3d(*X1)))
                    source.add(radiopropa.SourceDirection(radiopropa.Vector3d(*ray_dir)))
                    sim.setShowProgress(True)
                    ray = source.getCandidate()
                    sim.run(ray, True)
                    
                    current_rays = [ray]
                    while len(current_rays) > 0:
                        next_rays = []
                        for ray in current_rays:
                            if channel.checkDetection(ray.get()) == radiopropa.DETECTED:
                                detected_rays.append(ray)
                                result = {}
                                if n_reflections == 0:
                                    result['reflection']=0
                                    result['reflection_case']=1
                                elif self._ice_model.get_modules()["bottom reflection"].get_times_reflectedoff(ray.get()) <= n_reflections: 
                                    result['reflection']=self._ice_model.get_modules()["bottom reflection"].get_times_reflectedoff(ray.get())
                                    result['reflection_case']=int(np.ceil(theta/np.deg2rad(90)))
                                results.append(result)
                            for secondary in ray.secondaries:
                                next_rays.append(secondary)
                        current_rays = next_rays
            #loop over previous rays to find the upper and lower theta of each bundle of rays
            #uses step, but because step is initialized after this loop this ios the previous step size as intented
            if len(detected_rays) > 0:
                launch_lower.clear()
                launch_upper.clear()
                launch_theta_prev = None
                for iDC,DC in enumerate(detected_rays):
                    launch_theta = DC.getLaunchVector().getTheta()/radiopropa.rad
                    if iDC == (len(detected_rays)-1) or iDC == 0:
                        if iDC == 0: 
                            launch_lower.append(launch_theta-step)
                        if iDC == (len(detected_rays)-1): 
                            launch_upper.append(launch_theta+step)
                    elif abs(launch_theta - launch_theta_prev) > 1.1*step: ##take 1.1 times the step to be sure the next ray is not in the bundle of the previous one
                        launch_upper.append(launch_theta_prev+step)
                        launch_lower.append(launch_theta-step)
                    else:
                        pass
                    launch_theta_prev = launch_theta
            else:
                #if detected_rays is empty, no solutions where found and the tracer is terminated
                break

        self._rays = detected_rays
        self._results = results
        self.__used_method = 'iterator'
        launch_bundles = np.transpose([launch_lower, launch_upper])
        return launch_bundles

    def set_minimizer_tolerance(self, xtol=1e-3*units.deg, ztol=1e-3*units.meter):
        self.__xtol = xtol
        self.__ztol = ztol


    def raytracer_hybrid_minimizer(self,n_reflections=0):
        """
        Uses RadioPropa to first find all the numerical ray tracing solutions between sphere x1 
        and sphere x2 for a big sphere. After which it uses the scipy optimize.minimize module
        to find the best path in these angle intervals. 
        Tracer does not work for reflective bottoms or secondary creation at the moment
        """

        try:
            X1 = self._X1 * (radiopropa.meter/units.meter)
            X2 = self._X2 * (radiopropa.meter/units.meter)
        except TypeError: 
            self.__logger.error('NoneType: start or endpoint not initialized')
            raise TypeError('NoneType: start or endpoint not initialized')
      
        def get_ray(theta,phi):
            ray_dir = hp.spherical_to_cartesian(theta,phi)
            if ray_dir.shape==(3,1): ray_dir = ray_dir.T[0] #doesn't always give the right shape
            source = radiopropa.Source()
            source.add(radiopropa.SourcePosition(radiopropa.Vector3d(*X1)))
            source.add(radiopropa.SourceDirection(radiopropa.Vector3d(*ray_dir)))
            ray = source.getCandidate()
            return ray

        def shoot_ray(theta):
            ray = get_ray(theta,phi_direct)
            sim.run(ray, True)
            return ray

        def cot(x):
            return 1/np.tan(x)

        def arccot(x):
            return np.arctan(-x) + np.pi/2

        def delta_z(cot_theta):
            theta = arccot(cot_theta)
            ray = shoot_ray(theta)
            ray_endpoint = self.get_path_candidate(ray)[-1]
            return (ray_endpoint-self._X2)[2]

        def delta_z_squared(cot_theta):
            return delta_z(cot_theta)**2

        def MinimizeAble(lower,upper):
            #if (index > 1):
            #    return False
            if ((len(lower) == 2) and (len(upper)==2)):
                if (lower[1] > upper[0]) and (lower[0] < upper[0]) and (lower[1] < upper[1]):
                    return True
                else:
                    return False
            else:
                return False

        LetsMinimize = False
 
        v = (self._X2 - self._X1)
        u = copy.deepcopy(v)
        u[2] = 0
        theta_direct, phi_direct = hp.cartesian_to_spherical(*v) # zenith and azimuth for the direct linear ray solution (radians)
        cherenkov_angle = np.arccos(1. / self._medium.get_index_of_refraction(self._X1))
        
        if np.linalg.norm(u) != 0:
            delta_theta = 2*abs(self.delta_theta_direct(dz=self._sphere_sizes[0]))
        else:
            delta_theta = self._step_sizes[0]/units.radian
            
        ## regions of theta with posible solutions (radians)
        launch_lower = [0]
        launch_upper = [theta_direct + delta_theta] # below theta_direct no solutions are possible without upward reflections

        if n_reflections > 0:
            if self.medium.reflection is None:
                self.__logger.error("a solution for {:d} reflection(s) off the bottom reflective layer is requested,"
                                    +"but ice model does not specify a reflective layer".format(n_reflections))
                raise AttributeError("a solution for {:d} reflection(s) off the bottom reflective layer is requested,"
                                    +"but ice model does not specify a reflective layer".format(n_reflections))
            else:
                z_refl = self._medium.reflection
                rho_channel = np.linalg.norm(u)
                if self._X2[2] > self._X1[2]: 
                    z_up = self._X2[2]
                    z_down = self._X1[2]
                else:
                    z_up = self._X1[2]
                    z_down = self._X2[2]
                rho_bottom = (rho_channel * (z_refl - z_down)) / (2*z_refl - z_up - z_down)
                alpha = np.arctan((z_down - z_refl)/rho_bottom)
                ## when reflection on the bottom are allowed, a initial region for theta from 180-alpha to 180 degrees is added
                launch_lower.append(((np.pi/2 + alpha) - 2*abs(self.delta_theta_bottom(dz=self._sphere_sizes[0], z_refl=z_refl) / units.radian)))
                launch_upper.append(np.pi)
        
        for s,sphere_size in enumerate(self._sphere_sizes):

            sphere_size = sphere_size * (radiopropa.meter/units.meter)
            detected_rays = []
            results = []

            ##define module list for simulation
            sim = radiopropa.ModuleList()
            sim.add(radiopropa.PropagationCK(self._ice_model.get_scalar_field(), 1E-8, .001, 1.)) ## add propagation to module list
            for module in self._ice_model.get_modules().values(): 
                sim.add(module)
            sim.add(radiopropa.MaximumTrajectoryLength(self._max_traj_length * (radiopropa.meter/units.meter)))

            ## define observer for detection (channel)            
            obs = radiopropa.Observer()
            obs.setDeactivateOnDetection(True)
            channel = radiopropa.ObserverSurface(radiopropa.Sphere(radiopropa.Vector3d(*X2), sphere_size)) ## when making the radius larger than 2 meters, somethimes three solution times are found
             
            obs.add(channel)
            sim.add(obs)

            ## define observer for stopping simulation (boundaries)
            obs2 = radiopropa.Observer()
            obs2.setDeactivateOnDetection(True)
            w = (u / np.linalg.norm(u)) * 2*sphere_size
            boundary_behind_channel = radiopropa.ObserverSurface(radiopropa.Plane(radiopropa.Vector3d(*(X2 + w)), radiopropa.Vector3d(*w)))
            obs2.add(boundary_behind_channel)
            boundary_above_surface = radiopropa.ObserverSurface(radiopropa.Plane(radiopropa.Vector3d(0, 0, 1*radiopropa.meter), radiopropa.Vector3d(0, 0, 1)))
            obs2.add(boundary_above_surface)
            sim.add(obs2)
            
            #create total scanning range from the upper and lower thetas of the bundles
            step = self._step_zeniths[s] / units.radian
            theta_scanning_range = np.array([])
            for iL in range(len(launch_lower)):
                new_scanning_range = np.arange(launch_lower[iL], launch_upper[iL]+step, step)
                theta_scanning_range = np.concatenate((theta_scanning_range, new_scanning_range))

            for theta in theta_scanning_range:
                ray_dir = hp.spherical_to_cartesian(theta, phi_direct)
                
                def delta(ray_dir,shower_dir):
                    viewing = np.arccos(np.dot(shower_dir, ray_dir)) * units.radian
                    return viewing - cherenkov_angle

                if (self._shower_axis is None) or (abs(delta(ray_dir,self._shower_axis)) < self._cut_viewing_angle):
                    source = radiopropa.Source()
                    source.add(radiopropa.SourcePosition(radiopropa.Vector3d(*X1)))
                    source.add(radiopropa.SourceDirection(radiopropa.Vector3d(*ray_dir)))
                    sim.setShowProgress(True)
                    ray = source.getCandidate()
                    sim.run(ray, True)
                    
                    current_rays = [ray]
                    while len(current_rays) > 0:
                        next_rays = []
                        for ray in current_rays:
                            if channel.checkDetection(ray.get()) == radiopropa.DETECTED:
                                detected_rays.append(ray)
                                result = {}
                                if n_reflections == 0:
                                    result['reflection']=0
                                    result['reflection_case']=1
                                elif self._ice_model.get_modules()["bottom reflection"].get_times_reflectedoff(ray.get()) <= n_reflections: 
                                    result['reflection']=self._ice_model.get_modules()["bottom reflection"].get_times_reflectedoff(ray.get())
                                    result['reflection_case']=int(np.ceil(theta/np.deg2rad(90)))
                                results.append(result)
                            for secondary in ray.secondaries:
                                next_rays.append(secondary)
                        current_rays = next_rays
            #loop over previous rays to find the upper and lower theta of each bundle of rays
            #uses step, but because step is initialized after this loop this ios the previous step size as intented
            if len(detected_rays) > 0:
                launch_lower.clear()
                launch_upper.clear()
                launch_theta_prev = None
                for iDC,DC in enumerate(detected_rays):
                    launch_theta = DC.getLaunchVector().getTheta()/radiopropa.rad
                    if iDC == (len(detected_rays)-1) or iDC == 0:
                        if iDC == 0: 
                            launch_lower.append(launch_theta-step)
                        if iDC == (len(detected_rays)-1): 
                            launch_upper.append(launch_theta+step)
                    elif abs(launch_theta - launch_theta_prev) > 1.1*step: ##take 1.1 times the step to be sure the next ray is not in the bundle of the previous one
                        launch_upper.append(launch_theta_prev+step)
                        launch_lower.append(launch_theta-step)
                    else:
                        pass
                    launch_theta_prev = launch_theta
                isMinimizeAble = MinimizeAble(launch_lower,launch_upper)
                if isMinimizeAble and (s < 1):
                    LetsMinimize = True
                    break

            else:
                #if detected_rays is empty, no solutions where found and the tracer is terminated
                break
        if LetsMinimize:
            iterative = False
            ##define module list for simulation, this needs to be redone to get rid of the spherical observer
            sim = radiopropa.ModuleList()
            sim.add(radiopropa.PropagationCK(self._ice_model.get_scalar_field(), 1E-8, .001, 1.)) ## add propagation to module list
            for module in self._ice_model.get_modules().values():
                sim.add(module)
            sim.add(radiopropa.MaximumTrajectoryLength(self._max_traj_length * (radiopropa.meter/units.meter)))

            ## define observer for detection (channel)
            obs = radiopropa.Observer()
            obs.setDeactivateOnDetection(True)
            plane_channel = radiopropa.ObserverSurface(radiopropa.Plane(radiopropa.Vector3d(*X2), radiopropa.Vector3d(*u)))
            obs.add(plane_channel)
            sim.add(obs)

            detected_rays = []
            detected_theta = []
            #we minimize the cotangens of the zenith to reflect the same resolution in z to the different angles (vertical vs horizontal)

            for i in range(len(launch_lower)):
                ll = launch_lower[i]
                lu = launch_upper[i]
                cotll = cot(ll)
                cotlu = cot(lu)
                if cotlu > cotll:
                    bounds = scipy.optimize.Bounds(lb=cotll, ub=cotlu, keep_feasible=False)
                else:
                    bounds = scipy.optimize.Bounds(lb=cotlu, ub=cotll, keep_feasible=False)
                root = optimize.minimize(delta_z_squared,x0=cot((ll+lu)/2),bounds=bounds,options={'xatol':self.__xtol**2,'fatol':self.__ztol**2},method='Nelder-Mead')
                theta = arccot(root.x)
                detected_theta.append(theta)
                detected_rays.append(shoot_ray(theta))
            self._rays = detected_rays
            self._results = results
            launch_bundles = np.transpose([launch_lower, launch_upper])
            self.__used_method = 'hybrid minimizer'
            return launch_bundles,iterative

        else:
            iterative = True
            self._rays = detected_rays
            self._results = results
            self.__used_method = 'iterator'
            launch_bundles = np.transpose([launch_lower, launch_upper])
            return launch_bundles,iterative


        
    def raytracer_minimizer(self, n_reflections=0):
        """
        Uses RadioPropa to find all the numerical ray tracing solutions between sphere x1 and x2.
        Tracer does not work for reflective bottoms or secondary creation at the moment, it only
        allows for 2 solutions at the moment which should also be further improved.
        """
        try:
            x1 = self._X1  * (radiopropa.meter/units.meter)
            x2 = self._X2  * (radiopropa.meter/units.meter)
        except TypeError: 
            self.__logger.error('NoneType: start or endpoint not initialized')
            raise TypeError('NoneType: start or endpoint not initialized')

      
        v = (self._X2-self._X1)
        u = copy.deepcopy(v)
        u[2] = 0
        theta_direct, phi_direct = hp.cartesian_to_spherical(*v) ## zenith and azimuth for the direct linear ray solution (radians)

        if n_reflections > 0:
            self.__logger.error("Radiopropa minimizer tracer can not be used for reflections at the bottom")
            raise AttributeError("Radiopropa minimizer tracer can not be used for reflections at the bottom")

        ##define module list for simulation
        sim = radiopropa.ModuleList()
        sim.add(radiopropa.PropagationCK(self._ice_model.get_scalar_field(), 1E-8, .001, 1.)) ## add propagation to module list
        for module in self._ice_model.get_modules().values(): 
            sim.add(module)
        sim.add(radiopropa.MaximumTrajectoryLength(self._max_traj_length*(radiopropa.meter/units.meter)))

        ## define observer
        obs2 = radiopropa.Observer()
        obs2.setDeactivateOnDetection(True)
        plane_channel = radiopropa.ObserverSurface(radiopropa.Plane(radiopropa.Vector3d(*x2), radiopropa.Vector3d(*u)))
        obs2.add(plane_channel)
        sim.add(obs2)

        detected_rays = []
        detected_theta = []
        results = []
        res_angle = 0.001*units.degree/units.radian

        def get_ray(theta,phi):
            ray_dir = hp.spherical_to_cartesian(theta,phi)
            if ray_dir.shape==(3,1): ray_dir = ray_dir.T[0]
            source = radiopropa.Source()
            source.add(radiopropa.SourcePosition(radiopropa.Vector3d(*x1)))
            source.add(radiopropa.SourceDirection(radiopropa.Vector3d(*ray_dir)))
            ray = source.getCandidate()
            return ray

        def shoot_ray(theta):
            ray = get_ray(theta,phi_direct)
            sim.run(ray, True)
            return ray

        def cot(x):
            return 1/np.tan(x)
        def arccot(x):
            return np.arctan(-x) + np.pi/2

        def delta_z(cot_theta):
            theta = arccot(cot_theta)
            ray = shoot_ray(theta)
            ray_endpoint = self.get_path_candidate(ray)[-1]
            return (ray_endpoint-self._X2)[2]

        def delta_z_squared(cot_theta):
            return delta_z(cot_theta)**2
        
        #we minimize the cotangens of the zenith to reflect the same resolution in z to the different angles (vertical vs horizontal) 
        root1 = optimize.minimize(delta_z_squared,x0=cot(theta_direct),method='Nelder-Mead',options={'xatol':self.__xtol**2,'fatol':self.__ztol**2})
        t2 = time.time()
        if root1.success:# and np.sqrt(root1.fun) < 1e-2*units.meter:
            #if root1.fun > self.__ztol**2: print(root1,'\n',20*'#')
            theta1 = arccot(root1.x)
            detected_theta.append(theta1)
            detected_rays.append(shoot_ray(theta1))

            theta_min = theta1 - res_angle
            theta_plus = theta1 + res_angle
            delta_z_min = delta_z(cot(theta_min))
            delta_z_plus = delta_z(cot(theta_plus))
            delta_z_vertical = delta_z(cot(res_angle))
            delta_z_direct = delta_z(cot(theta_direct))
            
            def find_second_root_proof_of_principle(theta_a,theta_b):
                try:
                    cotll = cot(theta_a)
                    cotlu = cot(theta_b)
                    if cotlu > cotll:
                        bounds = scipy.optimize.Bounds(lb=cotll, ub=cotlu, keep_feasible=False)
                    else:
                        bounds = scipy.optimize.Bounds(lb=cotlu, ub=cotll, keep_feasible=False)
                    root2 = optimize.minimize(delta_z_squared,x0=cot((theta_a+theta_b)/2),bounds=bounds,options={'xatol':self.__xtol**2,'fatol':self.__ztol**2},method='Nelder-Mead')
                    theta2 = arccot(root2.x)
                    if(np.round(theta2, 2) not in np.round(detected_theta, 2)):
                        detected_theta.append(theta2)
                        detected_rays.append(shoot_ray(theta2))
                    return root2
                except RuntimeError:
                    pass


            if np.sign(delta_z_min) != np.sign(delta_z_vertical):
                root2 = find_second_root_proof_of_principle(theta_a = theta_min, theta_b = res_angle)  
            elif np.sign(delta_z_plus) != np.sign(delta_z_direct):
                root2 = find_second_root_proof_of_principle(theta_a = theta_plus, theta_b = theta_direct)

        
        self._rays = detected_rays
        self._results = [{'reflection':0,'reflection_case':1} for ray in detected_rays]
        self.__used_method = 'minimizer'


    def set_solutions(self,raytracing_results):
        """
        Read an already calculated raytracing solution from the input array

        Parameters
        ----------
        raytracing_results: dict
            The dictionary containing the raytracing solution.
        """
        results = []
        rays = []
        for iS in range(len(raytracing_results['ray_tracing_solution_type'])):
            results.append({'type' : raytracing_results['ray_tracing_solution_type'][iS],
                            'reflection' : raytracing_results['ray_tracing_reflection'][iS],
                            'reflection_case' : raytracing_results['ray_tracing_reflection_case'][iS]
                            })
            launch_vector = raytracing_results['launch_vector'][iS]
            ##use launch vector to contruct the candidate again
            rays.append(None)

        self._results = results
        self._rays = rays


    def find_solutions(self):
        """
        find all solutions between X1 and X2
        """
        results = []
        rays_results = []
        
        if self._config['propagation']['radiopropa']['mode'] == 'minimizing':
            has_reflec = (hasattr(self._medium,'reflection') or (self.__ice_model_nuradio.reflection is not None))
            if isinstance(self._medium, medium_base.IceModelSimple): #and not has_reflec:
                #TDWFSR
                self.raytracer_minimizer(n_reflections=self._n_reflections)
                results = []
                for iS in range(len(self._rays)):
                    results.append({'type':self.get_solution_type(iS), 
                                    'reflection':0,
                                    'reflection_case':1})
                self._results = results
            else:
                self.__logger.error("at the moment the RadioPropa ray tracer in minimize mode can only handle simple ice models without a reflective bottom")
          
        elif self._config['propagation']['radiopropa']['mode'] == 'hybrid minimizing':
            launch_bundles,iterative = self.raytracer_hybrid_minimizer(self._n_reflections)
            
            if not iterative:
                results = []
                for iS in range(len(self._rays)):
                    results.append({'type':self.get_solution_type(iS), 
                                    'reflection':0,
                                    'reflection_case':1})
                self._results = results

            else:
                launch_zeniths = []
                iSs = np.array(np.arange(0, len(self._rays), 1))

                for iS in iSs:
                    launch_zeniths.append(hp.cartesian_to_spherical(*(self.get_launch_vector(iS)))[0])

                mask_lower = {i: (launch_zeniths>launch_bundles[i, 0]) for i in range(len(launch_bundles))} 
                mask_upper = {i: (launch_zeniths<launch_bundles[i, 1]) for i in range(len(launch_bundles))}   
            
                for i in range(len(launch_bundles)):
                    mask = (mask_lower[i] & mask_upper[i])
                    if mask.any():
                        delta_min = np.deg2rad(90)
                        final_iS = None
                        for iS in iSs[mask]: #index of rays in the bundle
                            vector = self.get_path_candidate(self._rays[iS])[-1] - self._X2 #position of the receive vector on the sphere around the channel
                            vector_zenith = hp.cartesian_to_spherical(vector[0],vector[1],vector[2])[0]
                            receive_zenith = hp.cartesian_to_spherical(*(self.get_receive_vector(iS)))[0]
                            delta = abs(vector_zenith - receive_zenith)
                            if delta < delta_min: #select the most normal ray on the sphere in the bundle
                                final_iS = iS 
                                delta_min = delta
                        rays_results.append(self._rays[final_iS])
                        results.append({'type' : self.get_solution_type(final_iS), 
                                        'reflection' : self._results[final_iS]['reflection'],
                                        'reflection_case' : self._results[final_iS]['reflection_case']})
                
                self._rays = rays_results
                self._results = results
            

 

        else:
            launch_bundles = self.raytracer_iterative(self._n_reflections)

            launch_zeniths = []
            iSs = np.array(np.arange(0, len(self._rays), 1))

            for iS in iSs:
                launch_zeniths.append(hp.cartesian_to_spherical(*(self.get_launch_vector(iS)))[0])

            mask_lower = {i: (launch_zeniths>launch_bundles[i, 0]) for i in range(len(launch_bundles))} 
            mask_upper = {i: (launch_zeniths<launch_bundles[i, 1]) for i in range(len(launch_bundles))}   
        
            for i in range(len(launch_bundles)):
                mask = (mask_lower[i] & mask_upper[i])
                if mask.any():
                    delta_min = np.deg2rad(90)
                    final_iS = None
                    for iS in iSs[mask]: #index of rays in the bundle
                        vector = self.get_path_candidate(self._rays[iS])[-1] - self._X2 #position of the receive vector on the sphere around the channel
                        vector_zenith = hp.cartesian_to_spherical(vector[0],vector[1],vector[2])[0]
                        receive_zenith = hp.cartesian_to_spherical(*(self.get_receive_vector(iS)))[0]
                        delta = abs(vector_zenith - receive_zenith)
                        if delta < delta_min: #select the most normal ray on the sphere in the bundle
                            final_iS = iS 
                            delta_min = delta
                    rays_results.append(self._rays[final_iS])
                    results.append({'type' : self.get_solution_type(final_iS), 
                                    'reflection' : self._results[final_iS]['reflection'],
                                    'reflection_case' : self._results[final_iS]['reflection_case']})
            
            self._rays = rays_results
            self._results = results
        

        if(self.get_number_of_solutions() > self.get_number_of_raytracing_solutions()):
            self.__logger.error(f"{self.get_number_of_solutions()} were found but only {self.get_number_of_raytracing_solutions()} are allowed!")

    def get_path_candidate(self, candidate):
        """
        helper function that returns the 3D ray tracing path of a candidate

        Parameters
        ----------
        candidate: radiopropa.candidate

        Returns
        -------
        path: 2dim np.array of shape (n,3)
              x, y, z coordinates along second axis
        """
        path_x = np.array([x * (units.meter/radiopropa.meter) for x in candidate.getPathX()])
        path_y = np.array([y * (units.meter/radiopropa.meter) for y in candidate.getPathY()])
        path_z = np.array([z * (units.meter/radiopropa.meter) for z in candidate.getPathZ()])
        return np.stack([path_x, path_y, path_z], axis=1)
    
    def get_path(self, iS, n_points=None):
        """
        function that returns the 3D ray tracing path of solution iS

        Parameters
        ----------
        iS: int
            ray tracing solution
        n_points: int
                  number of points of path
                  if none, the original calculated path is returned

        Returns
        -------
        path: 2dim np.array of shape (n_points,3)
              x, y, z coordinates along second axis
        """
        n = self.get_number_of_solutions()
        if(iS >= n):
            self.__logger.error("solution number {:d} requested but only {:d} solutions exist".format(iS + 1, n))
            raise IndexError

        path = self.get_path_candidate(self._rays[iS])

        if n_points != None:
            path_x = path[:, 0]
            path_y = path[:, 1]
            path_z = path[:, 2]

            phi = hp.cartesian_to_spherical(*(self._X2 - self._X1))[1]
            path_r = path_x / np.cos(phi)

            interpol = interpolate.interp1d(path_r, path_z)
            new_path_r = np.linspace(path_r[0], path_r[-1], num=n_points)
            
            path_x = new_path_r * np.cos(phi)
            path_y = new_path_r * np.sin(phi)
            path_z = interpol(new_path_r)
            path = np.stack([path_x, path_y, path_z], axis=1)

        return path

    def get_solution_type(self, iS):
        """ 
        returns the type of the solution

        Parameters
        ----------
        iS: int
            choose for which solution to compute the solution type, 
            counting starts at zero

        Returns
        -------
        solution_type: int
            integer corresponding to the types in the dictionary solution_types
        """
        n = self.get_number_of_solutions()
        if(iS >= n):
            self.__logger.error("solution number {:d} requested but only {:d} solutions exist".format(iS + 1, n))
            raise IndexError

        pathz = self.get_path(iS)[:, 2]
        if (self._results[iS]['reflection'] != 0) or (self.get_reflection_angle(iS) != None):
            solution_type = solution_types_revert['reflected']
        elif(pathz[-1] < max(pathz)):
            solution_type = solution_types_revert['refracted']
        else:
            solution_type = solution_types_revert['direct']

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
        n = self.get_number_of_solutions()
        if(iS >= n):
            self.__logger.error("solution number {:d} requested but only {:d} solutions exist".format(iS + 1, n))
            raise IndexError

        launch_vector = np.array([self._rays[iS].getLaunchVector().x, 
                                  self._rays[iS].getLaunchVector().y, 
                                  self._rays[iS].getLaunchVector().z])
        return launch_vector/np.linalg.norm(launch_vector)

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
        n = self.get_number_of_solutions()
        if(iS >= n):
            self.__logger.error("solution number {:d} requested but only {:d} solutions exist".format(iS + 1, n))
            raise IndexError

        receive_vector = np.array([self._rays[iS].getReceiveVector().x, 
                                  self._rays[iS].getReceiveVector().y, 
                                  self._rays[iS].getReceiveVector().z])
        return receive_vector/np.linalg.norm(receive_vector)

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
        n = self.get_number_of_solutions()
        if(iS >= n):
            self.__logger.error("solution number {:d} requested but only {:d} solutions exist".format(iS + 1, n))
            raise IndexError

        reflection_angles = np.array([ra * (units.degree/radiopropa.deg) for ra in self._rays[iS].getReflectionAngles()])
        if len(reflection_angles) == 0: 
            return None
        else: 
            return np.squeeze(reflection_angles)

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
        n = self.get_number_of_solutions()
        if(iS >= n):
            self.__logger.error("solution number {:d} requested but only {:d} solutions exist".format(iS + 1, n))
            raise IndexError

        end_of_path = self.get_path_candidate(self._rays[iS])[-1] #position of the receive vector on the sphere around the channel in detector coordinates
        receive_vector = self.get_receive_vector(iS)
        
        vector = end_of_path - self._X2 #position of the receive vector on the sphere around the channel
        vector_zen,vector_az = hp.cartesian_to_spherical(vector[0], vector[1], vector[2])
        receive_zen,receive_az = hp.cartesian_to_spherical(receive_vector[0], receive_vector[1], receive_vector[2])

        path_correction_arrival_direction = abs(np.cos(receive_zen - vector_zen)) * self._sphere_sizes[-1]
        
        if abs(receive_az - vector_az) > np.deg2rad(90): 
            path_correction_overshoot = np.linalg.norm(vector[0: 2]) * abs(np.cos(receive_az - vector_az))
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
        n = self.get_number_of_solutions()
        if(iS >= n):
            self.__logger.error("solution number {:d} requested but only {:d} solutions exist".format(iS + 1, n))
            raise IndexError

        refrac_index = self._medium.get_index_of_refraction(self._X2)
        return self.get_correction_path_length(iS) / ((scipy.constants.c*units.meter/units.second)/refrac_index)


    def get_path_length(self, iS):
        """
        calculates the path length of solution iS

        Parameters
        ----------
        iS: int
            choose for which solution to compute the path length, 
            counting starts at zero

        Returns
        -------
        distance: float
            distance from X1 to X2 along the ray path
        """
        n = self.get_number_of_solutions()
        if(iS >= n):
            self.__logger.error("solution number {:d} requested but only {:d} solutions exist".format(iS + 1, n))
            raise IndexError

        path_length = self._rays[iS].getTrajectoryLength() * (units.meter/radiopropa.meter)
        if self.__used_method == 'iterator':
            path_length += self.get_correction_path_length(iS)
        return path_length

    def get_travel_time(self, iS):
        """
        calculates the travel time of solution iS

        Parameters
        ----------
        iS: int
            choose for which solution to compute the travel time, 
            counting starts at zero

        Returns
        -------
        time: float
            travel time
        """
        n = self.get_number_of_solutions()
        if(iS >= n):
            self.__logger.error("solution number {:d} requested but only {:d} solutions exist".format(iS + 1, n))
            raise IndexError

        travel_time = self._rays[iS].getPropagationTime() * (units.second/radiopropa.second)
        if self.__used_method == 'iterator':
            travel_time += self.get_correction_travel_time(iS)
        return travel_time


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
        n = self.get_number_of_solutions()
        if(iS >= n):
            self.__logger.error("solution number {:d} requested but only {:d} solutions exist".format(iS + 1, n))
            raise IndexError

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

    def get_focusing(self, iS, dz=-1. * units.cm, limit=2.):
        """
        calculate the focusing effect in the medium
        
        Parameters
        ----------
        iS: int
            choose for which solution to compute the launch vector, counting
            starts at zero
        dz: float
            the infinitesimal change of the depth of the receiver, 1cm by default
        
        Returns
        -------
        focusing: a float
            gain of the signal at the receiver due to the focusing effect
        """
        recVec = self.get_receive_vector(iS)
        recVec = -1.0 * recVec
        recAng = np.arccos(recVec[2] / np.sqrt(recVec[0]**2 + recVec[1]**2 + recVec[2] **2))
        lauVec = self.get_launch_vector(iS)
        lauAng = np.arccos(lauVec[2] / np.sqrt(lauVec[0] ** 2 + lauVec[1] ** 2 + lauVec[2] ** 2))        
        distance = self.get_path_length(iS)
       
        vetPos = copy.copy(self._X1)
        recPos = copy.copy(self._X2)
        recPos1 = np.array([self._X2[0], self._X2[1], self._X2[2] + dz])
        if not hasattr(self, "_r1"):
            self._r1 = radiopropa_ray_tracing(self._medium, self._attenuation_model, logging.WARNING, self._n_frequencies_integration, self._n_reflections, config = self._config)
        self._r1.set_shower_axis(self._shower_axis)
        self._r1.set_start_and_end_point(vetPos, recPos1)
        self._r1.find_solutions()
        if iS < self._r1.get_number_of_solutions():
            lauVec1 = self._r1.get_launch_vector(iS)
            lauAng1 = np.arccos(lauVec1[2] / np.sqrt(lauVec1[0] ** 2 + lauVec1[1] ** 2 + lauVec1[2] ** 2))
            focusing = np.sqrt(distance / np.sin(recAng) * np.abs((lauAng1 - lauAng) / (recPos1[2] - recPos[2])))
            if(self.get_solution_type(iS) != self._r1.get_solution_type(iS)):
                self.__logger.error("solution types are not the same")
        else:
            focusing = 1.0
            self.__logger.info("too few ray tracing solutions, setting focusing factor to 1")
        self.__logger.debug(f'amplification due to focusing of solution {iS:d} = {focusing:.3f}')
        if(focusing > limit):
            self.__logger.info(f"amplification due to focusing is {focusing:.1f}x -> limiting amplification factor to {limit:.1f}x")
            focusing = limit

        # now also correct for differences in refractive index between emitter and receiver position

        n1 = self._medium.get_index_of_refraction(self._X1)  # emitter
        n2 = self._medium.get_index_of_refraction(self._X2)  # receiver
        return focusing * (n1 / n2) ** 0.5
        
        

    def apply_propagation_effects(self, efield, i_solution):
        """
        Apply propagation effects to the electric field
        Note that the 1/r weakening of the electric field is already accounted for in the signal generation

        Parameters
        ----------
        efield: ElectricField object
            The electric field that the effects should be applied to
        i_solution: int
            Index of the raytracing solution the propagation effects should be based on

        Returns
        -------------
        efield: ElectricField object
            The modified ElectricField object
        """
        spec = efield.get_frequency_spectrum()
        ## aply attenuation
        apply_attenuation = self._config['propagation']['attenuate_ice']
        if apply_attenuation:
            if self._max_detector_frequency is None:
                max_freq = np.max(efield.get_frequencies())
            else:
                max_freq = self._max_detector_frequency
            attenuation = self.get_attenuation(i_solution, efield.get_frequencies(), max_freq)
            spec *= attenuation
        
        ## apply reflections
        zenith_reflections = np.atleast_1d(self.get_reflection_angle(i_solution))
        for zenith_reflection in zenith_reflections:
            if (zenith_reflection is None):
                continue
            r_theta = NuRadioReco.utilities.geometryUtilities.get_fresnel_r_p(zenith_reflection, 
                n_2=self._medium.get_index_of_refraction(np.array([self._X2[0], self._X2[1], +1 * units.cm])), 
                n_1=self._medium.get_index_of_refraction(np.array([self._X2[0], self._X2[1], -1 * units.cm])))
            r_phi = NuRadioReco.utilities.geometryUtilities.get_fresnel_r_s(zenith_reflection, 
                n_2=self._medium.get_index_of_refraction(np.array([self._X2[0], self._X2[1], +1 * units.cm])),
                n_1=self._medium.get_index_of_refraction(np.array([self._X2[0], self._X2[1], -1 * units.cm])))
            efield[efp.reflection_coefficient_theta] = r_theta
            efield[efp.reflection_coefficient_phi] = r_phi

            spec[1] *= r_theta
            spec[2] *= r_phi
            self.__logger.debug(
                "ray hits the surface at an angle {:.2f}deg -> reflection coefficient is r_theta = {:.2f}, r_phi = {:.2f}".format(
                    zenith_reflection / units.deg, r_theta, r_phi))

        i_reflections = self.get_results()[i_solution]['reflection']
        if (i_reflections > 0):  # take into account possible bottom reflections
            # each reflection lowers the amplitude by the reflection coefficient and introduces a phase shift
            reflection_coefficient = self._medium.reflection_coefficient ** i_reflections
            phase_shift = (i_reflections * self._medium.reflection_phase_shift) % (2 * np.pi)
            # we assume that both efield components are equally affected
            spec[1] *= reflection_coefficient * np.exp(1j * phase_shift)
            spec[2] *= reflection_coefficient * np.exp(1j * phase_shift)
            self.__logger.debug(
                f"ray is reflecting {i_reflections:d} times at the bottom -> reducing the signal by a factor of {reflection_coefficient:.2f}")


        ## apply focussing effect
        if self._config['propagation']['focusing']:
            focusing = self.get_focusing(i_solution, limit=float(self._config['propagation']['focusing_limit']))
            spec[1:] *= focusing

        efield.set_frequency_spectrum(spec, efield.get_sampling_rate())
        return efield


    def get_output_parameters(self):
        """
        Returns a list with information about parameters to include in the output data structure that are specific
        to this raytracer

        ! be sure that the first entry is specific to your raytracer !

        Returns
        -------
        list with entries of form [{'name': str, 'ndim': int}]
            ! be sure that the first entry is specific to your raytracer !
            'name': Name of the new parameter to include in the data structure
            'ndim': Dimension of the data structure for the parameter
        """
        return [
            {'name': 'sphere_sizes','ndim':len(self._sphere_sizes)},
            {'name': 'launch_vector', 'ndim': 3},
            {'name': 'focusing_factor', 'ndim': 1},
            {'name': 'ray_tracing_reflection', 'ndim': 1},
            {'name': 'ray_tracing_reflection_case', 'ndim': 1},
            {'name': 'ray_tracing_solution_type', 'ndim': 1}
        ]

    def get_raytracing_output(self, i_solution):
        """
        Write parameters that are specific to this raytracer into the output data.

        Parameters
        ----------
        i_solution: int
            The index of the raytracing solution

        Returns
        -------
        dictionary with the keys matching the parameter names specified in get_output_parameters and the values being
        the results from the raytracing
        """
        if self._config['propagation']['focusing']:    
            focusing = self.get_focusing(i_solution, limit=float(self._config['propagation']['focusing_limit']))
        else: 
            focusing = 1
        output_dict = {
            'sphere_sizes': self._sphere_sizes,
            'launch_vector': self.get_launch_vector(i_solution),
            'focusing_factor': focusing,
            'ray_tracing_reflection': self.get_results()[i_solution]['reflection'],
            'ray_tracing_reflection_case': self.get_results()[i_solution]['reflection_case'],
            'ray_tracing_solution_type': self.get_solution_type(i_solution)            
        }
        return output_dict

    def set_config(self, config):
        """
        Function to change the configuration file used by the raytracer

        Parameters
        ----------
        config: dict
            The new configuration settings
        """
        if config == None:
            config = dict()
            config['propagation'] = dict(
                attenuate_ice = True,
                focusing_limit = 2,
                focusing = False,
                radiopropa = dict(
                    mode = 'iterative',
                    iter_steps_channel = [25., 2., .5], #unit is meter
                    iter_steps_zenith = [.5, .05, .005], #unit is degree
                    auto_step_size = False,
                    max_traj_length = 10000) #unit is meter
            )
            config['speedup'] = dict(
                delta_C_cut = 40 * units.degree
            )
        super().set_config(config)

        self._cut_viewing_angle = self._config['speedup']['delta_C_cut'] * units.radian
        self._max_traj_length = self._config['propagation']['radiopropa']['max_traj_length'] * units.meter
        self.set_iterative_sphere_sizes(np.array(self._config['propagation']['radiopropa']['iter_steps_channel']) * units.meter)
        self._auto_step = self._config['propagation']['radiopropa']['auto_step_size']
        self.set_iterative_step_sizes(np.array(self._config['propagation']['radiopropa']['iter_steps_zenith']) * units.degree)

    ## helper functions
    def delta_theta_direct(self, dz):
        v = (self._X2 - self._X1)
        u = copy.deepcopy(v)
        u[2] = 0
        rho = np.linalg.norm(u)
        return dz * rho / ((self._X1[2] - self._X2[2])**2 + rho**2) * units.radian

    def delta_theta_bottom(self, dz, z_refl):
        v = (self._X2 - self._X1)
        u = copy.deepcopy(v)
        u[2] = 0
        rho = np.linalg.norm(u)
        return dz * rho / ((self._X1[2] + self._X2[2] - 2*z_refl)**2 + rho**2) * units.radian

    def delta_theta_reflective(self, dz, n_bottom_reflections):
        v = (self._X2 - self._X1)
        u = copy.deepcopy(v)
        u[2] = 0
        rho = np.linalg.norm(u)
        ice_thickness = self._medium.z_air_boundary - self._medium.z_bottom
        if n_bottom_reflections > 0:
            if self.medium.reflection is None:
                self.__logger.error("a solution for {:d} reflection(s) off the bottom reflective layer is requested,"
                                    +"but ice model does not specify a reflective layer".format(n_bottom_reflections))
                raise AttributeError("a solution for {:d} reflection(s) off the bottom reflective layer is requested,"
                                    +"but ice model does not specify a reflective layer".format(n_bottom_reflections))
            else:
                ice_thickness = self._medium.z_air_boundary - self._medium.reflection
        return -dz * rho / ((self._X1[2] + self._X2[2] + 2*n_bottom_reflections*ice_thickness)**2 + rho**2) * units.radian
