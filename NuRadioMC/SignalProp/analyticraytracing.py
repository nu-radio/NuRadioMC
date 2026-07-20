"""
Wrapper for different implementations of a 2D analytic ray tracer to get ray tracing solutions in 3D for two arbitrary points x1 and x2.
The 2D ray tracer is chosen depending on the provided ice model and can either be the single layer analytic raytracer (when IceModelSimple is given)
or the new multilayer version of it (when a medium of type IceModelExpLayers is used).
Implementations for the single layer 2D ray tracer include a
CPP version, a python version with numba and a python version without numba, the multilayer version is currently limited to either python with numba and python without numba.
The CPP version is the default if available, otherwise the python version with numba is used if available,
otherwise the python version without numba is used.

Implementations are in NuRadioMC/SignalProp/AnalyticRayTracing/
"""

from NuRadioReco.utilities import units, geometryUtilities
from NuRadioMC.utilities import medium
from NuRadioMC.utilities.medium_base import IceModelSimple, IceModelExpLayers
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework import base_trace
from NuRadioMC.SignalProp.propagation_base_class import ray_tracing_base
from NuRadioMC.SignalProp.AnalyticRayTracingImpl.single_layer_analytic_raytracer import (
    ray_tracing_2D, SPEED_OF_LIGHT, N_AIR, _get_zenith, _n
)
from NuRadioMC.SignalProp.AnalyticRayTracing.MultilayerAnalyticRayTracing.multilayeranalyticraytracing import multi_layer_ray_tracing_2D
from NuRadioMC.utilities.birefringence import get_effective_index_birefringence, get_polarization_birefringence

import numpy as np
import copy
import logging
logger = logging.getLogger("NuRadioMC.analytic_ray_tracing")

class ray_tracing(ray_tracing_base):
    """
    Utility class (wrapper around the 2D analytic ray tracing code) to get
    ray tracing solutions in 3D for two arbitrary points x1 and x2
    """

    def __init__(self, medium, attenuation_model=None, log_level=logging.NOTSET,
                 n_frequencies_integration=None, n_reflections=None, config=None,
                 detector=None, ray_tracing_2D_kwards={},
                 use_cpp=None, compile_numba=None):
        """
        Class initilization

        Parameters
        ----------
        medium: medium class
            Class describing the index-of-refraction profile

        attenuation_model: string
            Signal attenuation model
            (default: None -> 'SP1' (see ``ray_tracing_base._set_arguments``))

        log_name:  string
            Name under which things should be logged

        log_level: logging object
            Specify the log level of the ray tracing class

            * logging.ERROR
            * logging.WARNING
            * logging.INFO
            * logging.DEBUG

            default is NOTSET (global control)

        n_frequencies_integration: int
            The number of frequencies for which the frequency dependent attenuation
            length is being calculated. The attenuation length for all other frequencies
            is obtained via linear interpolation.
            (default: None -> 100 (see ``ray_tracing_base._set_arguments``))

        n_reflections: int
            In case of a medium with a reflective layer at the bottom, how many reflections should be considered
            (default: None -> 0 (see ``ray_tracing_base._set_arguments``))

        config: dict
            a dictionary with the optional config settings. If None, the config is intialized with default values,
            which is needed to avoid any "key not available" errors. The default settings are

                * self._config = {'propagation': {}}
                * self._config['propagation']['attenuate_ice'] = True
                * self._config['propagation']['focusing_limit'] = 2
                * self._config['propagation']['focusing'] = False
                * self._config['propagation']['birefringence'] = False

        detector: detector object

        ray_tracing_2D_kwards: dict
            Additional arguments which are passed to ray_tracing_2D

        use_cpp: bool (default: None)
            If True, use the CPP implementation of the ray tracer; if explicitly set to True but
            the CPP version is not available, a RuntimeError is raised.
            If None, the CPP version is used whenever it is available.

        compile_numba: bool (default: None)
            If True, numba-compile the standalone python functions used as a fallback when not
            using the CPP backend. Only relevant if `use_cpp` is (or resolves to) False.
            If None, numba is used whenever it is available.
        """
        self.__logger = logging.getLogger('NuRadioMC.ray_tracing')
        self.__logger.setLevel(log_level)

        if not isinstance(medium, (IceModelSimple, IceModelExpLayers)):
            self.__logger.error(
                "The analytic raytracer can only handle ice models of the type 'IceModelSimple' "
                "(single layer) or 'IceModelExpLayers' (multilayer) (see NuRadioMC.utilities.medium)!")
            raise TypeError(
                "The analytic raytracer can only handle ice models of the type 'IceModelSimple' "
                "(single layer) or 'IceModelExpLayers' (multilayer)")

        super().__init__(
            medium=medium,
            attenuation_model=attenuation_model,
            log_level=log_level,
            n_frequencies_integration=n_frequencies_integration,
            n_reflections=n_reflections,
            config=config,
            detector=detector)

        self.set_config(config=config)

        # `ray_tracing_2D`/`multi_layer_ray_tracing_2D` already resolve use_cpp/compile_numba
        # (from None defaults and availability) and log the outcome; we just mirror the result
        # here rather than resolving (and logging) it a second time.
        if isinstance(medium, IceModelSimple):
            self.__logger.status("IceModelSimple was provided: using the single layer analytic ray tracer as the 2D raytracing module.")
            self._r2d = ray_tracing_2D(
                self._medium, self._attenuation_model, log_level=log_level,
                n_frequencies_integration=self._n_frequencies_integration,
                **ray_tracing_2D_kwards, use_cpp=use_cpp, compile_numba=compile_numba)

        else:
            self.__logger.status("IceModelExpLayers was provided: using the multilayer analytic ray tracer as the 2D raytracing module.")
            self._r2d = multi_layer_ray_tracing_2D(
                self._medium, self._attenuation_model, log_level=log_level,
                n_frequencies_integration=self._n_frequencies_integration,
                **ray_tracing_2D_kwards, use_cpp=use_cpp, compile_numba=compile_numba)

        self.use_cpp = self._r2d.use_cpp
        self.compile_numba = self._r2d.compile_numba

        # As long as we use horizontal-translational invariant raytracing/ice models (2d)
        # this should be fine. _n(0) is also used in _get_delta_y for air-ice raytracing
        self.n_at_surface = _n(0, self._medium.n_ice, self._medium.delta_n, self._medium.z_0)

        # Some consitency checks...

        # Check that `self.n_at_surface` is reasonably large to avoid a bug where the index of air
        # is returned
        if self.n_at_surface < 1.1:
            raise ValueError(f"Calculated index of refraction for ice at the ice-air boundary is {self.n_at_surface} which is to small.")

        # `z_air_boundary`/`z_shift` describe the single, flat ice-air interface `IceModelSimple`
        # assumes; `IceModelExpLayers` has no such single interface (its `z_air_boundary`/`z_shift`
        # properties are only kept for backwards compatibility and don't carry the same meaning),
        # so these checks don't apply to the multilayer raytracer.
        if self._medium.z_air_boundary != 0:
            raise ValueError(f"The configured ice model has `z_air_boundary != 0`. This is not supported by this raytracer!")

        if hasattr(self._medium, "z_shift") and self._medium.z_shift != 0:
            raise ValueError(f"The configured ice model has `z_shift != 0`. This is not supported by this raytracer!")

        self._swap = None
        self._dPhi = None
        self._R = None
        self._x1 = None
        self._x2 = None
        # caches for the attenuation and focusing factors. They are only valid for the current
        # geometry and are invalidated whenever the solutions are reset. This avoids recalculating
        # these (expensive) quantities if several showers/emitters are simulated at the same position
        self._cache_attenuation = {}
        self._cache_focusing = {}


    def reset_solutions(self):
        """
        Resets the raytracing solutions back to None. This is useful to do when changing the start and end
        points in order to not accidentally use results from previous raytracings.

        """
        super().reset_solutions()
        self._x1 = None
        self._x2 = None
        self._swap = None
        self._dPhi = None
        self._R = None
        self._cache_attenuation = {}
        self._cache_focusing = {}

    def set_start_and_end_point(self, x1, x2, autoswap=True):
        """
        Set the start and end points of the raytracing

        If the start and end points are identical to those of the previous ray tracing,
        the existing solutions are kept and `find_solutions` will not recalculate them.
        Call `reset_solutions` before this function to force a recalculation.

        Parameters
        ----------
        x1: 3dim np.array
            Start point of the ray
        x2: 3dim np.array
            Stop point of the ray

        Returns
        -------
        geometry_changed: bool
            False if the start and end points are unchanged with respect to the previous
            ray tracing (in which case the existing solutions are kept), True otherwise.
        """
        if not super().set_start_and_end_point(x1, x2):
            return False

        self._swap = False
        if autoswap is True:
            if(self._X2[2] < self._X1[2]):
                self._swap = True
                self.__logger.debug('swap = True')
                self._X2 = np.array(x1, dtype=float)
                self._X1 = np.array(x2, dtype=float)

        dX = self._X2 - self._X1
        self._dPhi = -np.arctan2(dX[1], dX[0])
        c, s = np.cos(self._dPhi), np.sin(self._dPhi)
        self._R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
        X1r = self._X1
        X2r = np.dot(self._R, self._X2 - self._X1) + self._X1
        self.__logger.debug("X1 = {}, X2 = {}".format(self._X1, self._X2))
        self.__logger.debug('dphi = {:.1f}'.format(self._dPhi / units.deg))
        self.__logger.debug("X2 - X1 = {}, X1r = {}, X2r = {}".format(self._X2 - self._X1, X1r, X2r))
        self._x1 = np.array([X1r[0], X1r[2]])
        self._x2 = np.array([X2r[0], X2r[2]])
        self.__logger.debug("2D points {} {}".format(self._x1, self._x2))

    def set_start_and_end_point_no_swap(self, x1, x2):
        """
        Set the start and end points of the raytracing without automatically swapping x1 and x2 if z2 > z1.

        Parameters
        ----------
        x1: 3dim np.array
            start point of the ray
        x2: 3dim np.array
            stop point of the ray
        """


        super().set_start_and_end_point(x1, x2)

        dX = self._X2 - self._X1
        self._dPhi = -np.arctan2(dX[1], dX[0])
        c, s = np.cos(self._dPhi), np.sin(self._dPhi)
        self._R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
        X1r = self._X1
        X2r = np.dot(self._R, self._X2 - self._X1) + self._X1
        self.__logger.debug("X1 = %s, X2 = %s", self._X1, self._X2)
        self.__logger.debug("dphi = %.1f", self._dPhi / units.deg)
        self.__logger.debug("X2 - X1 = %s, X1r = %s, X2r = %s", dX, X1r, X2r)
        self._x1 = np.array([X1r[0], X1r[2]])
        self._x2 = np.array([X2r[0], X2r[2]])
        self.__logger.debug("2D points %s %s", self._x1, self._x2)

        return True

    def set_solution(self, raytracing_results):
        """
        Read an already calculated raytracing solution from the input array

        Parameters
        ----------
        raytracing_results: dict
            The dictionary containing the raytracing solution.
        """
        results = []
        C0s = raytracing_results['ray_tracing_C0']
        for i in range(len(C0s)):
            if(not np.isnan(C0s[i])):
                if 'ray_tracing_reflection' in raytracing_results.keys():  # for backward compatibility: Check if reflection layer information exists in data file
                    reflection = raytracing_results['ray_tracing_reflection'][i]
                    reflection_case = raytracing_results['ray_tracing_reflection_case'][i]
                else:
                    reflection = 0
                    reflection_case = 0
                results.append({'type': raytracing_results['ray_tracing_solution_type'][i],
                                'C0': C0s[i],
                                'C1': raytracing_results['ray_tracing_C1'][i],
                                'reflection': reflection,
                                'reflection_case': reflection_case})
        self._results = results

    def find_solutions(self):
        """
        Find all solutions between x1 and x2

        If solutions for the current start and end points already exist (i.e., the geometry
        did not change since the last ray tracing or the solutions were set with `set_solution`),
        they are kept and not recalculated. Call `reset_solutions` first to force a recalculation.
        """
        if self._results is not None:
            self.__logger.debug("solutions for the current geometry already exist, skipping ray tracing")
            return

        self._results = self._r2d.find_solutions(self._x1, self._x2)
        for i in range(self._n_reflections):
            for j in range(2):
                self._results.extend(self._r2d.find_solutions(self._x1, self._x2, reflection=i + 1, reflection_case=j + 1))

        # check if not too many solutions were found (the same solution can potentially found twice because of numerical imprecision)
        if(self.get_number_of_solutions() > self.get_number_of_raytracing_solutions()):
            self.__logger.warning(f"[x1 {self._x1}, x2 {self._x2}] {self.get_number_of_solutions()} were found but only {self.get_number_of_raytracing_solutions()} are allowed!")
            #self._results = []

    def get_solution_type(self, iS):
        """ Returns the type of the solution

        Parameters
        ----------
        iS: int
            Choose for which solution to compute the launch vector, counting
            starts at zero

        Returns
        -------
        solution_type: int
            integer corresponding to the types in the dictionary solution_types
        """
        return self._r2d.determine_solution_type(self._x1, self._x2, self._results[iS]['C0'])

    def get_path(self, iS, n_points=1000):

        n = self.get_number_of_solutions()
        if(iS >= n):
            self.__logger.error(f"[x1 {self._x1}, x2 {self._x2}] solution number {iS + 1} requested but only {n} solutions exist")
            raise IndexError
        result = self._results[iS]
        xx, zz = self._r2d.get_path_reflections(self._x1, self._x2, result['C0'], n_points=n_points,
                                                 reflection=result['reflection'],
                                                 reflection_case=result['reflection_case'])
        path_2d = np.array([xx, np.zeros_like(xx), zz]).T

        dP = path_2d - np.array([self._X1[0], 0, self._X1[2]])
        MM = np.matmul(self._R.T, dP.T)
        path = MM.T + self._X1
        return path

    def get_pulse_propagation_birefringence(self, pulse, samp_rate, i_solution, bire_model = 'southpole_A'):

        """
        Function for the time trace propagation according to the polarization change due to birefringence.
        The trace propagation is explained in this paper: https://link.springer.com/article/10.1140/epjc/s10052-023-11238-y

        Parameters
        ----------
        pulse: np.ndarray
            3d array with the frequency spectrum of np.array([eR, eTheta, ePhi]), usually provided by the apply_propagation_effects function
        samp_rate: float
            Sampling rate of the time traces
        i_solution: int
            Choose which ray-tracing solution should be propagated
        bire_model: string
            Choose the interpolation to fit the measured refractive index data
            options include (A, B, C, D, E) description can be found under: NuRadioMC/NuRadioMC/utilities/birefringence_models/model_description

        Returns
        -------

        final pulse: numpy.array([eR, eTheta, ePhi])
            [0] - eR        - final frequency spectrum of the radial component - not altered by the function
            [1] - eTheta    - final frequency spectrum of the theta component
            [2] - ePhi      - final frequency spectrum of the phi component
        """

        t_fast = base_trace.BaseTrace()

        ice_n = self._medium
        ice_birefringence = medium.get_ice_model('birefringence_medium')
        ice_birefringence.__init__(bire_model)

        acc = int(self.get_path_length(i_solution) / units.m)
        path = self.get_path(i_solution, n_points=acc)

        if 'angle_to_iceflow' in self._config['propagation']:
            rotation_angle = self._config['propagation']['angle_to_iceflow'] * units.deg
            rot = np.matrix([[np.cos(rotation_angle), -np.sin(rotation_angle)], [np.sin(rotation_angle), np.cos(rotation_angle)]])
            path[:, :2] = np.swapaxes(np.matmul(rot, np.swapaxes(path[:, :2],0,1)),0,1)

        for i in range(acc - 1):

            refractive_index = ice_n.get_index_of_refraction(path[i])
            refractive_index_birefringence = ice_birefringence.get_birefringence_index_of_refraction(path[i])

            nx, ny, nz = refractive_index + refractive_index_birefringence - 1.78
            dD = path[i + 1] - path[i]

            direction = np.array(dD)
            len_diff = np.linalg.norm(direction)
            direction = direction / len_diff

            N_effective = get_effective_index_birefringence(direction, nx, ny, nz)
            sky_polarization = get_polarization_birefringence(N_effective[0], N_effective[1], direction, nx, ny, nz, logger=self.__logger)

            t_0, t_1 = len_diff * N_effective / (SPEED_OF_LIGHT * units.m / units.ns)

            a, b = sky_polarization[0, 1:]
            c, d = sky_polarization[1, 1:]

            if np.isclose(a * d - b * c, 0) or np.isnan([a, b, c, d]).any():
                self.__logger.warning("warning: Polarization vectors similar, R-matrix not invertible, iteration" + str(i))
                continue

            R = np.matrix([[a, b], [c, d]])

            birefringent_base = R * pulse[1:]

            t_fast.set_frequency_spectrum(birefringent_base[1], sampling_rate=samp_rate)
            t_fast.apply_time_shift(t_1 - t_0)
            birefringent_base[1] = t_fast.get_frequency_spectrum()

            Rtransp = np.matrix.transpose(R)
            pulse[1:]  = Rtransp * birefringent_base

        return pulse

    def get_path_properties_birefringence(self, i_solution, bire_model = 'southpole_A'):

        """
        Function to extract important information about the birefringent propagation along the path.
        The important properties include effective refractive indices, polarization eigenvectors, and incremental time delays

        Parameters
        ----------

        i_solution: int
            Choose which ray-tracing solution should be propagated
        bire_model: string
            Choose the interpolation to fit the measured refractive index data
            options include (A, B, C, D, E) description can be found under: NuRadioMC/NuRadioMC/utilities/birefringence_models/model_description

        Returns
        -------

        path_properties: dict
            a dictionary containing the following keys:

            * 'path': np.ndarray - propagation path in x, y, z with the same granularity as the nirefringent propagation
            * 'nominal_refractive_index': np.ndarray - nominal refractive index if only density effects were taken into account
            * 'refractive_index_x': np.ndarray - refractive index for the x-direction
            * 'refractive_index_y': np.ndarray - refractive index for the y-direction
            * 'refractive_index_z': np.ndarray - refractive index for the z-direction
            * 'first_refractive_index': np.ndarray - effective refractive index of the first birefringent state along the full path
            * 'second_refractive_index': np.ndarray - effective refractive index of the second birefringent state along the full path
            * 'first_polarization_vector': np.ndarray - polarization vector of the first birefringent state in spherical coordinates along the full path
            * 'second_polarization_vector': np.ndarray - polarization vector of the second birefringent state in spherical coordinates along the full path
            * 'first_time_delay': np.ndarray - incremental time delays of the first birefringent state along the full path
            * 'second_time_delay': np.ndarray - incremental time delays of the second birefringent state along the full path

        """

        ice_n = self._medium
        ice_birefringence = medium.get_ice_model('birefringence_medium')
        ice_birefringence.__init__(bire_model)

        acc = int(self.get_path_length(i_solution) / units.m)
        path = self.get_path(i_solution, n_points=acc)

        if 'angle_to_iceflow' in self._config['propagation']:
            rotation_angle = self._config['propagation']['angle_to_iceflow'] * units.deg
            rot = np.matrix([[np.cos(rotation_angle), -np.sin(rotation_angle)], [np.sin(rotation_angle), np.cos(rotation_angle)]])
            path[:, :2] = np.swapaxes(np.matmul(rot, np.swapaxes(path[:, :2],0,1)),0,1)

        n_nominal = np.zeros(acc - 1)

        N1 = np.zeros(acc - 1)
        N2 = np.zeros(acc - 1)

        Nx = np.zeros(acc - 1)
        Ny = np.zeros(acc - 1)
        Nz = np.zeros(acc - 1)

        P1 = np.zeros((acc - 1, 3))
        P2 = np.zeros((acc - 1, 3))

        T1 = np.zeros(acc - 1)
        T2 = np.zeros(acc - 1)

        for i in range(acc - 1):

            refractive_index = ice_n.get_index_of_refraction(path[i])
            refractive_index_birefringence = ice_birefringence.get_birefringence_index_of_refraction(path[i])

            nx, ny, nz = refractive_index + refractive_index_birefringence - 1.78
            dD = path[i + 1] - path[i]

            direction = np.array(dD)
            len_diff = np.linalg.norm(direction)
            direction = direction / len_diff

            N_effective = get_effective_index_birefringence(direction, nx, ny, nz)
            sky_polarization = get_polarization_birefringence(N_effective[0], N_effective[1], direction, nx, ny, nz, logger=self.__logger)

            t_0, t_1 = len_diff * N_effective / (SPEED_OF_LIGHT * units.m / units.ns)
            n_nominal[i] = refractive_index

            Nx[i] = refractive_index_birefringence[0]
            Ny[i] = refractive_index_birefringence[1]
            Nz[i] = refractive_index_birefringence[2]

            N1[i] = N_effective[0]
            N2[i] = N_effective[1]

            P1[i] = sky_polarization[0]
            P2[i] = sky_polarization[1]

            T1[i] = t_0
            T2[i] = t_1

        path_properties = {}

        path_properties['path'] = path[1:]
        path_properties['nominal_refractive_index'] = n_nominal

        path_properties['refractive_index_x'] = Nx
        path_properties['refractive_index_y'] = Ny
        path_properties['refractive_index_z'] = Nz

        path_properties['first_refractive_index'] = N1
        path_properties['second_refractive_index'] = N2

        path_properties['first_polarization_vector'] = P1
        path_properties['second_polarization_vector'] = P2

        path_properties['first_time_delay'] = T1
        path_properties['second_time_delay'] = T2

        return path_properties

    def get_launch_vector(self, iS):
        """
        Calculates the launch vector (in 3D) of solution iS

        Parameters
        ----------
        iS: int
            Choose for which solution to compute the launch vector, counting
            starts at zero

        Returns
        -------
        launch_vector: 3dim np.array
            the launch vector
        """
        n = self.get_number_of_solutions()
        if(iS >= n):
            self.__logger.error("solution number {:d} requested but only {:d} solutions exist".format(iS + 1, n))
            raise IndexError

        result = self._results[iS]
        alpha = self._r2d.get_launch_angle(self._x1, result['C0'], reflection=result['reflection'],
                                            reflection_case=result['reflection_case'])
        launch_vector_2d = np.array([np.sin(alpha), 0, np.cos(alpha)])
        if self._swap:
            alpha = self._r2d.get_receive_angle(self._x1, self._x2, result['C0'],
                                                 reflection=result['reflection'],
                                                 reflection_case=result['reflection_case'])
            launch_vector_2d = np.array([-np.sin(alpha), 0, np.cos(alpha)])
        self.__logger.debug(self._R.T)
        launch_vector = np.dot(self._R.T, launch_vector_2d)
        return launch_vector

    def get_receive_vector(self, iS):
        """
        Calculates the receive vector (in 3D) of solution iS

        Parameters
        ----------
        iS: int
            Choose for which solution to compute the launch vector, counting
            starts at zero

        Returns
        -------
        receive_vector: 3dim np.array
            the receive vector
        """
        n = self.get_number_of_solutions()
        if(iS >= n):
            self.__logger.error("solution number {:d} requested but only {:d} solutions exist".format(iS + 1, n))
            raise IndexError

        result = self._results[iS]
        alpha = self._r2d.get_receive_angle(self._x1, self._x2, result['C0'],
                                             reflection=result['reflection'],
                                             reflection_case=result['reflection_case'])
        receive_vector_2d = np.array([-np.sin(alpha), 0, np.cos(alpha)])
        if self._swap:
            alpha = self._r2d.get_launch_angle(self._x1, result['C0'],
                                                reflection=result['reflection'],
                                                reflection_case=result['reflection_case'])
            receive_vector_2d = np.array([np.sin(alpha), 0, np.cos(alpha)])
        receive_vector = np.dot(self._R.T, receive_vector_2d)
        return receive_vector

    def get_reflection_angle(self, iS):
        """
        Calculates the angle of reflection at the surface (in case of a reflected ray)

        Parameters
        ----------
        iS: int
            Choose for which solution to compute the launch vector, counting
            starts at zero

        Returns
        -------
        reflection_angle: float or None
            the reflection angle (for reflected rays) or None for direct and refracted rays
        """
        n = self.get_number_of_solutions()
        if(iS >= n):
            self.__logger.error("solution number {:d} requested but only {:d} solutions exist".format(iS + 1, n))
            raise IndexError

        result = self._results[iS]
        return self._r2d.get_reflection_angle(self._x1, self._x2, result['C0'],
                                               reflection=result['reflection'], reflection_case=result['reflection_case'])

    def get_fresnel_coefficients(self, iS):
        """
        Calculates the fresnel coefficients for all interactions with the ice-air surface of solution iS

        For rays that are reflected off the ice-air surface (emitter and receiver in the ice),
        the fresnel reflection coefficients are calculated. If the ray crosses the ice-air
        interface (emitter or receiver in the air), the fresnel transmission coefficients
        are calculated. Reflections at an in-ice reflective (bottom) layer are not included here,
        they are treated separately (see `apply_propagation_effects`).

        Parameters
        ----------
        iS: int
            Choose for which solution to compute the fresnel coefficients, counting
            starts at zero

        Returns
        -------
        fresnel_coefficients: list of dict
            One dictionary per interaction with the ice-air surface (empty list for rays
            that never reach the surface). Each dictionary contains:

            * "zenith": zenith angle under which the ray hits the surface
            * "case": "reflection" (in-ice reflection off the surface) or
              "transmission" (ice-to-air / air-to-ice)
            * "theta": fresnel coefficient for the eTheta (p-polarization) component
            * "phi": fresnel coefficient for the ePhi (s-polarization) component
        """

        fresnel_coefficients = []
        # lets handle the general case of multiple reflections off the ice-air surface
        # Multiple relfections are possible if a reflective bottom layer exists.
        for zenith_reflection in np.atleast_1d(self.get_reflection_angle(iS)):

            # skip all ray segments where no interaction with the surface happens
            if zenith_reflection is None:
                continue

            # we need to treat the case of air to ice/ice to air propagation separately:
            if self._x2[1] > 0:
                # air/ice propagation
                if not self._swap:
                    # ice to air case
                    t_theta = geometryUtilities.get_fresnel_t_p(zenith_reflection, n_2=N_AIR, n_1=self.n_at_surface)
                    t_phi = geometryUtilities.get_fresnel_t_s(zenith_reflection, n_2=N_AIR, n_1=self.n_at_surface)
                    self.__logger.info(f"propagating from ice to air: transmission coefficient is {t_theta:.2f}, {t_phi:.2f}")
                else:
                    # air to ice
                    incoming_angle = np.arcsin(np.sin(zenith_reflection) * self.n_at_surface / N_AIR)
                    t_theta = geometryUtilities.get_fresnel_t_p(incoming_angle, n_1=N_AIR, n_2=self.n_at_surface)
                    t_phi = geometryUtilities.get_fresnel_t_s(incoming_angle, n_1=N_AIR, n_2=self.n_at_surface)
                    self.__logger.info(f"propagating from air to ice: transmission coefficient is {t_theta:.2f}, {t_phi:.2f}")

                fresnel_coefficients.append(
                    {'zenith': zenith_reflection, 'case': 'transmission', 'theta': t_theta, 'phi': t_phi})
            else:
                # in-ice propagation, reflection off the surface
                r_theta = geometryUtilities.get_fresnel_r_p(zenith_reflection, n_2=N_AIR, n_1=self.n_at_surface)
                r_phi = geometryUtilities.get_fresnel_r_s(zenith_reflection, n_2=N_AIR, n_1=self.n_at_surface)
                self.__logger.info(
                    "ray hits the surface at an angle {:.2f}deg -> reflection coefficient is r_theta = {:.2f}, r_phi = {:.2f}".format(
                        zenith_reflection / units.deg, r_theta, r_phi))

                fresnel_coefficients.append(
                    {'zenith': zenith_reflection, 'case': 'reflection', 'theta': r_theta, 'phi': r_phi})

        return fresnel_coefficients

    def get_path_length(self, iS, analytic=True):
        """
        Calculates the path length of solution iS

        Parameters
        ----------
        iS: int
            Choose for which solution to compute the launch vector, counting
            starts at zero

        analytic: bool
            If True the analytic solution is used. If False, a numerical integration is used. (default: True)

        Returns
        -------
        distance: float
            distance from x1 to x2 along the ray path

        Notes
        -----
        The analytic solution is based on the equation in the appendix of Sjoerd Bouma's PhD thesis.
        For more details, see there, or see the notes of ``ray_tracing_2D.get_path_length_analytic``.

        """
        n = self.get_number_of_solutions()
        if(iS >= n):
            self.__logger.error("solution number {:d} requested but only {:d} solutions exist".format(iS + 1, n))
            raise IndexError

        result = self._results[iS]
        if analytic:
            try:
                analytic_length = self._r2d.get_path_length_analytic(self._x1, self._x2, result['C0'],
                                                                      reflection=result['reflection'],
                                                                      reflection_case=result['reflection_case'])
                if (analytic_length != None):
                    return analytic_length
            except:
                self.__logger.warning("analytic calculation of travel time failed, switching to numerical integration")
                return self._r2d.get_path_length(self._x1, self._x2, result['C0'],
                                                  reflection=result['reflection'],
                                                  reflection_case=result['reflection_case'])
        else:
            return self._r2d.get_path_length(self._x1, self._x2, result['C0'],
                                              reflection=result['reflection'],
                                              reflection_case=result['reflection_case'])

    def get_travel_time(self, iS, analytic=True):
        """
        Calculates the travel time of solution iS

        Parameters
        ----------
        iS : int
            Choose for which solution to compute the launch vector, counting
            starts at zero

        analytic : bool
            If True the analytic solution is used. If False, a numerical integration is used. (default: True)

        Returns
        -------
        time: float
            travel time

        Notes
        -----
        The analytic solution is based on the equation in the appendix of Sjoerd Bouma's PhD thesis.
        For more details, see there, or see the notes of ``ray_tracing_2D.get_travel_time_analytic``.

        """
        n = self.get_number_of_solutions()
        if(iS >= n):
            self.__logger.error("solution number {:d} requested but only {:d} solutions exist".format(iS + 1, n))
            raise IndexError

        result = self._results[iS]
        if(analytic):
            try:
                analytic_time = self._r2d.get_travel_time_analytic(self._x1, self._x2, result['C0'],
                                                                reflection=result['reflection'],
                                                                reflection_case=result['reflection_case'])
                if (analytic_time != None):
                    return analytic_time
            except KeyError:
                self.__logger.warning("analytic calculation of travel time failed, switching to numerical integration")
                return self._r2d.get_travel_time(self._x1, self._x2, result['C0'],
                                                  reflection=result['reflection'],
                                                  reflection_case=result['reflection_case'])
        else:
            return self._r2d.get_travel_time(self._x1, self._x2, result['C0'],
                                              reflection=result['reflection'],
                                              reflection_case=result['reflection_case'])

    def get_attenuation(self, iS, frequency, max_detector_freq=None):
        """
        Calculates the signal attenuation due to attenuation in the medium (ice)

        Parameters
        ----------
        iS: int
            Choose for which solution to compute the launch vector, counting
            starts at zero

        frequency: array of floats
            The frequencies for which the attenuation is calculated

        max_detector_freq: float or None
            The maximum frequency of the final detector sampling
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

        result = self._results[iS]
        # the C0 parameter (together with the reflection specifiers) uniquely identifies the ray path
        # for the current geometry, hence the attenuation only needs to be calculated once per path
        # and frequency grid
        cache_key = (self._x1.tobytes(), self._x2.tobytes(), result['C0'], result['reflection'], result['reflection_case'],
                     np.asarray(frequency).tobytes(), max_detector_freq)
        if cache_key not in self._cache_attenuation:
            self._cache_attenuation[cache_key] = self._r2d.get_attenuation_along_path(
                self._x1, self._x2, result['C0'], frequency, max_detector_freq,
                reflection=result['reflection'],
                reflection_case=result['reflection_case'])

        return np.copy(self._cache_attenuation[cache_key])

    def get_focusing(self, iS, dz=-1. * units.cm, limit=2., analytic=False):
        """
        Calculate the focusing effect in the medium

        Parameters
        ----------
        iS: int
            Choose for which solution to compute the launch vector, counting
            starts at zero
        dz: float
            The infinitesimal change of the depth of the receiver, 1cm by default
            Only used if ``analytic=False``
        limit: float, default: 2
            The maximum signal focusing. Note that this limit is applied to the
            geometric focusing, i.e. before the impedance factor sqrt(n1/n2)
            is applied.
        analytic : bool, default: False
            If False, solve the ray tracing equation again for a slightly
            displaced receiver and obtain the ray convergence that way.

            If True, use the analytic solution for the focusing factor. Note
            that the analytic solution is not valid for horizontal rays (e.g.
            refracted rays); in that case, the numeric solution is automatically
            used instead.

        Returns
        -------
        focusing: float
            gain of the signal at the receiver due to the focusing effect

        Notes
        -----
        An extensive description of the focusing correction can be found in
        appendix A of https://doi.org/10.25593/open-fau-2262. This correction
        assumes a point source.

        Note that in the case of air-to-ice transmission (or vice versa),
        the fresnel coefficients already include both the impedance
        and a plane-wave (geometric) focusing correction. In order to avoid
        double-counting, this method returns the focusing factor multiplied
        by the inverse of the 'focusing' part of the fresnel coefficients
        for air-to-ice trajectories.

        """
        # the C0 parameter uniquely identifies the ray path for the current geometry. The focusing
        # factor is requested several times per solution (e.g. by `get_raytracing_output` and
        # `apply_propagation_effects`), hence, caching it avoids expensive recalculations
        # (the numerical calculation requires an additional ray tracing)
        cache_key = (self._x1.tobytes(), self._x2.tobytes(), iS, self._results[iS]['C0'], dz, limit, analytic)
        if cache_key in self._cache_focusing:
            return self._cache_focusing[cache_key]

        recVec = -1.0 * self.get_receive_vector(iS)
        # whether recVec or -recVec does not matter as sin(x) = sin(x+pi)
        # and cos(x) = - cos(x+pi) but cos terms only appear in abs(...)
        recAng = _get_zenith(recVec)
        lauVec = self.get_launch_vector(iS)
        lauAng = _get_zenith(lauVec)

        # we need to be careful here. If X1 (the emitter) is above the X2 (the receiver) the positions are swapped
        # do to technical reasons. Here, we want to change the receiver position slightly, so we need to check
        # is X1 and X2 was swapped and use the receiver value!
        if self._swap:
            vetPos = copy.copy(self._X2) # emitter
            recPos = copy.copy(self._X1) # receiver
        else:
            vetPos = copy.copy(self._X1) # emitter
            recPos = copy.copy(self._X2) # receiver

        recPos1 = np.array([recPos[0], recPos[1], recPos[2] + dz])
        n1 = self._medium.get_index_of_refraction(vetPos)
        n2 = self._medium.get_index_of_refraction(recPos)

        f = np.nan
        if analytic:
            res = self.get_results()[iS]
            impedance_factor = np.sqrt(n1 / n2) # used for debugging only
            f = self._r2d.get_focusing_analytic(
                self._x1, self._x2, res['C0'],
                res['reflection'], res['reflection_case']
            )

        if np.isnan(f): # either the analytic calculation failed, or we asked for the numerical solution
            distance = self.get_path_length(iS)
            if not hasattr(self, "_r1"):
                self._r1 = ray_tracing(
                    self._medium, self._attenuation_model,
                    log_level=self.__logger.level,
                    n_frequencies_integration=self._n_frequencies_integration,
                    n_reflections=self._n_reflections,
                    use_cpp=self.use_cpp, compile_numba=self.compile_numba)

            self._r1.set_start_and_end_point(vetPos, recPos1)
            self._r1.find_solutions()

            if iS < self._r1.get_number_of_solutions():
                lauVec1 = self._r1.get_launch_vector(iS)
                lauAng1 = _get_zenith(lauVec1)

                self.__logger.debug(
                    "focusing: receive angle %.2f / launch angle %.2f / d_launch_angle %.4f",
                    recAng / units.deg, lauAng / units.deg, (lauAng1-lauAng) / units.deg
                )

                focusing = np.sqrt(distance / np.sin(recAng) * np.abs((lauAng1 - lauAng) / (recPos1[2] - recPos[2])))

                # also take into account focussing in the phi-direction
                radius = np.linalg.norm(recPos - vetPos)
                sinTheta = np.linalg.norm((recPos-vetPos)[:-1]) / radius
                dphi_flat = distance * np.sin(lauAng)
                dphi_curved = radius * sinTheta
                focusing *= np.sqrt(dphi_flat / dphi_curved)

                if (self.get_results()[iS]['reflection'] != self._r1.get_results()[iS]['reflection']
                        or self.get_results()[iS]['reflection_case'] != self._r1.get_results()[iS]['reflection_case']):
                    self.__logger.error("Number or type of reflections are different between solutions - focusing correction may not be reliable.")
            else:
                focusing = 1.0
                self.__logger.warning("too few ray tracing solutions, setting focusing factor to 1")

            # now also correct for differences in refractive index between emitter and receiver position
            # (this is already included in the analytic calculation)
            impedance_factor = np.sqrt(n1 / n2)
            f = focusing * impedance_factor

        self.__logger.debug('amplification due to focusing of solution %d = %.3f x %.3f = %.3f ', iS, f / impedance_factor, impedance_factor, f)
        if f / impedance_factor > limit:
            self.__logger.info(f"amplification due to focusing is {f / impedance_factor:.1f}x -> limiting amplification factor to {limit:.1f}x")
            f = limit * impedance_factor

        # for ice-to-air transmission, the fresnel amplitude coefficients include an impedance factor
        # as well as a correction for the focusing for a plane wave. We have already included these
        # in the focusing factor f, so we should correct for this:
        if recPos[-1] > 0: # receiver in air
            reflection_angle = np.atleast_1d(self.get_reflection_angle(iS))[0]
            correction_term = np.sqrt(
                n2 / self.n_at_surface * np.abs(np.cos(recAng) / np.cos(reflection_angle)))
            self.__logger.debug('raytracing to air - correct focusing by %.3f', correction_term)
            f *= correction_term

        elif vetPos[-1] > 0: # emitter in air
            reflection_angle = np.atleast_1d(self.get_reflection_angle(iS))[0]
            correction_term = np.sqrt(
                self.n_at_surface / n1 * np.abs(np.cos(reflection_angle) / np.cos(lauAng)))
            self.__logger.debug('raytracing from air to ice - correct focusing by %.3f', correction_term)
            f *= correction_term

        self._cache_focusing[cache_key] = f
        return f

    def get_ray_path(self, iS):
        return self._r2d.get_path_reflections(self._x1, self._x2, self._results[iS]['C0'], 10000,
                                   reflection=self._results[iS]['reflection'],
                                   reflection_case=self._results[iS]['reflection_case'])

    def get_output_parameters(self):
        return [
            {'name': 'ray_tracing_C0', 'ndim': 1},
            {'name': 'ray_tracing_C1', 'ndim': 1},
            {'name': 'focusing_factor', 'ndim': 1},
            {'name': 'ray_tracing_reflection', 'ndim': 1},
            {'name': 'ray_tracing_reflection_case', 'ndim': 1},
            {'name': 'ray_tracing_solution_type', 'ndim': 1}
        ]

    def get_raytracing_output(self, i_solution):
        """
        Get the output of the ray tracing for a specific solution

        Parameters
        ----------
        i_solution: int
            Index of the raytracing solution

        Returns
        -------
        output_dict: dict
            Dictionary containing the output of the ray tracing.
            The C_0 and C_1 parameters are the parameters of the analytic function that describes the ray path.
            The solution type is the type of the solution (1: direct, 2:refracted, 3: reflected off the surface).
            The reflection parameter specifies the number of bottom reflections (in case of reflective layers in the ice).
            The reflection case specifies if the ray starts upward or downward (1: upward, 2: downward) (only relevant if bottom reflection > 0).
        """
        if self._config['propagation']['focusing']:
            focusing = self.get_focusing(i_solution, limit=float(self._config['propagation']['focusing_limit']))
        else:
            focusing = 1

        output_dict = {
            'ray_tracing_C0': self.get_results()[i_solution]['C0'],
            'ray_tracing_C1': self.get_results()[i_solution]['C1'],
            'ray_tracing_reflection': self.get_results()[i_solution]['reflection'],
            'ray_tracing_reflection_case': self.get_results()[i_solution]['reflection_case'],
            'ray_tracing_solution_type': self.get_solution_type(i_solution),
            'focusing_factor': focusing
        }
        return output_dict

    def apply_propagation_effects(self, efield, i_solution):
        """
        Apply propagation effects to the electric field

        Note that the 1/r weakening of the electric field is already accounted for in the signal generation.
        This function applies the 4 effects (if configured to do so...):
        1. Attenuation
        2. Reflection/Transmission (first at ice-air boundary and than at in-ice reflective layer)
        3. Focusing
        4. Birefringence

        Parameters
        ----------
        efield: ElectricField object
            The electric field that the effects should be applied to
        i_solution: int
            Index of the raytracing solution the propagation effects should be based on

        Returns
        -------
        efield: ElectricField object
            The modified ElectricField object
        """
        s_rate = efield.get_sampling_rate()
        spec = efield.get_frequency_spectrum()


        apply_attenuation = self._config['propagation']['attenuate_ice']
        if apply_attenuation:
            if self._max_detector_frequency is None:
                max_freq = np.max(efield.get_frequencies())
            else:
                max_freq = self._max_detector_frequency
            attenuation = self.get_attenuation(i_solution, efield.get_frequencies(), max_freq)
            spec *= attenuation

        # lets handle the general case of multiple reflections off the ice-air surface
        # Multiple relfections are possible if a reflective bottom layer exists.
        for fresnel_coefficients in self.get_fresnel_coefficients(i_solution):
            if fresnel_coefficients['case'] == 'reflection':
                efield[efp.reflection_coefficient_theta] = fresnel_coefficients['theta']
                efield[efp.reflection_coefficient_phi] = fresnel_coefficients['phi']

            spec[1] *= fresnel_coefficients['theta']
            spec[2] *= fresnel_coefficients['phi']

        # Mow also take possible bottom reflections into account (not included in the previous loop!)
        i_reflections = self.get_results()[i_solution]['reflection']
        if i_reflections > 0:
            # each reflection lowers the amplitude by the reflection coefficient and introduces a phase shift
            reflection_coefficient = self._medium.reflection_coefficient ** i_reflections
            phase_shift = (i_reflections * self._medium.reflection_phase_shift) % (2 * np.pi)
            # we assume that both efield components are equally affected
            spec[1] *= reflection_coefficient * np.exp(1j * phase_shift)
            spec[2] *= reflection_coefficient * np.exp(1j * phase_shift)
            self.__logger.debug(
                "ray is reflecting %d times at the bottom -> reducing the signal by a factor of %.2f",
                i_reflections, reflection_coefficient)

        # apply the focusing effect
        if self._config['propagation']['focusing']:
            focusing = self.get_focusing(i_solution, limit=float(self._config['propagation']['focusing_limit']))
            spec[1:] *= focusing

        # apply the birefringence effect
        if self._config['propagation']['birefringence']:
            bire_model = self._config['propagation']['birefringence_model']

            if self._config['propagation']['birefringence_propagation'] == 'analytical':
                spec = self.get_pulse_propagation_birefringence(spec, s_rate, i_solution, bire_model = bire_model)

            elif self._config['propagation']['birefringence_propagation'] == 'numerical':
                from NuRadioMC.SignalProp import radioproparaytracing
                launch_v = self.get_launch_vector(i_solution)
                radiopropa_rays = radioproparaytracing.radiopropa_ray_tracing(self._medium)
                radiopropa_rays.set_start_and_end_point(self._X1, self._X2)
                spec = radiopropa_rays.raytracer_birefringence(launch_v, spec, s_rate) #, bire_model = bire_model --> has to be implemented

        efield.set_frequency_spectrum(spec, efield.get_sampling_rate())
        return efield

    def set_config(self, config):
        """
        Change the configuration file used by the raytracer

        Parameters
        ----------
        config: dict or None
            The new configuration settings
            If None, the default config settings will be applied
        """
        if config is None:
            self._config = {'propagation': {}}
            self._config['propagation']['attenuate_ice'] = True
            self._config['propagation']['focusing_limit'] = 2
            self._config['propagation']['focusing'] = False
            self._config['propagation']['birefringence'] = False
        else:
            self._config = config

    def get_time_difference_plane_wave(self, src_zenith, src_azimuth, azimuth_convention = 'nuradio'):

        if src_zenith > np.pi/2:
            self.__logger.warning(f"Source zenith angle: {src_zenith:3f} ({src_zenith/units.deg:2f} deg) is above pi/2 (90 deg)! The plane wave time difference calculation only works for signals traversing from air to ice, aka coming from above, aka having theta between 0 and 90 degrees. Make sure to catch this!")
            return np.nan

        dt = self._r2d.get_time_difference_plane_wave_analytic(self._X1, self._X2, src_zenith, src_azimuth, azimuth_convention)
        return dt
