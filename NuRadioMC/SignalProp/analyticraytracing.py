"""
Wrapper for different implementations of a 2D analytic ray tracer to get ray tracing solutions
in 3D for two arbitrary points x1 and x2. Implementations for the 2D ray tracer include a
CPP version, a python version with numba and a python version without numba.
The CPP version is the default if available, otherwise the python version with numba is used if available,
otherwise the python version without numba is used.

Implementations are in NuRadioMC/SignalProp/AnalyticRayTracing/
"""



from NuRadioReco.utilities import units, geometryUtilities, constants
from NuRadioMC.utilities import medium as medium_util, birefringence

from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework import base_trace

from NuRadioMC.SignalProp.propagation_base_class import ray_tracing_base
from NuRadioMC.SignalProp.AnalyticRayTracing.single_layer_analytic_raytracer import (
    cpp_available, numba_available, ray_tracing_2D
)

import numpy as np

import logging
logger = logging.getLogger("NuRadioMC.analytic_ray_tracing")

class ray_tracing(ray_tracing_base):
    """
    utility class (wrapper around the 2D analytic ray tracing code) to get
    ray tracing solutions in 3D for two arbitrary points x1 and x2
    """

    def __init__(self, medium, attenuation_model=None, log_level=logging.NOTSET,
                 n_frequencies_integration=None, n_reflections=None, config=None,
                 detector=None, ray_tracing_2D_kwards={},
                 use_cpp=None, compile_numba=None):
        """
        class initilization

        Parameters
        ----------
        medium: medium class
            class describing the index-of-refraction profile

        attenuation_model: string
            signal attenuation model
            (default: None -> 'SP1' (see `ray_tracing_base._set__set_arguments`))

        log_name:  string
            name under which things should be logged

        log_level: logging object
            specify the log level of the ray tracing class

            * logging.ERROR
            * logging.WARNING
            * logging.INFO
            * logging.DEBUG

            default is NOTSET (global control)

        n_frequencies_integration: int
            the number of frequencies for which the frequency dependent attenuation
            length is being calculated. The attenuation length for all other frequencies
            is obtained via linear interpolation.
            (default: None -> 100 (see `ray_tracing_base._set__set_arguments`))

        n_reflections: int
            in case of a medium with a reflective layer at the bottom, how many reflections should be considered
            (default: None -> 0 (see `ray_tracing_base._set__set_arguments`))

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

        use_cpp: bool
            if True, use CPP implementation of minimization routines
            default: True if CPP version is available

        compile_numba: bool (default: None)
            Only relevant if `use_cpp` is False. If None, the default is True (if `use_cpp` is False).
        """
        self.__logger = logging.getLogger('NuRadioMC.ray_tracing')
        self.__logger.setLevel(log_level)

        from NuRadioMC.utilities.medium_base import IceModelSimple
        if not isinstance(medium, IceModelSimple):
            self.__logger.error("The analytic raytracer can only handle ice model of the type 'IceModelSimple'")
            raise TypeError("The analytic raytracer can only handle ice model of the type 'IceModelSimple'")

        super().__init__(medium=medium,
                         attenuation_model=attenuation_model,
                         log_level=log_level,
                         n_frequencies_integration=n_frequencies_integration,
                         n_reflections=n_reflections,
                         config=config,
                         detector=detector)

        self.set_config(config=config)

        if use_cpp is None:
            use_cpp = cpp_available

        self.use_cpp = use_cpp
        if use_cpp:
            self.__logger.status("Using CPP version of ray tracer")
        else:
            # If we do not want to or can not use CPP, by default we try to use numba
            if compile_numba is None:
                compile_numba = True

            if compile_numba and numba_available:
                self.__logger.status("Using python with numba version of ray tracer")
            else:
                self.__logger.status("Using python without numba version of ray tracer")

        self._r2d = ray_tracing_2D(self._medium, self._attenuation_model, log_level=log_level,
                                    n_frequencies_integration=self._n_frequencies_integration,
                                    **ray_tracing_2D_kwards, use_cpp=use_cpp, compile_numba=compile_numba)

        self._swap = None
        self._dPhi = None
        self._R = None
        self._x1 = None
        self._x2 = None


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

        self._swap = False
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
        find all solutions between x1 and x2
        """
        self._results = self._r2d.find_solutions(self._x1, self._x2)
        for i in range(self._n_reflections):
            for j in range(2):
                self._results.extend(self._r2d.find_solutions(self._x1, self._x2, reflection=i + 1, reflection_case=j + 1))

        # check if not too many solutions were found (the same solution can potentially found twice because of numerical imprecision)
        if(self.get_number_of_solutions() > self.get_number_of_raytracing_solutions()):
            self.__logger.error(f"{self.get_number_of_solutions()} were found but only {self.get_number_of_raytracing_solutions()} are allowed! Returning zero solutions")
            self._results = []

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
            integer corresponding to the types in the dictionary solution_types
        """
        return self._r2d.determine_solution_type(self._x1, self._x2, self._results[iS]['C0'])

    def get_path(self, iS, n_points=1000):

        n = self.get_number_of_solutions()
        if(iS >= n):
            self.__logger.error("solution number {:d} requested but only {:d} solutions exist".format(iS + 1, n))
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
            sampling rate of the time traces
        i_solution: int
            choose which ray-tracing solution should be propagated
        bire_model: string
            choose the interpolation to fit the measured refractive index data
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
        ice_birefringence = medium_util.get_ice_model('birefringence_medium')
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

            N_effective = birefringence.get_effective_index_birefringence(direction, nx, ny, nz)
            sky_polarization = birefringence.get_polarization_birefringence(N_effective[0], N_effective[1], direction, nx, ny, nz, logger=self.__logger)

            t_0, t_1 = len_diff * N_effective / (constants.c * units.m / units.ns)

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
            choose which ray-tracing solution should be propagated
        bire_model: string
            choose the interpolation to fit the measured refractive index data
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
        ice_birefringence = medium_util.get_ice_model('birefringence_medium')
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

            N_effective = birefringence.get_effective_index_birefringence(direction, nx, ny, nz)
            sky_polarization = birefringence.get_polarization_birefringence(N_effective[0], N_effective[1], direction, nx, ny, nz, logger=self.__logger)

            t_0, t_1 = len_diff * N_effective / (constants.c * units.m / units.ns)
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
        n = self.get_number_of_solutions()
        if(iS >= n):
            self.__logger.error("solution number {:d} requested but only {:d} solutions exist".format(iS + 1, n))
            raise IndexError

        result = self._results[iS]
        return self._r2d.get_reflection_angle(self._x1, self._x2, result['C0'],
                                               reflection=result['reflection'], reflection_case=result['reflection_case'])

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

        Notes
        -----
        The analytic solution is based on the equation in the appendix of Sjoerd Bouma's PhD thesis.
        For more details, see there, or see the notes of `ray_tracing_2D.get_path_length_analytic`.

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
        calculates the travel time of solution iS

        Parameters
        ----------
        iS : int
            choose for which solution to compute the launch vector, counting
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
        For more details, see there, or see the notes of `ray_tracing_2D.get_travel_time_analytic`.

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
        n = self.get_number_of_solutions()
        if(iS >= n):
            self.__logger.error("solution number {:d} requested but only {:d} solutions exist".format(iS + 1, n))
            raise IndexError

        result = self._results[iS]
        return self._r2d.get_attenuation_along_path(self._x1, self._x2, result['C0'], frequency, max_detector_freq,
                                                     reflection=result['reflection'],
                                                     reflection_case=result['reflection_case'])

    def get_focusing(self, iS, dz=-1. * units.cm, limit=2., analytic=False):
        """
        calculate the focusing effect in the medium

        Parameters
        ----------
        iS: int
            choose for which solution to compute the launch vector, counting
            starts at zero
        dz: float
            the infinitesimal change of the depth of the receiver, 1cm by default
            Only used if ``analytic=False``
        limit: float, default: 2
            The maximum signal focusing.
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
        """

        recVec = self.get_receive_vector(iS)
        recVec = -1.0 * recVec
        recAng = np.arccos(recVec[2] / np.sqrt(recVec[0] ** 2 + recVec[1] ** 2 + recVec[2] ** 2))
        lauVec = self.get_launch_vector(iS)
        lauAng = np.arccos(lauVec[2] / np.sqrt(lauVec[0] ** 2 + lauVec[1] ** 2 + lauVec[2] ** 2))
        # we need to be careful here. If X1 (the emitter) is above the X2 (the receiver) the positions are swapped
        # do to technical reasons. Here, we want to change the receiver position slightly, so we need to check
        # is X1 and X2 was swapped and use the receiver value!
        if self._swap:
            vetPos = self._X2.copy()
            recPos = self._X1.copy()
            recPos1 = np.array([self._X1[0], self._X1[1], self._X1[2] + dz])
        else:
            vetPos = self._X1.copy()
            recPos = self._X2.copy()
            recPos1 = np.array([self._X2[0], self._X2[1], self._X2[2] + dz])

        f = np.nan
        if analytic:
            res = self.get_results()[iS]
            f = self._r2d.get_focusing_analytic(
                self._x1, self._x2, res['C0'],
                res['reflection'], res['reflection_case']
            )

        if np.isnan(f): # either the analytic calculation failed, or we asked for the numerical solution
            distance = self.get_path_length(iS)
            if not hasattr(self, "_r1"):
                self._r1 = ray_tracing(self._medium, self._attenuation_model, logging.WARNING,
                                self._n_frequencies_integration, self._n_reflections, use_cpp=self.use_cpp)

            self._r1.set_start_and_end_point(vetPos, recPos1)
            self._r1.find_solutions()
            if iS < self._r1.get_number_of_solutions():
                lauVec1 = self._r1.get_launch_vector(iS)
                lauAng1 = np.arccos(lauVec1[2] / np.sqrt(lauVec1[0] ** 2 + lauVec1[1] ** 2 + lauVec1[2] ** 2))
                self.__logger.debug(
                    "focusing: receive angle {:.2f} / launch angle {:.2f} / d_launch_angle {:.4f}".format(
                        recAng / units.deg, lauAng / units.deg, (lauAng1-lauAng) / units.deg
                    )
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

            self.__logger.debug(f'amplification due to focusing of solution {iS:d} = {focusing:.3f}')
            if(focusing > limit):
                self.__logger.info(f"amplification due to focusing is {focusing:.1f}x -> limiting amplification factor to {limit:.1f}x")
                focusing = limit

            # now also correct for differences in refractive index between emitter and receiver position
            if self._swap:
                n1 = self._medium.get_index_of_refraction(self._X2)  # emitter
                n2 = self._medium.get_index_of_refraction(self._X1)  # receiver
            else:
                n1 = self._medium.get_index_of_refraction(self._X1)  # emitter
                n2 = self._medium.get_index_of_refraction(self._X2)  # receiver
            f =  focusing * (n1 / n2) ** 0.5

        # for ice-to-air transmission, the fresnel amplitude coefficients include an impedance factor
        # as well as a correction for the focusing for a plane wave. We have already included these
        # in the focusing factor f, so we should correct for this:
        if recPos[-1] > 0: # receiver in air
            n_at_surface = self._medium.get_index_of_refraction([0, 0, -0.01*units.m])
            f *= np.sqrt(n2/n_at_surface * np.abs(np.cos(recAng) / np.cos(np.arcsin(np.sin(recAng) / n_at_surface))))
        elif vetPos[-1] > 0: # emitter in air
            n_at_surface = self._medium.get_index_of_refraction([0, 0, -0.01*units.m])
            f *= np.sqrt(n_at_surface/n1 * np.abs(np.cos(np.arcsin(np.sin(lauAng) / n_at_surface)) / np.cos(lauAng)))

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
        Note that the 1/r weakening of the electric field is already accounted for in the signal generation

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

        zenith_reflections = np.atleast_1d(self.get_reflection_angle(i_solution))  # lets handle the general case of multiple reflections off the surface (possible if also a reflective bottom layer exists)
        for zenith_reflection in zenith_reflections:  # loop through all possible reflections
            if (zenith_reflection is None):  # skip all ray segments where not reflection at surface happens
                continue
            if(self._x2[1] > 0):  # we need to treat the case of air to ice/ice to air propagation sepatately:
                # air/ice propagation
                self.__logger.warning(f"calculation of transmission coefficients and focussing factor for air/ice propagation is experimental and needs further validation")
                if(not self._swap):  # ice to air case
                    t_theta = geometryUtilities.get_fresnel_t_p(
                        zenith_reflection, n_2=1., n_1=self._medium.get_index_of_refraction([self._X2[0], self._X2[1], -1 * units.cm]))
                    t_phi = geometryUtilities.get_fresnel_t_s(
                        zenith_reflection, n_2=1., n_1=self._medium.get_index_of_refraction([self._X2[0], self._X2[1], -1 * units.cm]))
                    self.__logger.info(f"propagating from ice to air: transmission coefficient is {t_theta:.2f}, {t_phi:.2f}")
                else:   # air to ice
                    t_theta = geometryUtilities.get_fresnel_t_p(
                        zenith_reflection, n_1=1., n_2=self._medium.get_index_of_refraction([self._X2[0], self._X2[1], -1 * units.cm]))
                    t_phi = geometryUtilities.get_fresnel_t_s(
                        zenith_reflection, n_1=1., n_2=self._medium.get_index_of_refraction([self._X2[0], self._X2[1], -1 * units.cm]))
                    self.__logger.info(f"propagating from air to ice: transmission coefficient is {t_theta:.2f}, {t_phi:.2f}")
                spec[1] *= t_theta
                spec[2] *= t_phi
            else:
                #in-ice propagation
                r_theta = geometryUtilities.get_fresnel_r_p(
                    zenith_reflection, n_2=1., n_1=self._medium.get_index_of_refraction([self._X2[0], self._X2[1], -1 * units.cm]))
                r_phi = geometryUtilities.get_fresnel_r_s(
                    zenith_reflection, n_2=1., n_1=self._medium.get_index_of_refraction([self._X2[0], self._X2[1], -1 * units.cm]))
                efield[efp.reflection_coefficient_theta] = r_theta
                efield[efp.reflection_coefficient_phi] = r_phi
                spec[1] *= r_theta
                spec[2] *= r_phi
                self.__logger.info(
                    "ray hits the surface at an angle {:.2f}deg -> reflection coefficient is r_theta = {:.2f}, r_phi = {:.2f}".format(
                        zenith_reflection / units.deg,
                        r_theta, r_phi))
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
