from __future__ import absolute_import, division, print_function
from NuRadioReco.utilities import units
from NuRadioMC.SignalProp.propagation import solution_types, solution_types_revert
import numpy as np
import logging
logging.basicConfig()

"""
Structure of a ray-tracing module. For documentation and development purposes.
"""

class ray_tracing_base:
    """
    base class of ray tracer. All ray tracing modules need to prodide the following functions
    """


    def __init__(self, medium, attenuation_model=None, log_level=logging.WARNING, 
                 n_frequencies_integration=None, n_reflections=None, config=None, 
                 detector = None):
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
        n_reflections: int (default 0)
            in case of a medium with a reflective layer at the bottom, how many reflections should be considered
        config: nested dictionary
            loaded yaml config file
        detector: detector object
        """
        self.__logger = logging.getLogger('ray_tracing_base')
        self.__logger.setLevel(log_level)

        self._medium = medium
        self._attenuation_model = attenuation_model
        self._n_frequencies_integration = n_frequencies_integration
        if(n_reflections):
            if(not hasattr(self._medium, "reflection") or self._medium.reflection is None):
                self.__logger.warning("ray paths with bottom reflections requested medium does not have any reflective layer, setting number of reflections to zero.")
                n_reflections = 0
        self._n_reflections = n_reflections

        self._config = config
        self._detector = detector
        self._max_detector_frequency = None
        if self._detector is not None:
            for station_id in self._detector.get_station_ids():
                sampling_frequency = self._detector.get_sampling_frequency(station_id, 0)
                if self._max_detector_frequency is None or sampling_frequency * .5 > self._max_detector_frequency:
                    self._max_detector_frequency = sampling_frequency * .5

        self._X1 = None
        self._X2 = None
        self._results = None

    def reset_solutions(self):
        self._X1 = None
        self._X2 = None
        self._results = None

    def set_start_and_end_point(self, x1, x2):
        """
        Set the start and end points between which raytracing solutions shall be found
        It is recommended to also reset the solutions from any previous raytracing to avoid
        confusing them with the current solution

        Parameters
        ----------
        x1: np.array of shape (3,), default unit
            start point of the ray
        x2: np.array of shape (3,), default unit 
            stop point of the ray
        """
        self.reset_solutions()
        self._X1 = np.array(x1, dtype =np.float)
        self._X2 = np.array(x2, dtype = np.float)
        if (self._n_reflections):
            if (self._X1[2] < self._medium.reflection or self._X2[2] < self._medium.reflection):
                self.__logger.error("start or stop point is below the reflective bottom layer at {:.1f}m".format(
                    self._medium.reflection / units.m))
                raise AttributeError("start or stop point is below the reflective bottom layer at {:.1f}m".format(
                    self._medium.reflection / units.m))

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

        Examples
        --------
        .. code-block::
        
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
        self.__logger.error('function not defined')
        raise NotImplementedError

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
        self.__logger.error('function not defined')
        raise NotImplementedError

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
        self.__logger.error('function not defined')
        raise NotImplementedError

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
        self.__logger.error('function not defined')
        raise NotImplementedError

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
        self.__logger.error('function not defined')
        raise NotImplementedError

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
        self.__logger.error('function not defined')
        raise NotImplementedError

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
        self.__logger.error('function not defined')
        raise NotImplementedError

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
        self.__logger.error('function not defined')
        raise NotImplementedError

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
        self.__logger.error('function not defined')
        raise NotImplementedError

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
        self.__logger.error('function not defined')
        raise NotImplementedError

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
        self.__logger.error('function not defined')
        raise NotImplementedError

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
        self.__logger.error('function not defined')
        raise NotImplementedError

    def get_number_of_raytracing_solutions(self):
        """
        Function that returns the maximum number of raytracing solutions that can exist between each given
        pair of start and end points
        """
        return 2 + 4 * self._n_reflections # number of possible ray-tracing solutions

    def get_config(self):
        """
        Function that returns the configuration currently used by the raytracer
        """
        return self._config

    def set_config(self, config):
        """
        Function to change the configuration file used by the raytracer

        Parameters
        ----------
        config: dict
            The new configuration settings
        """
        self._config = config