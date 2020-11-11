from __future__ import absolute_import, division, print_function
import logging
logging.basicConfig()

"""
Structure of a ray-tracing module. For documentation and development purposes.
"""

solution_types = {1: 'direct',
                  2: 'refracted',
                  3: 'reflected'}


class ray_tracing_base:
    """
    base class of ray tracer. All ray tracing modules need to prodide the following functions
    """

    def __init__(self, medium, attenuation_model="SP1", log_level=logging.WARNING,
                 n_frequencies_integration=6):
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
        pass

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
        pass

    def find_solutions(self):
        """
        find all solutions between x1 and x2
        """
        pass

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
        pass

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
        pass

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
        pass

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
        pass

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
        pass

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
        pass

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
        pass

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
        pass

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
        pass

    def get_output_parameters(self):
        """
        Returns a list with information about parameters to include in the output data structure that are specific
        to this raytracer

        Returns:
        -----------------
        list with entries of form [{'name': str, 'ndim': int}]
            'name': Name of the new parameter to include in the data structure
            'ndim': Dimension of the data structure for the parameter
        """
        pass

    def get_raytracing_output(self, i_solution):
        """
        Write parameters that are specific to this raytracer into the output data.

        Parameters:
        ---------------
        i_solution: int
            The index of the raytracing solution

        Returns:
        ---------------
        dictionary with the keys matching the parameter names specified in get_output_parameters and the values being
        the results from the raytracing
        """
        pass

    def get_number_of_raytracing_solutions(self):
        """
        Function that returns the maximum number of raytracing solutions that can exist between each given
        pair of start and end points
        """
        pass

    def get_ray_tracing_perfomed(self, station_dictionary, station_id):
        """
        Function that can tell from the input dictionary if a raytracing with this raytracer has already
        been performed and written into the dictionary.


        Parameters:
        -------------
        station_dictionary: dict
            The input dictionary in which to search for raytracing results
        station_id: int
            ID of the station for which to check for a raytracing solution

        Returns
        -------------------
        True if there is a raytracing solution in the dictionary, otherwise false
        """
        pass

    def get_config(self):
        """
        Function that returns the configuration currently used by the raytracer
        """
        pass

    def set_config(self, config):
        """
        Function to change the configuration file used by the raytracer

        Parameters:
        ----------------
        config: dict
            The new configuration settings
        """