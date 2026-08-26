"""
Ray tracing module for layered exponential refractive index profiles.

This module implements analytic ray tracing in glacial ice or another medium where the
refractive index can be described as layers of exponentials where each layer follows

n(z) = n_ice - delta_n * exp(z / z_0)

This also supports propagation across layer boundaries where n(z) is not continuous, so also air-to-ice tracing.
We can also model the refractive index of air as one or more exponential layers. Note that we expect the ice surface at z=0 and an in-ice antenna (z<0). This is important, since we search for the in-ice reflected rays for signals from within the ice and for the ones passing through the surface coming from the air and therefore treat both situations a bit differently.

To find the ray solutions we needed to implement a number of functions
that you can use to :

* evaluate refractive index profiles
* compute analytic ray trajectories
* determine turning points
* solve for ray parameters connecting two points
* classify solutions (direct, refracted, reflected)

Once a solution is found we can evaluate a number of path specific parameters, such as:

* 2D path coordinates
* path length
* light travel time
* signal path angles at the emitter and receiver
* possibly reflection angle
* attenuation factor (frequency dependent)
* focusing factor

The implementation is an expansion of the previous analytic ray
tracing solver used in NuRadioMC . In order to enable compilation with
``numba.njit(nopython=True)``, the layer definitions are internally
converted from dictionary-based objects to arrays.
For more insight into the physics behind this approach and remarks on the solving strategy it is recommended
to have a look at the companion note to this module ``AnalyticRayTracingImpl/Notes-on-Ray-Tracing.pdf`` and the appendix C of
"NuRadioMC: Simulating the Radio Emission of Neutrinos from Interaction to Detector“ (2020). https://doi.org/10.1140/epjc/s10052-020-7612-8.
This appendix describes the implementation of the previously used single layer analytic raytracer.


Notes
-----
Coordinates are given as (y, z) with units of meters:

* y : horizontal distance
* z : vertical coordinate

z = 0 corresponds to the ice surface.

The ray parameter ``c0`` determines the curvature of the trajectory.
It can be seen as c0 = 1/(n(z)*sin(theta)) where n(z) is the refractive index and theta is the angle relative to the horizontal at the current depth z.

Layer definitions
-----------------
Layers are initially defined as dictionaries with the following keys:

z_min : float
    Lower depth boundary of the layer (m).

z_max : float
    Upper depth boundary of the layer.

n_ice : float
    Asymptotic refractive index of deep ice.

delta_n : float
    Steepness factor of refractive index change.

z_0 : float
    Depth factor controlling the transition depth of the index profile.

region : str
    Internal identifier of the physical region.

region_name : str
    Human-readable name of the region. For plotting.

Internally these definitions are converted to arrays using
:func:`layers_to_arrays` in order to support Numba compilation.

Contact: hannes.warnhofer@desy.de
Created 2026
Developed for use in NuRadioMC ray tracing.
"""

import numpy as np

from NuRadioMC.SignalProp.propagation import solution_types_revert
from NuRadioReco.utilities import units, constants

from NuRadioMC.SignalProp.propagation_base_class import ray_tracing_base


import logging
logger = logging.getLogger("NuRadioMC.analytic_ray_tracing")

NumbaList = list # fallback for get_path_segments function

from NuRadioMC.SignalProp.AnalyticRayTracingImpl.MultilayerAnalyticRayTracing.corefunctions import get_layer_index, get_n_1D, analytic_F, compute_offsets, build_y_field, evaluate_y, find_z_turn, get_turning_point, get_c0_from_theta, get_delta_y, get_skim_angle, determine_solution_type
from NuRadioMC.SignalProp.AnalyticRayTracingImpl.MultilayerAnalyticRayTracing.solver import find_solutions, find_solutions_bulk, reduce_solutions
from NuRadioMC.SignalProp.AnalyticRayTracingImpl.MultilayerAnalyticRayTracing.getrayparameters import get_path, get_path_segments, get_path_length_analytic, get_travel_time_analytic, get_launch_angle, get_receiving_angle, get_reflection_angle, get_attenuation_along_path, get_focusing_factor, get_launch_vector, get_receiving_vector, ds_dz_layer, get_path_length_numerical, get_travel_time_numerical
from NuRadioMC.SignalProp.AnalyticRayTracingImpl.MultilayerAnalyticRayTracing.planewave import get_inice_quantities, get_time_difference_plane_wave_analytic


import time
import functools
import logging


def log_timing(level=logging.DEBUG):

    def decorator(func):

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):

            start = time.perf_counter()

            try:
                result = func(self, *args, **kwargs)

            except Exception:

                elapsed_ms = (time.perf_counter() - start) * 1000

                self._logger.exception(
                    "%s failed after %.3f ms",
                    func.__name__,
                    elapsed_ms
                )

                raise

            elapsed_ms = (time.perf_counter() - start) * 1000

            self._logger.info(
                level,
                "%s completed in %.3f ms",
                func.__name__,
                elapsed_ms
            )

            return result

        return wrapper

    return decorator




class multi_layer_ray_tracing_2D(ray_tracing_base):

    def __init__(self, medium, attenuation_model=None,
                 log_level=logging.NOTSET,
                 n_frequencies_integration=32, dz=10*units.m,
                 use_optimized_start_values=False,
                 overwrite_speedup=None,
                 use_cpp=None,
                 compile_numba=True):
        """
        initialize 2D analytic ray tracing class for multilayer analytic raytracing

        This class is designed to have the same appearance and user interface as the corresponding 2D class from analyticraytracing.py,
        which is why sometimes there are seemingly unnecessary variables defined and the naming and structure of some functions might seem a bit off.
        This is done, so that we can reuse the ray_tracing class from analyticraytracing.py which maps from 3D to 2D and then back after the relevant parameters are calculated.

        Parameters
        ----------
        medium: NuRadioMC.utilities.medium class
            details of the medium
        attenuation_model: string
            specifies which attenuation model to use
            (default: None -> 'SP1' (see `ray_tracing_base._set__set_arguments`))
        log_level: logging.loglevel object
            Overrides verbosity (default NOTSET)
        n_frequencies_integration: int
            specifies for how many frequencies the signal attenuation is being calculated
            (default: None -> 100 (see `ray_tracing_base._set__set_arguments`))

        """
        self._logger = logging.getLogger('NuRadioMC.multi_layer_ray_tracing_2D')
        self._logger.setLevel(log_level)

        self.medium = medium

        self.attenuation_model = attenuation_model or "SP1"
        #if self.attenuation_model not in attenuation_util.model_to_int:
        #    raise NotImplementedError("attenuation model {} is not implemented".format(self.attenuation_model))

        #self.attenuation_model_int = attenuation_util.model_to_int[self.attenuation_model]

        self.__n_frequencies_integration = n_frequencies_integration
        self.dz = dz
        self.use_cpp = False # For compatibility with old raytracer
        #self.compile_numba = None # For compatibility with old raytracer
        self.compile_numba = compile_numba

        numba_available = False


        try:
            from numba import njit
            from numba.core.registry import CPUDispatcher
            from numba.typed import List as NumbaList
            numba_available = True
            self._logger.status("Numba version of raytracer is available")
        except ImportError:
            self._logger.warning("Numba is not available")
            NumbaList = list # fallback for get_path_segments function
            numba_available = False

        use_ensure_jitted = False

        if compile_numba and use_ensure_jitted:

            def ensure_jitted(func): # Function to check if already jitted or not and use jitted if available
                if isinstance(func, CPUDispatcher):
                    return func
                return njit(cache=True)(func)

            if numba_available:

                global get_layer_index, analytic_F, compute_offsets, build_y_field, evaluate_y, find_z_turn
                global get_turning_point, get_delta_y, get_n_1D, get_c0_from_theta, get_skim_angle
                global determine_solution_type, get_path_segments, get_path_length_analytic, get_launch_angle
                global get_receiving_angle, get_reflection_angle, get_travel_time_analytic, ds_dz_layer, get_focusing_factor
                global get_inice_quantities, get_time_difference_plane_wave_analytic

                try:
                    get_layer_index = ensure_jitted(get_layer_index)
                    analytic_F = ensure_jitted(analytic_F)
                    compute_offsets = ensure_jitted(compute_offsets)
                    build_y_field = ensure_jitted(build_y_field)
                    evaluate_y = ensure_jitted(evaluate_y)
                    find_z_turn = ensure_jitted(find_z_turn)
                    get_turning_point = ensure_jitted(get_turning_point)
                    get_delta_y = ensure_jitted(get_delta_y)
                    get_n_1D = ensure_jitted(get_n_1D)
                    get_c0_from_theta = ensure_jitted(get_c0_from_theta)
                    get_skim_angle = ensure_jitted(get_skim_angle)
                    determine_solution_type = ensure_jitted(determine_solution_type)
                    get_path_segments = ensure_jitted(get_path_segments)
                    get_path_length_analytic = ensure_jitted(get_path_length_analytic)
                    get_launch_angle = ensure_jitted(get_launch_angle)
                    get_receiving_angle = ensure_jitted(get_receiving_angle)
                    get_reflection_angle = ensure_jitted(get_reflection_angle)
                    get_travel_time_analytic = ensure_jitted(get_travel_time_analytic)
                    ds_dz_layer = ensure_jitted(ds_dz_layer)
                    get_focusing_factor = ensure_jitted(get_focusing_factor)
                    get_inice_quantities = ensure_jitted(get_inice_quantities)
                    get_time_difference_plane_wave_analytic = ensure_jitted(get_time_difference_plane_wave_analytic)

                    self.use_cpp = False

                except Exception:

                    self._logger.warning("Error in compiling methods using jit - proceeding without numba")
                    compile_numba = False

    @property
    def _layers_arr(self):
        return self.medium.get_layers_array

    #@log_timing()
    def determine_solution_type(self, x1, x2, c0):

        y1, z1 = x1
        y2, z2 = x2

        with_air = False
        if (z1 > 0.0) or (z2 > 0.0):
            with_air = True

        downgoing = False
        if z1 > z2:
            z1, z2 = z2, z1
            downgoing = True


        solution_type = determine_solution_type(y1, z1, y2, z2, c0, self._layers_arr, downgoing, with_air)

        self._logger.info(
            "solution_type | x1=%s x2=%s c0=%s type=%s",
            x1,
            x2,
            c0,
            solution_type
            )

        return solution_type

    def find_solutions(self, x1, x2, plot=False, reduce=True, *_, **__):

        with_air = (x1[1] > 0 or x2[1] > 0)

        dx = abs(x2[0] - x1[0])
        dz = abs(x2[1] - x1[1])

        horizontal = dx > 0 and dz < 5*units.m #(dz / dx < 1e-3)
        small_dx = dx < 5 * units.m

        if horizontal or small_dx:
            solver_chain = [find_solutions_bulk, find_solutions]
        else:
            solver_chain = [find_solutions, find_solutions_bulk]

        solutions = []
        for solver in solver_chain:
            solutions = solver(x1, x2, self._layers_arr)
            if solutions:
                break

        self._logger.info(
            "find_solutions | x1=%s x2=%s solutions=%s",
            x1, x2, solutions
        )

        if reduce:
            try:
                return reduce_solutions(solutions, with_air)
            except Exception:
                return solutions

        return solutions

    def get_n_2D(self, x1):
        n_z = get_n_1D(x1[1],self._layers_arr)
        return n_z

    #@log_timing()
    def get_travel_time_analytic(self, x1, x2, c0, *_, **__):

        travel_time = get_travel_time_analytic(c0, x1, x2, self._layers_arr)

        solution_type = self.determine_solution_type(x1, x2, c0)

        # Only sanity-check DIRECT solutions
        if solution_type == solution_types_revert['direct']:

            direct_len = np.sqrt((x2[0]-x1[0])**2 + (x2[1]-x1[1])**2) * units.m
            n_z1 = self.get_n_2D(x1)
            direct_time_lightspeed = direct_len * n_z1 / constants.c

            #if travel_time > direct_time_lightspeed * 1.5: # Break for obviously unphysical solutions (from wrong solutions, makes plotting easier)
            #    return None
            #if travel_time < direct_time_lightspeed * 0.9:
            #    return None

        self._logger.info(
            "get_travel_time_analytic | x1=%s x2=%s c0=%s travel_time=%s",
            x1,
            x2,
            c0,
            travel_time
            )

        return travel_time
    
    def get_travel_time(self,x1, x2, c0, *_, **__):

        travel_time = get_travel_time_numerical(c0, x1, x2, self._layers_arr)

        self._logger.info(
            "get_ptravel_time_numerical | x1=%s x2=%s c0=%s travel_time=%s",
            x1,
            x2,
            c0,
            travel_time
            )

        return travel_time

    #@log_timing()
    def get_path_length_analytic(self, x1, x2, c0, *_, **__):

        path_length = get_path_length_analytic(c0, x1, x2, self._layers_arr)

        solution_type = self.determine_solution_type(x1, x2, c0)

        # Only sanity-check DIRECT solutions
        if solution_type == solution_types_revert['direct']:

            direct_len = np.sqrt((x2[0]-x1[0])**2 + (x2[1]-x1[1])**2) * units.m

            #if path_length > direct_len * 1.5: # Break for obviously unphysical solutions (from wrong solutions, makes plotting easier)
            #    return None

            #if path_length < direct_len * 0.9:
            #    return None

        self._logger.info(
            "get_path_length_analytic | x1=%s x2=%s c0=%s path_length=%s",
            x1,
            x2,
            c0,
            path_length
            )

        return path_length
    
    def get_path_length(self,x1, x2, c0, *_, **__):

        path_length = get_path_length_numerical(c0, x1, x2, self._layers_arr)

        self._logger.info(
            "get_path_length_numerical | x1=%s x2=%s c0=%s path_length=%s",
            x1,
            x2,
            c0,
            path_length
            )

        return path_length

    def get_launch_vector(self, x1, x2, c0):
        return get_launch_vector(c0, x1, x2, self._layers_arr)

    def get_receive_vector(self, x1, x2, c0):
        return get_receiving_vector(c0, x1, x2, self._layers_arr)

    #@log_timing()
    def get_launch_angle(self, x1, c0, *_, **__):

        launch_angle = get_launch_angle(c0, x1, x1, self._layers_arr)

        self._logger.info(
            "get_launch_angle | x1=%s c0=%s launch_angle=%s",
            x1,
            c0,
            launch_angle
            )

        return launch_angle

    #@log_timing()
    def get_receive_angle(self, x1, x2, c0, *_, **__):

        receive_angle = get_receiving_angle(c0, x1, x2, self._layers_arr)

        self._logger.info(
            "get_receive_angle | x1=%s x2=%s c0=%s receive_angle=%s",
            x1,
            x2,
            c0,
            receive_angle
            )

        return receive_angle

    #@log_timing()
    def get_reflection_angle(self, x1, x2, c0, *_, **__):

        reflection_angle = get_reflection_angle(c0, x1, x2, self._layers_arr)

        self._logger.info(
            "get_reflection_angle | x1=%s x2=%s c0=%s reflection_angle=%s",
            x1,
            x2,
            c0,
            reflection_angle
            )

        return reflection_angle


    def get_path_reflections(self, x1, x2, c0, npoints=1000,*_, **__):
        return get_path(c0, x1, x2, self._layers_arr, npoints)

    def get_path_segments(self, x1, x2, c0, *_, **__):
        return get_path_segments(c0, x1, x2, self._layers_arr)

    def get_turning_point(self, x1, c0):
        with_air = False
        if x1[1] > 0.0 : with_air = True
        return get_turning_point(x1[0], x1[1], c0, self._layers_arr, with_air=with_air)

    #@log_timing()
    def get_focusing_analytic(self, x1, x2, c0, *_, **__):

        focusing_factor = get_focusing_factor(c0, x1, x2, self._layers_arr)

        self._logger.info(
            "get_focusing_analytic | x1=%s x2=%s c0=%s focusing_factor=%s",
            x1,
            x2,
            c0,
            focusing_factor
            )

        return focusing_factor

    def __get_frequencies_for_attenuation(self, frequency, max_detector_freq=None):
        """ Returns a frequency vector for the attenuation calculation.

        It takes the frequency vector of a simulated electric field and makes it sparser.
        This function is used to reduce the number of frequencies for which the attenuation
        is calculated (which is time consuming). Afterwards the attenuation factors for the
        missing frequencies can be interpolated.

        If max_detector_freq is None, the function will return a frequency vector (0, f_max] with
        self.__n_frequencies_integration frequencies (unless the original frequency vector is already sparser).
        If max_detector_freq is not None, the function will return a frequency vector (0, max_detector_freq] + (max_detector_freq, f_max]
        with the first part having self.__n_frequencies_integration frequencies and the second part having
        self.__n_frequencies_integration // 2 frequencies.

        Parameters
        ----------
        frequency: array
            Frequency vector of the simulated electric field
        max_detector_freq: float
            Maximum frequency of the detector (the nyquist frequency)

        Returns
        -------
        freqs: array
            Sparse frequency vector for the attenuation calculation
        """

        non_null_freqs = frequency > 0
        n_freqs = min(self.__n_frequencies_integration, np.sum(non_null_freqs))

        freqs = np.linspace(frequency[non_null_freqs].min(), frequency[non_null_freqs].max(), n_freqs)

        if (n_freqs < np.sum(non_null_freqs)  # original frequency vector is already sparse
            and max_detector_freq is not None):

            det_mask = frequency <= max_detector_freq
            total_mask = det_mask & non_null_freqs

            n_freqs = min(self.__n_frequencies_integration, np.sum(total_mask))
            freqs = np.linspace(frequency[total_mask].min(), frequency[total_mask].max(), n_freqs)
            # Append n_freqs // 2 frequencies between detector nyquist frequency and simulated nyquist frequency
            if np.sum(~det_mask) > 1:
                freqs = np.append(freqs, np.linspace(frequency[~det_mask].min(), frequency[~det_mask].max(), n_freqs // 2))


        self._logger.debug("Frequency vector for attenuation calculation: {}".format(freqs))
        return freqs

    #@log_timing()
    def get_attenuation_along_path(self, x1, x2, c0, frequency, max_detector_frequency=None, *_, **__):

        attenuation_model =  self.attenuation_model
        dz = self.dz
        freqs = self.__get_frequencies_for_attenuation(frequency, max_detector_frequency)

        attenuation_factor = get_attenuation_along_path(c0, x1, x2, self._layers_arr, frequency, freqs, attenuation_model, dz)

        self._logger.info(
            "get_attenuation_along_path | x1=%s x2=%s c0=%s attenuation_factor=%s",
            x1,
            x2,
            c0,
            attenuation_factor
            )

        return attenuation_factor

    def get_time_difference_plane_wave_analytic(self, x1, x2, src_zenith, src_azimuth, azimuth_convention = 'nuradio'):

        dt = get_time_difference_plane_wave_analytic(x1, x2, src_zenith, src_azimuth, self._layers_arr, azimuth_convention)

        return dt