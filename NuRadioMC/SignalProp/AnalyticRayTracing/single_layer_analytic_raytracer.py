from NuRadioReco.utilities import units, geometryUtilities, constants
from NuRadioMC.utilities import attenuation as attenuation_util, medium as medium_util

from NuRadioMC.SignalProp.propagation_base_class import ray_tracing_base
from NuRadioMC.SignalProp.propagation import solution_types, solution_types_revert

from scipy import optimize, integrate
from operator import itemgetter
from functools import lru_cache
import numpy as np
import copy

import logging
logger = logging.getLogger("NuRadioMC.analytic_ray_tracing")

# check if CPP implementation is available
cpp_available = False

try:
    from NuRadioMC.SignalProp.AnalyticRayTracing.CPPAnalyticRayTracing import wrapper as cpp_wrapper
    cpp_available = True
    logger.status("CPP version of ray tracer is available")
except Exception:
    logger.info("trying to compile the CPP extension on-the-fly")
    try:
        import subprocess
        import os
        subprocess.call(os.path.join(os.path.dirname(os.path.abspath(__file__)), "install.sh"))
        from NuRadioMC.SignalProp.AnalyticRayTracing.CPPAnalyticRayTracing import wrapper as cpp_wrapper
        cpp_available = True
        logger.status("compilation was successful, CPP version of ray tracer is available")
    except Exception:
        logger.warning(
            "Compilation was not successful, using python version of ray tracer. "
            "Check NuRadioMC/NuRadioMC/SignalProp/AnalyticRayTracing/CPPAnalyticRayTracing for manual compilation.")
        cpp_available = False

numba_available = False

try:
    from numba import jit, njit
    numba_available = True
    logger.status("Numba version of raytracer is available")
except ImportError:
    logger.warning("Numba is not available")
    numba_available = False

"""
Models in the following list will use the speed-optimized algorithm to calculate the attenuation along the path.
Can be overwritten by init.
"""
speedup_attenuation_models = ["GL3"]


def get_n_steps(x1, x2, dx):
    """ Returns number of segments necessary for width to be approx dx """
    return max(int(abs(x1 - x2) // dx), 3)


def get_equidistant_steps(x1, x2, dx):
    """ Returns equi.dist. segments (np.linspace). Choose number of segments such that
        width of segments is approx. dx
    """
    if x1 == x2:
        return [x1]
    return np.linspace(x1, x2, get_n_steps(x1, x2, dx))


@lru_cache(maxsize=32)
def get_z_deep(ice_params):
    """
    Calculates the z_deep needed for integral along the homogeneous ice
    to know the path length or the times. We obtain the depth for which
    the index of refraction is 0.035% away of that of deep ice. This
    calculation assumes a monotonically increasing index of refraction
    with negative depth.
    """
    n_ice, z_0, delta_n = ice_params

    def diff_n_ice(z):

        rel_diff = 2e-5
        return delta_n * np.exp(z / z_0) / n_ice - rel_diff

    res = optimize.root(diff_n_ice, -100 * units.m).x[0]
    return res


def get_C0_from_log(logC0, n_ice):
    """
    transforms the fit parameter C_0 so that the likelihood looks better
    """
    return np.exp(logC0) + 1. / n_ice

def get_y(gamma, C_0, C_1, n_ice, b, z_0):
    """
    analytic form of the ray tracing part given an exponential index of refraction profile

    Parameters
    ----------
    gamma: (float or array)
        gamma is a function of the depth z
    C_0: (float)
        first parameter
    C_1: (float)
        second parameter
    """
    c = n_ice ** 2 - C_0 ** -2
    # we take the absolute number here but we only evaluate the equation for
    # positive outcome. This is to prevent rounding errors making the root
    # negative
    root = np.abs(gamma ** 2 - gamma * b + c)
    logargument = gamma / (2 * c ** 0.5 * (root) ** 0.5 - b * gamma + 2 * c)
    result = z_0 * (n_ice ** 2 * C_0 ** 2 - 1) ** -0.5 * np.log(logargument) + C_1
    return result

def get_gamma(z, delta_n, z_0):
    """
    transforms z coordinate into gamma
    """
    return delta_n * (np.exp(z / z_0))

def get_turning_point(c, b, z_0, delta_n):
    """
    calculate the turning point, i.e. the maximum of the ray tracing path;
    parameter is c = self.medium.n_ice ** 2 - C_0 ** -2

    This is either the point of reflection off the ice surface
    or the point where the saddle point of the ray (transition from upward to downward going)

    Technically, the turning point is set to z=0 if the saddle point is above the surface.

    Parameters
    ----------
    c: float
        related to C_0 parameter via c = self.medium.n_ice ** 2 - C_0 ** -2

    Returns
    -------
    typle (gamma, z coordinate of turning point)
    """
    gamma2 = np.array([b * 0.5 - (0.25 * b ** 2 - c) ** 0.5])  # first solution discarded
    z2 = np.log(gamma2 / delta_n) * z_0
    if(z2 > 0):
        z2 = np.array([0], dtype=np.float64)  # a reflection is just a turning point at z = 0, i.e. cases 2) and 3) are the same
        gamma2 = get_gamma(z2, delta_n, z_0)

    return gamma2, z2

def get_y_with_z_mirror(z, C_0, n_ice, b, delta_n, z_0, C_1=0.0):
    """
    analytic form of the ray tracing part given an exponential index of refraction profile

    this function automatically mirrors z values that are above the turning point,
    so that this function is defined for all z

    Parameters
    ----------
    z: (float or array)
        depth z
    C_0: (float)
        first parameter
    C_1: (float)
        second parameter
    """
    c = n_ice ** 2 - C_0 ** -2
    gamma_turn, z_turn = get_turning_point(c, b, z_0, delta_n)
    y_turn = get_y(gamma_turn, C_0, C_1, n_ice, b, z_0)
    if(z < z_turn):
        gamma = get_gamma(np.array([1])*z, delta_n, z_0)
        return get_y(gamma, C_0, C_1, n_ice, b, z_0)
    else:
        gamma = get_gamma(2 * z_turn - z, delta_n, z_0)
        return 2 * y_turn - get_y(gamma, C_0, C_1, n_ice, b, z_0)

def get_y_turn( C_0, x1, n_ice, b, delta_n, z_0):
    """
    calculates the y-coordinate of the turning point. This is either the point of reflection off the ice surface
    or the point where the saddle point of the ray (transition from upward to downward going)

    Parameters
    ----------
    C_0: float
        C_0 parameter of function
    x1: typle
        (y, z) start position of ray
    """
    c = n_ice ** 2 - C_0 ** -2
    gamma_turn, z_turn = get_turning_point(c, b, z_0, delta_n)
    C_1 = x1[0] - get_y_with_z_mirror(x1[1], C_0, n_ice, b, delta_n, z_0)[0]
    y_turn = get_y(gamma_turn[0], C_0, C_1, n_ice, b, z_0)
    return y_turn

def get_delta_y(C_0, x1, x2, n_ice, b, delta_n, z_0, medium_reflection, C0range=(-1.0,-1.0), reflection=0, reflection_case=2):
    """
    calculates the difference in the y position between the analytic ray tracing path
    specified by C_0 at the position x2
    """
    C_0_first = C_0

    if C0range[0] == -1.0 and C0range[1] == -1.0:
        C0range = (1. / n_ice, np.inf)
    else:
        C0range = (float(C0range[0]), float(C0range[1]))
    Corange_array = np.array(C0range ,  dtype=np.float64)
    if((C_0_first < Corange_array[0]) or(C_0_first > Corange_array[1])):
        return -np.inf
    c = n_ice ** 2 - C_0 ** -2
    # we consider two cases here,
    # 1) the rays start rising -> the default case
    # 2) the rays start decreasing -> we need to find the position left of the start point that
    #    that has rising rays that go through the point x1
    if(reflection > 0 and reflection_case == 2):
        y_turn = get_y_turn(C_0_first, x1, n_ice, b, delta_n, z_0)
        dy = y_turn - x1[0]
        x1[0] = x1[0] - 2.0 * dy

    for i in range(reflection):
        # we take account reflections at the bottom layer into account via
        # 1) calculating the point where the reflection happens
        # 2) starting a ray tracing from this new point

        # determine y translation first
        C_1 = x1[0] - get_y_with_z_mirror(x1[1], C_0_first, n_ice, b, delta_n, z_0)[0]

        x1 = get_reflection_point(C_0, C_1, n_ice, medium_reflection, b, z_0, delta_n)

    # determine y translation first
    C_1 = x1[0] - get_y_with_z_mirror(x1[1], C_0_first, n_ice, b, delta_n, z_0)[0]

    # for a given c_0, 3 cases are possible to reach the y position of x2
    # 1) direct ray, i.e., before the turning point
    # 2) refracted ray, i.e. after the turning point but not touching the surface
    # 3) reflected ray, i.e. after the ray reaches the surface
    gamma_turn, z_turn = get_turning_point(c, b, z_0, delta_n)
    y_turn = get_y(gamma_turn, C_0_first, C_1, n_ice, b, z_0)
    if(z_turn < x2[1]):  # turning points is deeper that x2 positions, can't reach target
        # the minimizer has problems finding the minimum if inf is returned here. Therefore, we return the distance
        # between the turning point and the target point + 10 x the distance between the z position of the turning points
        # and the target position. This results in a objective function that has the solutions as the only minima and
        # is smooth in C_0
        diff = ((z_turn - x2[1]) ** 2 + (y_turn - x2[0]) ** 2) ** 0.5 + 10 * np.abs(z_turn - x2[1])
        return -diff[0]
#             return -np.inf
    if(y_turn > x2[0]):  # we always propagate from left to right
        # direct ray
        y2_fit = get_y(get_gamma(x2[1], delta_n, z_0), C_0_first, C_1, n_ice, b, z_0)  # calculate y position at get_path position
        diff = (x2[0] - y2_fit)
        #if(hasattr(diff, '__len__')):
        #    diff = diff[0]

        return diff
    else:
        # now it's a bit more complicated. we need to transform the coordinates to
        # be on the mirrored part of the function
        z_mirrored = x2[1]
        gamma = get_gamma(z_mirrored, delta_n, z_0)
        y2_raw = get_y(gamma, C_0_first, C_1, n_ice, b, z_0)
        y2_fit = 2 * y_turn - y2_raw
        diff = (x2[0] - y2_fit)

        return -1 * diff[0]

def obj_delta_y_square( logC_0, x1, x2, n_ice, b, delta_n, z_0, medium_reflection, reflection=0, reflection_case=2):
    """
    objective function to find solution for C0
    """
    C_0 = get_C0_from_log(logC_0[0], n_ice)
    return get_delta_y(C_0, x1, x2, n_ice, b, delta_n, z_0, medium_reflection, (-1.0,-1.0), reflection=reflection, reflection_case=reflection_case) ** 2

def get_reflection_point(C_0, C_1, n_ice, medium_reflection, b, z_0, delta_n):
    """
    calculates the point where the signal gets reflected off the bottom of the ice shelf

    Returns tuple (y,z)
    """
    c = n_ice ** 2 - C_0 ** -2
    _gamma_turn, z_turn = get_turning_point(c, b, z_0, delta_n)
    x2 = np.array([0, medium_reflection],dtype = np.float64)
    x2[0]  = get_y_with_z_mirror(-x2[1] + 2 * z_turn, C_0, n_ice, b, delta_n, z_0, C_1)[0]
    return x2

def get_z_unmirrored(z, C_0, n_ice, b, z_0, delta_n):
    """
    calculates the unmirrored z position
    """
    c = n_ice ** 2 - C_0 ** -2
    gamma_turn, z_turn = get_turning_point(c, b, z_0, delta_n)
    gamma_turn = gamma_turn[0]
    z_turn = z_turn[0]
    z_unmirrored = z
    if(z > z_turn):
        z_unmirrored = 2 * z_turn - z
    return z_unmirrored

def get_y_diff(z_raw, C_0, n_ice, b, z_0, delta_n, in_air=False):
    """
    derivative dy(z)/dz

    Uses equation C.12 from [1]_

    References
    ----------
    .. [1] https://arxiv.org/abs/1906.01670
    """
    correct_for_air = False
    # if we are above the ice surface, the analytic expression below
    # does not apply. Therefore, we instead calculate dy/dz at z=0 where it is still valid
    # and then correct this using Snell's law
    if (not in_air) or (z_raw < 0):
        z = get_z_unmirrored(z_raw, C_0, n_ice, b, z_0, delta_n)
    else: # we are above the ice surface, where the below expression does not apply
        z = 0
        correct_for_air = True
    # There are two expressions for dy/dz in the NuRadioMC paper: C.12 and C.38
    # For some reason, C.38 was used initially, and is still included below in
    # case someone wants to verify they're equivalent. C.12 is much simpler and therefore used currently.
    #
    # c = n_ice ** 2 - C_0 ** -2
    # B = (0.2e1 * np.sqrt(c) * np.sqrt(-b * delta_n * np.exp(z / z_0) + delta_n **
    #                                   2 * np.exp(0.2e1 * z / z_0) + c) - b * delta_n * np.exp(z / z_0) + 0.2e1 * c)
    # D = n_ice ** 2 * C_0 ** 2 - 1
    # E1 = -b * delta_n * np.exp(z / z_0)
    # E2 = delta_n ** 2 * np.exp(0.2e1 * z / z_0)
    # E = (E1 + E2 + c)
    # res = (-np.sqrt(c) * np.exp(z / z_0) * b * delta_n + 0.2e1 * np.sqrt(-b * delta_n * np.exp(z /
    #          z_0) + delta_n ** 2 * np.exp(0.2e1 * z / z_0) + c) * c + 0.2e1 * c ** 1.5) / B * E ** -0.5 * (D ** (-0.5))

    n_z = n(z, n_ice, delta_n, z_0)

    if C_0**2 * n_z**2 > 1:
        res = 1 / np.sqrt(C_0**2 * n_z**2 - 1)
    else:
        res = np.inf

    if correct_for_air:
        n_surface = n(0, n_ice, delta_n, z_0)
        theta_ice = np.arctan(res)
        theta_air = np.arcsin(n_surface * np.sin(theta_ice))
        res = np.tan(theta_air)

    if z != z_raw:
        res *= -1

    return res


def n(z, n_ice, delta_n, z_0):
    """
    refractive index as a function of depth
    """
    res = n_ice - delta_n * np.exp(z / z_0)
#     if(type(z) is float):
#         if(z > 0):
#             return 1.
#         else:
#             return res
#     else:
#         res[z > 0] = 1.
    return res

class ray_tracing_2D(ray_tracing_base):

    def __init__(self, medium, attenuation_model=None,
                 log_level=logging.NOTSET,
                 n_frequencies_integration=None,
                 use_optimized_start_values=False,
                 overwrite_speedup=None,
                 use_cpp=None,
                 compile_numba=False):
        """
        initialize 2D analytic ray tracing class

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
        use_optimized_start_value: bool
            if True, the initial C_0 paramter (launch angle) is set to the ray that skims the surface
            (default: False)
        overwrite_speedup: bool
            The signal attenuation is calculated using a numerical integration
            along the ray path. This calculation can be computational inefficient depending on the details of
            the ice model. An optimization is implemented approximating the integral with a discrete sum with the loss
            of some accuracy, See PR #507. This optimization is used for all ice models listed in
            speedup_attenuation_models (i.e., "GL3"). With this argument you can explicitly activate or deactivate
            (True or False) if you want to use the optimization. (Default: None, i.e., use optimization if ice model is
            listed in speedup_attenuation_models)
        use_cpp: bool
            if True, use CPP implementation of minimization routines
            default: True if CPP version is available

        """
        self.__logger = logging.getLogger('NuRadioMC.ray_tracing_2D')
        self.__logger.setLevel(log_level)
        if use_cpp is None:
            use_cpp = cpp_available

        if cpp_available:
            if not use_cpp:
                self.__logger.info('C++ raytracer is available, but Python raytracer was requested. Using Python raytracer')
            else:
                self.__logger.info('Using C++ raytracer')
        else:
            if use_cpp:
                msg = ('C++ raytracer was explicitly requested, but is not available (i.e. on-the-fly compilation failed). '
					   'Abort.... ! Either fix the compilation or set use_cpp to False. '
					   'For compilation see NuRadioMC/SignalProp/install.sh resp. NuRadioMC/SignalProp/CPPAnalyticRayTracing.')
                self.__logger.error(msg)
                raise RuntimeError(msg)
            else:
                self.__logger.warning('C++ raytracer is not available. Using Python raytracer.')
                self.__logger.warning("check NuRadioMC/NuRadioMC/SignalProp/CPPAnalyticRayTracing for manual compilation")

        if isinstance(medium, medium_util.uniform_ice):
            msg = ('Analytic raytracer does not work with a uniform ice model. '
                    'Abort.... ! Use direct raytracing or a non-uniform ice model instead.')
            self.__logger.error(msg)
            raise RuntimeError(msg)

        self.medium = medium
        if not hasattr(self.medium, "reflection"):
            self.medium.reflection = None

        # This variable is needed for numba optimization as numba cannot associate None to a type
        self.reflection = 100
        if self.medium.reflection is not None:
            self.reflection = self.medium.reflection

        self.attenuation_model = attenuation_model or "SP1"
        if self.attenuation_model not in attenuation_util.model_to_int:
            raise NotImplementedError("attenuation model {} is not implemented".format(self.attenuation_model))

        self.attenuation_model_int = attenuation_util.model_to_int[self.attenuation_model]
        self.__b = 2 * self.medium.n_ice

        self.__n_frequencies_integration = n_frequencies_integration
        self.__use_optimized_start_values = use_optimized_start_values

        self._use_optimized_calculation = self.attenuation_model in speedup_attenuation_models
        if overwrite_speedup is not None:
            self._use_optimized_calculation = overwrite_speedup

        self.use_cpp = use_cpp
        if compile_numba:
            if numba_available:
                global get_reflection_point,obj_delta_y_square,get_delta_y
                global get_y_turn, get_y_with_z_mirror,get_turning_point
                global get_gamma, get_y, get_C0_from_log
                global get_z_unmirrored, n, get_y_diff
                try:
                    get_reflection_point = jit(get_reflection_point, nopython=True, cache=True)
                    obj_delta_y_square = jit(obj_delta_y_square, nopython=True, cache=True)
                    get_delta_y = jit(get_delta_y, nopython=True, cache=True)
                    get_y_turn = jit(get_y_turn, nopython=True, cache=True)
                    get_y_with_z_mirror = jit(get_y_with_z_mirror, nopython=True, cache=True)
                    get_turning_point = jit(get_turning_point, nopython=True, cache=True)
                    get_gamma = jit(get_gamma, nopython=True, cache=True)
                    get_y = jit(get_y, nopython=True, cache=True)
                    get_C0_from_log = jit(get_C0_from_log, nopython=True, cache=True)
                    get_z_unmirrored = jit(get_z_unmirrored, nopython=True, cache=True)
                    get_y_diff = jit(get_y_diff, nopython=True, cache=True)
                    n = jit(n, nopython=True, cache=True)
                    self.use_cpp = False
                except Exception:
                    self.__logger.warning("Error in compiling methods using jit - proceeding without numba")
                    compile_numba = False

    def get_C_1(self, x1, C_0):
        """
        calculates constant C_1 for a given C_0 and start point x1
        """
        return x1[0] - get_y_with_z_mirror(x1[1], C_0, self.medium.n_ice, self.__b, self.medium.delta_n, self.medium.z_0)[0]

    def get_c(self, C_0):
        return self.medium.n_ice ** 2 - C_0 ** -2

    def get_z_mirrored(self, x1, x2, C_0):
        """
        calculates the mirrored x2 position so that y(z) can be used as a continuous function
        """
        c = self.medium.n_ice ** 2 - C_0 ** -2
        C_1 = x1[0] - get_y_with_z_mirror(x1[1], C_0, self.medium.n_ice, self.__b, self.medium.delta_n, self.medium.z_0)[0]
        gamma_turn, z_turn = get_turning_point(c, self.__b, self.medium.z_0, self.medium.delta_n)
        gamma_turn = gamma_turn[0]
        z_turn = z_turn[0]
        y_turn = get_y(gamma_turn, C_0, C_1, self.medium.n_ice, self.__b, self.medium.z_0)
        zstart = x1[1]
        zstop = x2[1]
        if(y_turn < x2[0]):
            zstop = zstart + np.abs(z_turn - x1[1]) + np.abs(z_turn - x2[1])
        x2_mirrored = [x2[0], zstop]
        return x2_mirrored

    def ds(self, t, C_0):
        """
        helper to calculate line integral
        """
        return (get_y_diff(t, C_0, self.medium.n_ice, self.__b, self.medium.z_0, self.medium.delta_n) ** 2 + 1) ** 0.5

    def get_path_length(self, x1, x2, C_0, reflection=0, reflection_case=1):
        tmp = 0
        for iS, segment in enumerate(self.get_path_segments(x1, x2, C_0, reflection, reflection_case)):
            if(iS == 0 and reflection_case == 2):  # we can only integrate upward going rays, so if the ray starts downwardgoing, we need to mirror
                x11, x1, x22, x2, C_0, C_1 = segment
                x1t = copy.copy(x11)
                x2t = copy.copy(x2)
                x1t[1] = x2[1]
                x2t[1] = x11[1]
                x2 = x2t
                x1 = x1t
            else:
                x11, x1, x22, x2, C_0, C_1 = segment

            # first treat special case of ice to air propagation
            z_int = None
            points = None
            if(x2[1] > 0):
                # we need to integrat only until the ray touches the surface
                z_turn = 0
                y_turn = get_y(get_gamma(z_turn, self.medium.delta_n, self.medium.z_0), C_0, self.get_C_1(x1, C_0), self.medium.n_ice, self.__b, self.medium.z_0)
                d_air = ((x2[0] - y_turn) ** 2 + (x2[1]) ** 2) ** 0.5
                tmp += d_air
                z_int = z_turn
                self.__logger.info(f"adding additional propagation path through air of {d_air/units.m:.1f}m")
            else:
                x2_mirrored = self.get_z_mirrored(x1, x2, C_0)
                gamma_turn, z_turn = get_turning_point(self.medium.n_ice ** 2 - C_0 ** -2,self.__b, self.medium.z_0, self.medium.delta_n)
                z_turn = z_turn[0]
                if(x1[1] < z_turn and z_turn < x2_mirrored[1]):
                    points = [z_turn]
                z_int = x2_mirrored[1]
            path_length = integrate.quad(self.ds, x1[1], z_int, args=(C_0), points=points,
                                         epsabs=1e-4, epsrel=1.49e-08, limit=50)
            self.__logger.info(f"calculating path length ({solution_types[self.determine_solution_type(x1, x2, C_0)]}) \
                                from ({x1[0]:.0f}, {x1[1]:.0f}) to ({x2[0]:.2f}, {x2[1]:.2f}) = {path_length[0] / units.m:.2f} m")
            tmp += path_length[0]
        return tmp


    def get_travel_time(self, x1, x2, C_0, reflection=0, reflection_case=1):
        tmp = 0
        for iS, segment in enumerate(self.get_path_segments(x1, x2, C_0, reflection, reflection_case)):
            if(iS == 0 and reflection_case == 2):  # we can only integrate upward going rays, so if the ray starts downwardgoing, we need to mirror
                x11, x1, x22, x2, C_0, C_1 = segment
                x1t = copy.copy(x11)
                x2t = copy.copy(x2)
                x1t[1] = x2[1]
                x2t[1] = x11[1]
                x2 = x2t
                x1 = x1t
            else:
                x11, x1, x22, x2, C_0, C_1 = segment

            # first treat special case of ice to air propagation
            z_int = None
            points = None
            x2_mirrored = self.get_z_mirrored(x1, x2, C_0)
            if(x2[1] > 0):
                # we need to integrat only until the ray touches the surface
                z_turn = 0
                y_turn = get_y(get_gamma(z_turn, self.medium.delta_n, self.medium.z_0), C_0, self.get_C_1(x1, C_0), self.medium.n_ice, self.__b, self.medium.z_0)
                T_air = ((x2[0] - y_turn) ** 2 + (x2[1]) ** 2) ** 0.5 / constants.c
                tmp += T_air
                z_int = z_turn
                self.__logger.info(f"adding additional propagation path through air of {T_air/units.ns:.1f}ns")
            else:
                gamma_turn, z_turn = get_turning_point(self.medium.n_ice ** 2 - C_0 ** -2,self.__b, self.medium.z_0, self.medium.delta_n)
                z_turn = z_turn[0]
                if(x1[1] < z_turn and z_turn < x2_mirrored[1]):
                    points = [z_turn]
                z_int = x2_mirrored[1]
            def dt(t, C_0):
                z = get_z_unmirrored(t, C_0, self.medium.n_ice, self.__b, self.medium.z_0, self.medium.delta_n)
                return self.ds(t, C_0) / constants.c * n(z, self.medium.n_ice, self.medium.delta_n, self.medium.z_0)
            travel_time = integrate.quad(dt, x1[1], z_int, args=(C_0), points=points,
                                         epsabs=1e-10, epsrel=1.49e-08, limit=500)
            self.__logger.info("calculating travel time from ({:.0f}, {:.0f}) to ({:.0f}, {:.0f}) = ({:.0f}, {:.0f}) = {:.2f} ns".format(
                x1[0], x1[1], x2[0], x2[1], x2_mirrored[0], x2_mirrored[1], travel_time[0] / units.ns))
            tmp += travel_time[0]
        return tmp


    def get_path_length_analytic(self, x1, x2, C_0, reflection=0, reflection_case=1):
        r"""
        Analytic solution for the path length

        Notes
        -----
        Based on the equation in Sjoerd Bouma's PhD thesis, reproduced below.
        For an analytic ice model :math:`n(z) = n_\mathrm{ice} - \Delta n e^{z/z_0}`,
        a ray launched from depth :math:`z_s` at angle :math:`\theta_s`, and using the notation

        .. math::

          \beta &= n(z_s) \sin \theta_s, \\
	      \alpha &= n^2_\mathrm{ice} - \beta^2,\\
	      \gamma(z) &= n(z)^2 - \beta^2, \\
	      k_1(z) &= \sqrt{\alpha\gamma(z)} + n_\mathrm{ice} n(z) - \beta^2,

        the path length for one segment is given by

        .. math::

          s &= \int \frac{dz}{\cos{\theta}} = \int dz \sec\left[\arcsin\left(\frac{\beta}{n(z)}\right)\right] \\
	      &= \frac{n_\mathrm{ice}}{\sqrt{\alpha}} \left[z - z_0 \log \left(k_1(z)\right)\right] + z_0 \log \left(\sqrt{\gamma(z)} + n(z)\right) + C.


        """
        s = 0
        for iS, segment in enumerate(self.get_path_segments(x1, x2, C_0, reflection, reflection_case)):
            if(iS == 0 and reflection_case == 2):  # we can only integrate upward going rays, so if the ray starts downwardgoing, we need to mirror
                x11, x1, x22, x2, C_0, C_1 = segment
                x1t = copy.copy(x11)
                x2t = copy.copy(x2)
                x1t[1] = x2[1]
                x2t[1] = x11[1]
                x2 = x2t
                x1 = x1t
            else:
                x11, x1, x22, x2, C_0, C_1 = segment

            z1 = x1[1]
            z2 = x2[1]
            solution_type = self.determine_solution_type(x1, x2, C_0)

            # if x1, x2 are swapped, launch_angle and receive_angle might also be swapped
            # Fortunately, the path length is symmetric, so this does not matter
            launch_angle = self.get_launch_angle(x1, C_0, reflection, reflection_case)

            # define some constants and helper functions
            n_ice = self.medium.n_ice
            delta_n = self.medium.delta_n
            z_0 = self.medium.z_0
            n1 = n(x1[1], n_ice, delta_n, z_0)
            beta = n1 * np.sin(launch_angle)
            alpha = n_ice**2 - beta**2

            def gamma(z):
                return np.max([0, n(z, n_ice, delta_n, z_0)**2 - beta**2]) # due to numerical precision, could otherwise get slightly negative

            def l1(z):
                return np.sqrt(alpha * gamma(z)) + n_ice * n(z, n_ice, delta_n, z_0) - beta**2

            def l2(z):
                return np.sqrt(gamma(z)) + n(z, n_ice, delta_n, z_0)

            def get_s(z):
                s = n_ice / np.sqrt(alpha) * (z - z_0 * np.log(l1(z))) + z_0 * np.log(l2(z))
                return s

            if(x2[1] > 0): # ice-to-air case
                # we need to integrate only until the ray touches the surface
                z_turn = 0
                y_turn = get_y(get_gamma(z_turn, delta_n, z_0), C_0, self.get_C_1(x1, C_0), self.__b, z_0)
                d_air = ((x2[0] - y_turn) ** 2 + (x2[1]) ** 2) ** 0.5

                s += get_s(0) - get_s(z1) + d_air

            else:
                if(solution_type == 1):
                    s += get_s(z2) - get_s(z1)
                else:
                    if(solution_type == 3):
                        z_turn = 0
                    else:
                        gamma_turn, z_turn = get_turning_point(n_ice ** 2 - C_0 ** -2, self.__b, z_0, delta_n)
                        z_turn = z_turn[0]
        #             print('solution type {:d}, zturn = {:.1f}'.format(solution_type, z_turn))
                    s += 2 * get_s(z_turn) - get_s(z1) - get_s(z2)

        return s

    def get_travel_time_analytic(self, x1, x2, C_0, reflection=0, reflection_case=1):
        r"""
        Analytic solution for the travel time

        Notes
        -----
        Based on the equation in Sjoerd Bouma's PhD thesis, reproduced below.
        For an analytic ice model :math:`n(z) = n_\mathrm{ice} - \Delta n e^{z/z_0}`,
        a ray launched from depth :math:`z_s` at angle :math:`\theta_s`, and using the notation

        .. math::

          \beta &= n(z_s) \sin \theta_s, \\
	      \alpha &= n^2_\mathrm{ice} - \beta^2,\\
	      \gamma(z) &= n(z)^2 - \beta^2, \\
	      k_1(z) &= \sqrt{\alpha\gamma(z)} + n_\mathrm{ice} n(z) - \beta^2,

        the propagation time along one ray segment is given by:

        .. math::

            ct &= \int \frac{n(z)dz}{\cos{\theta}} \\
            &= \frac{n_\mathrm{ice}^2}{\sqrt{\alpha}} \left[z - z_0 \log \left(k_1(z)\right) \right]
            + z_0 \left[\sqrt{\gamma(z)} + n_\mathrm{ice} \log\left(\sqrt{\gamma(z)} + n(z) \right)\right] + C'.

        """
        ct = 0
        for iS, segment in enumerate(self.get_path_segments(x1, x2, C_0, reflection, reflection_case)):
            if(iS == 0 and reflection_case == 2):  # we can only integrate upward going rays, so if the ray starts downwardgoing, we need to mirror
                x11, x1, x22, x2, C_0, C_1 = segment
                x1t = copy.copy(x11)
                x2t = copy.copy(x2)
                x1t[1] = x2[1]
                x2t[1] = x11[1]
                x2 = x2t
                x1 = x1t
            else:
                x11, x1, x22, x2, C_0, C_1 = segment

            z1 = x1[1]
            z2 = x2[1]
            solution_type = self.determine_solution_type(x1, x2, C_0)
            # if x1, x2 are swapped, launch_angle and receive_angle might also be swapped
            # Fortunately, the path length is symmetric, so this does not matter
            launch_angle = self.get_launch_angle(x1, C_0, reflection, reflection_case)

            # define some constants and helper functions
            n_ice = self.medium.n_ice
            delta_n = self.medium.delta_n
            z_0 = self.medium.z_0
            n1 = n(x1[1], n_ice=n_ice, delta_n=delta_n, z_0=z_0)
            beta = n1 * np.sin(launch_angle)
            alpha = n_ice**2 - beta**2

            def gamma(z):
                return np.max([n(z, n_ice, delta_n, z_0)**2 - beta**2, 0])

            def l1(z):
                return np.sqrt(alpha * gamma(z)) + n_ice * n(z, n_ice, delta_n, z_0) - beta**2

            def l2(z):
                return np.sqrt(gamma(z)) + n(z, n_ice, delta_n, z_0)

            def get_ct(z):
                ct = z_0 * (
                    np.sqrt(gamma(z)) - n_ice**2/np.sqrt(alpha) * np.log(l1(z))
                    + n_ice * np.log(l2(z))
                ) + n_ice**2 * z / np.sqrt(alpha)
                return ct

            if(x2[1] > 0): # ice-to-air case
                # we need to integrate only until the ray touches the surface
                z_turn = 0
                y_turn = get_y(get_gamma(z_turn, delta_n, z_0), C_0, self.get_C_1(x1, C_0), n_ice, self.__b, z_0)
                d_air = ((x2[0] - y_turn) ** 2 + (x2[1]) ** 2) ** 0.5

                ct += get_ct(0) - get_ct(z1) + d_air

            else:
                if(solution_type == 1):
                    ct += get_ct(z2) - get_ct(z1)
                else:
                    if(solution_type == 3):
                        z_turn = 0
                    else:
                        gamma_turn, z_turn = get_turning_point(n_ice ** 2 - C_0 ** -2, self.__b, z_0, delta_n)
                        z_turn = z_turn[0]
        #             print('solution type {:d}, zturn = {:.1f}'.format(solution_type, z_turn))

                    ct += 2 * get_ct(z_turn) - get_ct(z1) - get_ct(z2)

        return ct / constants.c


    def get_focusing_analytic(self, x1, x2, C_0, reflection=0, reflection_case=1):
        """
        Analytic solution to calculate the focusing factor

        .. warning::

            Note that this solution is unstable for a refracted ray trajectory
            as the focusing integral diverges when the ray trajectory becomes horizontal!

        Based on the appendix of Sjoerd Bouma's PhD thesis.

        """

        s = self.get_path_length_analytic(x1, x2, C_0, reflection, reflection_case)
        # if x1, x2 are swapped, launch_angle and receive_angle might also be swapped
        # Fortunately, the focusing factor is symmetric, so this does not matter
        launch_angle = self.get_launch_angle(x1, C_0, reflection, reflection_case)
        receive_angle = self.get_receive_angle(x1, x2, C_0, reflection, reflection_case)

        w_phi = 0
        w_theta = 0


        n_ice = self.medium.n_ice
        delta_n = self.medium.delta_n
        z_0 = self.medium.z_0
        n1 = n(x1[1], n_ice, delta_n, z_0)
        n2 = n(x2[1], n_ice, delta_n, z_0)
        beta = n1 * np.sin(launch_angle)
        alpha = n_ice**2 - beta**2

        def gamma(z):
            return np.max([0, self.n(z)**2 - beta**2])

        def phi_focusing_width(z):
            w_phi = 1/np.sqrt(alpha) * (
                z - z_0 * np.log(
                    np.sqrt(alpha * gamma(z)) + n_ice*self.n(z) - beta**2
                )
            )
            return w_phi

        def theta_focusing_width(z):
            w_theta = (
                n_ice**2 * z / alpha**(3/2)
                + z_0 * (n_ice * self.n(z) + beta**2) / (alpha * np.sqrt(gamma(z)))
                - n_ice**2 * z_0 / alpha**(3/2) * np.log(
                    np.sqrt(alpha * gamma(z)) + n_ice*self.n(z) - beta**2
                )
            )
            return w_theta

        for iS, segment in enumerate(self.get_path_segments(x1, x2, C_0, reflection, reflection_case)):
            if(iS == 0 and reflection_case == 2):  # we can only integrate upward going rays, so if the ray starts downwardgoing, we need to mirror
                x11, x1, x22, x2, C_0, C_1 = segment
                x1t = copy.copy(x11)
                x2t = copy.copy(x2)
                x1t[1] = x2[1]
                x2t[1] = x11[1]
                x2 = x2t
                x1 = x1t
            else:
                x11, x1, x22, x2, C_0, C_1 = segment

            z1 = x1[1]
            z2 = x2[1]
            solution_type = self.determine_solution_type(x1, x2, C_0)


            if(x2[1] > 0): # ice-to-air case
                w_theta += theta_focusing_width(0) - theta_focusing_width(z1)
                w_phi += phi_focusing_width(0) - phi_focusing_width(z1)

                w_theta += z2 / (np.cos(receive_angle)**3)
                w_phi += z2 / np.cos(receive_angle)

            else:
                if(solution_type == 1):
                    w_theta += theta_focusing_width(z2) - theta_focusing_width(z1)
                    w_phi += phi_focusing_width(z2) - phi_focusing_width(z1)
                else:
                    if(solution_type == 3):
                        z_turn = 0
                    else:
                        gamma_turn, z_turn = self.get_turning_point(self.medium.n_ice ** 2 - C_0 ** -2)
                        # print('solution type {:d}, zturn = {:.1f}'.format(solution_type, z_turn))
                        self.__logger.info("Analytic focusing factor not valid for refracted trajectories, use numerical one instead...")
                        return np.nan

                    w_theta += 2 * theta_focusing_width(z_turn) - theta_focusing_width(z1) - theta_focusing_width(z2)
                    w_phi += 2 * phi_focusing_width(z_turn) - phi_focusing_width(z1) - phi_focusing_width(z2)


        f_inverse_squared = n1 * n2 * np.abs(np.cos(launch_angle) * np.cos(receive_angle)) * (
            w_theta * w_phi / s**2
        )

        return np.sqrt(1/f_inverse_squared)

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


        self.__logger.debug("Frequency vector for attenuation calculation: {}".format(freqs))
        return freqs

    def get_attenuation_along_path(self, x1, x2, C_0, frequency, max_detector_freq,
                                   reflection=0, reflection_case=1):

        attenuation_factor = np.ones_like(frequency)

        output = f"calculating attenuation for n_ref = {int(reflection):d}: "
        for iS, segment in enumerate(self.get_path_segments(x1, x2, C_0, reflection, reflection_case)):

            # we can only integrate upward going rays, so if the ray starts downwardgoing,
            # we need to mirror
            if iS == 0 and reflection_case == 2:
                x11, x1, x22, x2, C_0, C_1 = segment
                x1t = copy.copy(x11)
                x2t = copy.copy(x2)
                x1t[1] = x2[1]
                x2t[1] = x11[1]
                x2 = x2t
                x1 = x1t
            else:
                _, x1, _, x2, C_0, C_1 = segment

            # treat special ice to air case. Attenuation in air can be neglected, so only
            # calculate attenuation until the ray reaches the surface
            if x2[1] > 0:
                z_turn = 0
                y_turn = get_y(get_gamma(z_turn, self.medium.delta_n, self.medium.z_0), C_0, self.get_C_1(x1, C_0), self.medium.n_ice, self.__b, self.medium.z_0)
                x2 = [y_turn, z_turn]

            if self.use_cpp:
                mask = frequency > 0
                freqs = self.__get_frequencies_for_attenuation(frequency, max_detector_freq)

                tmp_attenuation_factor = np.zeros_like(freqs)
                for i, f in enumerate(freqs):
                    tmp_attenuation_factor[i] = cpp_wrapper.get_attenuation_along_path(
                        x1, x2, C_0, f, self.medium.n_ice, self.medium.delta_n,
                        self.medium.z_0, self.attenuation_model_int)

                n_inf_freqs = np.sum(np.isnan(tmp_attenuation_factor))
                if n_inf_freqs > 0:
                    self.__logger.warning(
                        f"CPP wrapper: Attenuation calculation failed for {n_inf_freqs} / {len(tmp_attenuation_factor)} frequencies, "
                        "setting attenuation (factor) to 0, i.e., inf attenuation in these bins")
                    tmp_attenuation_factor[np.isnan(tmp_attenuation_factor)] = 0

                self.__logger.debug(tmp_attenuation_factor)
                attenuation_factor_segment = np.ones_like(frequency)
                attenuation_factor_segment[mask] = np.interp(frequency[mask], freqs, tmp_attenuation_factor)

            else:

                x2_mirrored = self.get_z_mirrored(x1, x2, C_0)

                def dt(t, C_0, frequency):
                    z = get_z_unmirrored(t, C_0, self.medium.n_ice, self.__b, self.medium.z_0, self.medium.delta_n)
                    return self.ds(t, C_0) / attenuation_util.get_attenuation_length(z, frequency, self.attenuation_model)

                # to speed up things we only calculate the attenuation for a few frequencies
                # and interpolate linearly between them
                mask = frequency > 0
                freqs = self.__get_frequencies_for_attenuation(frequency, max_detector_freq)
                gamma_turn, z_turn = get_turning_point(self.medium.n_ice ** 2 - C_0 ** -2,self.__b, self.medium.z_0, self.medium.delta_n)
                z_turn = z_turn[0]
                self.__logger.info("_use_optimized_calculation {}".format(self._use_optimized_calculation))

                if self._use_optimized_calculation:
                    # The integration of the attenuation factor along the path with scipy.quad is inefficient. The
                    # reason for this is that attenuation profile is varying a lot with depth. Hence, to improve
                    # performance we sum over discrete segments with loss of some accuracy.
                    # However, when a path becomes to horizontal (i.e., at the turning point of an refracted ray)
                    # the calculation via a discrete sum becomes to inaccurate. This is because we describe the
                    # path as a function of dz (vertical distance) and have the sum over discrete bins in dz. To avoid that
                    # we fall back to a numerical integration with scipy.quad only around the turning point. However, instead
                    # of integrating over dt (the exponent of the attenuation factor), we integrate only over ds (the path length)
                    # and evaluate the attenuation (as function of depth and frequency) for the central depth of this segment
                    # (because this is what takes time)
                    # For more details and comparisons see PR #507 : https://github.com/nu-radio/NuRadioMC/pull/507

                    # define the width of the vertical (!) segments over which we sum.
                    # Since we use linspace the actual width will differ slightly
                    # The data for the attenuation length of GL3 is also spaced by 10m.
                    dx = 10 * units.m
                    # define the vertical window around a turning point within we will fall back to a numerical integration
                    integration_window_size = 20 * units.m

                    # Check if a turning point "z_turn" is within our ray path or close to it (i.e., right behind the antenna)
                    # if so we need to fallback to numerical integration
                    fallback = False
                    if x1[1] - integration_window_size / 2 < z_turn and z_turn < x2_mirrored[1] + integration_window_size / 2:
                        fallback = True

                    if fallback:
                        # If we need the fallback, make sure that the turning point is in the middle of a segment (unless it is at
                        # the edge of a path). Otherwise the segment next to the turning point will be inaccurate
                        integration_window = [max(x1[1], z_turn - integration_window_size / 2),
                            min(z_turn + integration_window_size / 2, x2_mirrored[1])]

                        # Merge two arrays which start and stop at integration_window (and thus include it). The width might be slightly different
                        path_steps = np.append(get_equidistant_steps(x1[1], integration_window[0], dx), get_equidistant_steps(integration_window[1], x2_mirrored[1], dx))
                    else:
                        path_steps = get_equidistant_steps(x1[1], x2_mirrored[1], dx)

                    # get the actual width of each segment and their center
                    dx_actuals = np.diff(path_steps)
                    mid_points = path_steps[:-1] + dx_actuals / 2

                    # calculate attenuation for the different sub segments using the middle depth of the segment
                    # has the shape (#path_steps, #frequencies)
                    attenuation_factor_exponent_tmp = np.array(
                        [[dt(x, C_0, f) * dx_actual for x, dx_actual in zip(mid_points, dx_actuals)]
                        for f in freqs])
                    # attenuation_factor_exponent_tmp = dt(mid_points, C_0, freqs) * dx_actuals

                    if fallback:
                        # for the segment around z_turn fall back to integration. We only integrate ds (and not dt) for performance reasons

                        # find segment which contains z_turn
                        idx = np.digitize(z_turn, path_steps) - 1

                        # if z_turn is outside of segments
                        if idx == len(path_steps) - 1:
                            idx -= 1
                        elif idx == -1:
                            idx = 0

                        integrand = integrate.quad(self.ds, path_steps[idx], path_steps[idx + 1], args=(C_0), epsrel=1e-2, points=[z_turn])[0]
                        attenuation = attenuation_util.get_attenuation_length(z_turn, freqs, self.attenuation_model)

                        attenuation_factor_exponent_tmp[:, idx] = integrand / attenuation

                    # sum over all path_steps -> only remaining dimension is frequency
                    attenuation_factor_exponent = np.sum(attenuation_factor_exponent_tmp, axis=1)

                else:
                    points = None
                    if x1[1] < z_turn and z_turn < x2_mirrored[1]:
                        points = [z_turn]

                    attenuation_factor_exponent = np.array([integrate.quad(dt, x1[1], x2_mirrored[1], args=(
                        C_0, f), epsrel=1e-2, points=points)[0] for f in freqs])

                # Function of the frequency (shape = n_freqs)
                attenuation_factor_segment_tmp = np.exp(-1 * attenuation_factor_exponent)

                attenuation_factor_segment = np.ones_like(frequency)
                attenuation_factor_segment[mask] = np.interp(frequency[mask], freqs, attenuation_factor_segment_tmp)
                self.__logger.info("calculating attenuation from ({:.0f}, {:.0f}) to ({:.0f}, {:.0f}) = ({:.0f}, {:.0f}) =  a factor {}".format(
                    x1[0], x1[1], x2[0], x2[1], x2_mirrored[0], x2_mirrored[1],  attenuation_factor_segment))

            iF = len(frequency) // 3
            output += "adding attenuation for path segment {:d} -> {:.2g} at {:.0f} MHz, ".format(
                iS, attenuation_factor_segment[iF], frequency[iF] / units.MHz)

            attenuation_factor *= attenuation_factor_segment

        self.__logger.info(output)
        return attenuation_factor

    def get_path_segments(self, x1, x2, C_0, reflection=0, reflection_case=1):
        """
        Calculates the different segments of the path that makes up the full ray tracing path
        One segment per bottom reflection.

        Parameters
        ----------
        x1: tuple
            (y, z) coordinate of start value
        x2: tuple
            (y, z) coordinate of stop value
        C_0: float
            C_0 parameter of analytic ray path function
        reflection: int (default 0)
            the number of bottom reflections to consider
        reflection_case: int (default 1)
            only relevant if `reflection` is larger than 0

            * 1: rays start upwards
            * 2: rays start downwards

        Returns
        -------
        segments: list
            list of segments, each segment is a list with the following elements:

                * x1: original x1
                * x1 of path segment
                * x2: original x2
                * x2 of path segment
                * C_0: C_0 of path segment
                * C_1: C_1 of path segment

        """
        x1 = copy.copy(x1)
        x2 = copy.copy(x2)
        x1_orig = copy.copy(x1)
        x2_orig = copy.copy(x2)

        if reflection == 0:
            C_1 = self.get_C_1(x1, C_0)
            return [[x1_orig, x1, x2_orig, x2, C_0, C_1]]

        if reflection_case == 2:
            # the code only allows upward going rays, thus we find a point left from x1 that has an upward going ray
            # that will produce a downward going ray through x1
            y_turn = get_y_turn(C_0, x1, self.medium.n_ice, self.__b, self.medium.delta_n, self.medium.z_0)
            dy = y_turn - x1[0]
            self.__logger.debug("relaction case 2: shifting x1 {} to {}".format(x1, x1[0] - 2 * dy))
            x1[0] = x1[0] - 2 * dy

        segments = []
        for i in range(reflection + 1):
            self.__logger.debug("calculation path for reflection = {}".format(i + 1))
            C_1 = self.get_C_1(x1, C_0)
            x2 = get_reflection_point(C_0, C_1, self.medium.n_ice, self.reflection, self.__b, self.medium.z_0, self.medium.delta_n)
            stop_loop = False
            if x2[0] > x2_orig[0]:
                stop_loop = True
                x2 = x2_orig

            segments.append([x1_orig, x1, x2_orig, x2, C_0, C_1])
            if stop_loop:
                break

            self.__logger.debug("setting x1 from {} to {}".format(x1, x2))
            x1 = x2

        return segments

    def get_angle(self, x, x_start, C_0, reflection=0, reflection_case=1, in_air=False):
        """
        calculates the angle with respect to the positive z-axis of the ray path at position x

        Parameters
        ----------
        x: tuple
            (y, z) coordinate to calculate the angle
        x_start: tuple
            (y, z) start position of the ray
        C_0: float
            C_0 parameter of analytic ray path function
        reflection: int (default 0)
            the number of bottom reflections to consider
        reflection_case: int (default 1)
            only relevant if `reflection` is larger than 0

            * 1: rays start upwards
            * 2: rays start downwards

        """
        last_segment = self.get_path_segments(x_start, x, C_0, reflection, reflection_case)[-1]
        x_start = last_segment[1]

        if in_air:
            z = x[1]
        else:
            z = self.get_z_mirrored(x_start, x, C_0)[1]
        dy = get_y_diff(z, C_0, self.medium.n_ice, self.__b, self.medium.z_0, self.medium.delta_n, in_air=in_air)
        angle = np.arctan(dy)
        if(angle < 0):
            angle = np.pi + angle
        return angle

    def get_launch_angle(self, x1, C_0, reflection=0, reflection_case=1):
        return self.get_angle(x1, x1, C_0, reflection, reflection_case, in_air=x1[1]>0)

    def get_receive_angle(self, x1, x2, C_0, reflection=0, reflection_case=1):
        return np.pi - self.get_angle(x2, x1, C_0, reflection, reflection_case, in_air=x2[1]>0)

    def get_reflection_angle(self, x1, x2, C_0, reflection=0, reflection_case=1):
        """
        calculates the angle under which the ray reflects off the surface. If not reflection occurs, None is returned

        If reflections off the bottom (e.g. Moore's Bay) are simulated, an array with reflection angles (one for
        each track segment) is returned

        Parameters
        ----------
        x1: tuple
            (y, z) start position of ray
        x2: tuple
            (y, z) stop position of the ray
        C_0: float
            C_0 parameter of analytic ray path function
        reflection: int (default 0)
            the number of bottom reflections to consider
        reflection_case: int (default 1)
            only relevant if `reflection` is larger than 0
            * 1: rays start upwards
            * 2: rays start downwards
        """
        output = []
        c = self.medium.n_ice ** 2 - C_0 ** -2
        for segment in self.get_path_segments(x1, x2, C_0, reflection, reflection_case):
            x11, x1, x22, x2, C_0, C_1 = segment
            gamma_turn, z_turn = get_turning_point(c,self.__b, self.medium.z_0, self.medium.delta_n)
            z_turn = z_turn[0]
            y_turn = get_y_turn(C_0, x1, self.medium.n_ice, self.__b, self.medium.delta_n, self.medium.z_0)
            if((z_turn >= 0) and (y_turn > x11[0]) and (y_turn < x22[0])):  # for the first track segment we need to check if turning point is right of start point (otherwise we have a downward going ray that does not have a turning point), and for the last track segment we need to check that the turning point is left of the stop position.
                r = self.get_angle(np.array([y_turn, 0]), x1, C_0)
                self.__logger.debug(
                    "reflecting off surface at y = {:.1f}m, reflection angle = {:.1f}deg".format(y_turn, r / units.deg))
                output.append(r)
            else:
                output.append(None)
        return np.squeeze(output)

    def get_path(self, x1, x2, C_0, n_points=1000):
        """
        for plotting purposes only, returns the ray tracing path between x1 and x2

        the result is only valid if C_0 is a solution to the ray tracing problem

        Parameters
        ----------
        x1: array
            start position (y, z)
        x2: array
            stop position (y, z)
        C_0: (float)
            first parameter
        n_points: integer (optional)
            the number of coordinates to calculate

        Returns
        -------
        yy: array
            the y coordinates of the ray tracing path
        zz: array
            the z coordinates of the ray tracing path
        """
        c = self.medium.n_ice ** 2 - C_0 ** -2
        C_1 = x1[0] - get_y_with_z_mirror(x1[1], C_0, self.medium.n_ice, self.__b, self.medium.delta_n, self.medium.z_0)[0]
        gamma_turn, z_turn = get_turning_point(c, self.__b, self.medium.z_0, self.medium.delta_n)
        gamma_turn = gamma_turn[0]
        z_turn = z_turn[0]
        y_turn = get_y(gamma_turn, C_0, C_1, self.medium.n_ice, self.__b, self.medium.z_0)
        zstart = x1[1]
        zstop = self.get_z_mirrored(x1, x2, C_0)[1]
        z = np.linspace(zstart, zstop, n_points)
        mask = z < z_turn
        res = np.zeros_like(z)
        zs = np.zeros_like(z)
        gamma = get_gamma(z[mask], self.medium.delta_n, self.medium.z_0)
        zs[mask] = z[mask]
        res[mask] = get_y(gamma, C_0, C_1, self.medium.n_ice, self.__b, self.medium.z_0)
        if x2[1] > 0:  # treat ice to air case
            zenith_reflection = self.get_reflection_angle(x1, x2, C_0)
            n_1 = self.medium.get_index_of_refraction([y_turn, 0, z_turn])
            zenith_air = geometryUtilities.get_fresnel_angle(zenith_reflection, n_1=n_1, n_2=1)
            zs[~mask] = z[~mask]
            res[~mask] = zs[~mask] * np.tan(zenith_air) + y_turn
        else:
            gamma = get_gamma(2 * z_turn - z[~mask], self.medium.delta_n, self.medium.z_0)
            res[~mask] = 2 * y_turn - get_y(gamma, C_0, C_1, self.medium.n_ice, self.__b, self.medium.z_0)
            zs[~mask] = 2 * z_turn - z[~mask]

        self.__logger.debug('turning points for C_0 = {:.2f}, b= {:.2f}, gamma = {:.4f}, z = {:.1f}, y_turn = {:.0f}'.format(
            C_0, self.__b, gamma_turn, z_turn, y_turn))
        return res, zs

    def get_path_reflections(self, x1, x2, C_0, n_points=1000, reflection=0, reflection_case=1):
        """
        calculates the ray path in the presence of reflections at the bottom
        The full path is constructed by multiple calls to the `get_path()` function to put together the full path

        Parameters
        ----------
        x1: tuple
            (y, z) coordinate of start value
        x2: tuple
            (y, z) coordinate of stop value
        C_0: float
            C_0 parameter of analytic ray path function
        n_points: int (default 1000)
            the number of points of the numeric path
        reflection: int (default 0)
            the number of bottom reflections to consider
        reflection_case: int (default 1)
            only relevant if `reflection` is larger than 0

            * 1: rays start upwards
            * 2: rays start downwards

        Returns
        -------
        yy: array
            the y coordinates of the ray tracing path
        zz: array
            the z coordinates of the ray tracing path
        """
        yy = []
        zz = []
        x1 = copy.copy(x1)
        x11 = copy.copy(x1)

        if(reflection and reflection_case == 2):
            # the code only allows upward going rays, thus we find a point left from x1 that has an upward going ray
            # that will produce a downward going ray through x1
            y_turn = get_y_turn(C_0, x1, self.medium.n_ice, self.__b, self.medium.delta_n, self.medium.z_0)
            dy = y_turn - x1[0]
            self.__logger.debug("relaction case 2: shifting x1 {} to {}".format(x1, x1[0] - 2 * dy))
            x1[0] = x1[0] - 2 * dy

        if(reflection == 0):
            # in case of no bottom reflections, return path right away
            return self.get_path(x1, x2, C_0, n_points)
        x22 = copy.copy(x2)
        for i in range(reflection + 1):
            self.__logger.debug("calculation path for reflection = {}".format(i))
            C_1 = x1[0] - get_y_with_z_mirror(x1[1], C_0,self.medium.n_ice, self.__b, self.medium.delta_n, self.medium.z_0)[0]
            x2 = get_reflection_point(C_0, C_1,  self.medium.n_ice, self.reflection, self.__b, self.medium.z_0, self.medium.delta_n)
            if(x2[0] > x22[0]):
                x2 = x22
            yyy, zzz = self.get_path(x1, x2, C_0, n_points)
            yy.extend(yyy)
            zz.extend(zzz)
            self.__logger.debug("setting x1 from {} to {}".format(x1, x2))
            x1 = x2

        yy = np.array(yy)
        zz = np.array(zz)
        mask = yy > x11[0]
        return yy[mask], zz[mask]

    def obj_delta_y(self, logC_0, x1, x2, reflection=0, reflection_case=2):
        """
        function to find solution for C0, returns distance in y between function and x2 position
        result is signed! (important to use a root finder)
        """
        C_0 = get_C0_from_log(logC_0,self.medium.n_ice)
        return get_delta_y(C_0, np.array(x1), np.array(x2),self.medium.n_ice, self.__b, self.medium.delta_n, self.medium.z_0, self.reflection, (-1.0,-1.0), reflection, reflection_case)

    def determine_solution_type(self, x1, x2, C_0):
        """ returns the type of the solution

        Parameters
        ----------
        x1: 2dim np.array
            start position
        x2: 2dim np.array
            stop position
        C_0: float
            C_0 value of ray tracing solution

        Returns
        -------
        solution_type: int

            * 1: 'direct'
            * 2: 'refracted'
            * 3: 'reflected

        """
        c = self.medium.n_ice ** 2 - C_0 ** -2
        C_1 = x1[0] - get_y_with_z_mirror(x1[1], C_0, self.medium.n_ice, self.__b, self.medium.delta_n, self.medium.z_0)[0]
        gamma_turn, z_turn = get_turning_point(c, self.__b, self.medium.z_0, self.medium.delta_n)
        gamma_turn = gamma_turn[0]
        z_turn = z_turn[0]
        y_turn = get_y(gamma_turn, C_0, C_1, self.medium.n_ice, self.__b, self.medium.z_0)
        if(x2[0] < y_turn):
            return solution_types_revert['direct']
        else:
            if(z_turn == 0):
                return solution_types_revert['reflected']
            else:
                return solution_types_revert['refracted']

    def find_solutions(self, x1, x2, plot=False, reflection=0, reflection_case=1):
        """
        this function finds all ray tracing solutions

        prerequesite is that x2 is above and to the right of x1, this is not a violation of universality
        because this requirement can be achieved with a simple coordinate transformation

        Parameters
        ----------
        x1: tuple
            (y,z) coordinate of start point
        x2: tuple
            (y,z) coordinate of stop point
        reflection: int (default 0)
            how many reflections off the reflective layer (bottom of ice shelf) should be simulated


        returns an array of the C_0 paramters of the solutions (the array might be empty)

        """

        if(reflection > 0 and self.medium.reflection is None):
            self.__logger.error("a solution for {:d} reflection(s) off the bottom reflective layer is requested, but ice model does not specify a reflective layer".format(reflection))
            raise AttributeError("a solution for {:d} reflection(s) off the bottom reflective layer is requested, but ice model does not specify a reflective layer".format(reflection))

        if(self.use_cpp):
            tmp_reflection = copy.copy(self.medium.reflection)
            if(tmp_reflection is None):
                tmp_reflection = 100  # this parameter will never be used but is required to be an into to be able to pass it to the C++ module, so set it to a positive number, i.e., a reflective layer above the ice
            solutions = cpp_wrapper.find_solutions(x1, x2, self.medium.n_ice, self.medium.delta_n, self.medium.z_0, reflection, reflection_case, tmp_reflection)
            return solutions
        else:

            tol = 1e-6
            results = []
            C0s = []  # intermediate storage of results

            if(x2[1] > 0):
                # special case of ice to air ray tracing. There is always one unique solution between C_0 = inf and C_0 that
                # skims the surface. Therefore, we can find the solution using an efficient root finding algorithm.
                logC0_start = 100  # infinity is bad, 100 is steep enough
                C_0_stop = self.get_C_0_from_angle(np.arcsin(1/self.medium.get_index_of_refraction([0, x1[0], x1[1]])), x1[1]).x[0]
                logC0_stop = np.log(C_0_stop - 1/self.medium.n_ice)
                delta_ys = [self.obj_delta_y(logC0, x1, x2, reflection, reflection_case) for logC0 in [logC0_start, logC0_stop]]
                self.__logger.debug("Looking for ice-air solutions between log(C0) ({}, {}) with delta_y ({}, {})".format(logC0_start, logC0_stop, *delta_ys))
                if(np.sign(delta_ys[0]) == np.sign(delta_ys[1])):
                    self.__logger.warning(f"can't find a solution for ice/air propagation. The trajectory might be too vertical! This is currently not"\
                                          " supported because of numerical instabilities.")
                    return results
                result = optimize.brentq(self.obj_delta_y, logC0_start, logC0_stop, args=(x1, x2, reflection, reflection_case))

                C_0 = get_C0_from_log(result, self.medium.n_ice)
                C0s.append(C_0)
                solution_type = self.determine_solution_type(x1, x2, C_0)
                self.__logger.info("found {} solution C0 = {:.2f} (internal logC = {:.2f})".format(solution_types[solution_type], C_0, result))
                results.append({'type': solution_type,
                                'C0': C_0,
                                'C1': self.get_C_1(x1, C_0),
                                'reflection': reflection,
                                'reflection_case': reflection_case})
                return results

            # calculate optimal start value. The objective function becomes infinity if the turning point is below the z
            # position of the observer. We calculate the corresponding value so that the minimization starts at one edge
            # of the objective function
            # c = self.__b ** 2 / 4 - (0.5 * self.__b - np.exp(x2[1] / self.medium.z_0) * self.medium.n_ice) ** 2
            # C_0_start = (1 / (self.medium.n_ice ** 2 - c)) ** 0.5
            # R.L. March 15, 2019: This initial condition does not find a solution for e.g.:
            # emitter  at [-400.0*units.m,-732.0*units.m], receiver at [0., -2.0*units.m]

            if(self.__use_optimized_start_values):
                # take surface skimming ray as start value
                C_0_start, th_start = self.get_surf_skim_angle(x1)
                logC_0_start = np.log(C_0_start - 1. / self.medium.n_ice)
                self.__logger.debug(
                    'starting optimization with x0 = {:.2f} -> C0 = {:.3f}'.format(logC_0_start, C_0_start))
            else:
                logC_0_start = -1
            obj_delta_y_sqr = obj_delta_y_square
            result = optimize.root(obj_delta_y_sqr, x0=logC_0_start, args=(np.array(x1), np.array(x2),self.medium.n_ice,self.__b, self.medium.delta_n, self.medium.z_0, self.reflection, reflection, reflection_case), tol=tol)
            if(plot):
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(1, 1)
            if(result.fun < 1e-7):
                if(plot):
                    self.plot_result(x1, x2, get_C0_from_log(result.x[0], self.medium.n_ice), ax)
                if(np.round(result.x[0], 3) not in np.round(C0s, 3)):
                    C_0 = get_C0_from_log(result.x[0], self.medium.n_ice)
                    C0s.append(C_0)
                    solution_type = self.determine_solution_type(x1, x2, C_0)
                    self.__logger.info("found {} solution C0 = {:.2f}".format(solution_types[solution_type], C_0))
                    results.append({'type': solution_type,
                                    'C0': C_0,
                                    'C1': self.get_C_1(x1, C_0),
                                    'reflection': reflection,
                                    'reflection_case': reflection_case})

            # check if another solution with higher logC0 exists
            logC0_start = result.x[0] + 0.0001
            logC0_stop = 100
            delta_start = self.obj_delta_y(logC0_start, x1, x2, reflection, reflection_case)
            delta_stop = self.obj_delta_y(logC0_stop, x1, x2, reflection, reflection_case)
            if(np.sign(delta_start) != np.sign(delta_stop)):
                self.__logger.info("solution with logC0 > {:.3f} exists".format(result.x[0]))
                result2 = optimize.brentq(self.obj_delta_y, logC0_start, logC0_stop, args=(x1, x2, reflection, reflection_case))
                if(plot):
                    self.plot_result(x1, x2, get_C0_from_log(result2, self.medium.n_ice), ax)
                if(np.round(result2, 3) not in np.round(C0s, 3)):
                    C_0 = get_C0_from_log(result2, self.medium.n_ice)
                    C0s.append(C_0)
                    solution_type = self.determine_solution_type(x1, x2, C_0)
                    self.__logger.info("found {} solution C0 = {:.2f}".format(solution_types[solution_type], C_0))
                    results.append({'type': solution_type,
                                    'C0': C_0,
                                    'C1': self.get_C_1(x1, C_0),
                                    'reflection': reflection,
                                    'reflection_case': reflection_case})
            else:
                self.__logger.info("no solution with logC0 > {:.3f} exists".format(result.x[0]))

            logC0_start = -100
            logC0_stop = result.x[0] - 0.0001
            delta_start = self.obj_delta_y(logC0_start, x1, x2, reflection, reflection_case)
            delta_stop = self.obj_delta_y(logC0_stop, x1, x2, reflection, reflection_case)
            if(np.sign(delta_start) != np.sign(delta_stop)):
                self.__logger.info("solution with logC0 < {:.3f} exists".format(result.x[0]))
                result3 = optimize.brentq(self.obj_delta_y, logC0_start, logC0_stop, args=(x1, x2, reflection, reflection_case))

                if(plot):
                    self.plot_result(x1, x2, get_C0_from_log(result3, self.medium.n_ice), ax)
                if(np.round(result3, 3) not in np.round(C0s, 3)):
                    C_0 = get_C0_from_log(result3, self.medium.n_ice)
                    C0s.append(C_0)
                    solution_type = self.determine_solution_type(x1, x2, C_0)
                    self.__logger.info("found {} solution C0 = {:.2f}".format(solution_types[solution_type], C_0))
                    results.append({'type': solution_type,
                                    'C0': C_0,
                                    'C1': self.get_C_1(x1, C_0),
                                    'reflection': reflection,
                                    'reflection_case': reflection_case})
            else:
                self.__logger.info("no solution with logC0 < {:.3f} exists".format(result.x[0]))

            if(plot):
                import matplotlib.pyplot as plt
                plt.show()

            return sorted(results, key=itemgetter('reflection', 'C0'))

    def plot_result(self, x1, x2, C_0, ax):
        """
        helper function to visualize results
        """
        C_1 = self.get_C_1(x1, C_0)

        zs = np.linspace(x1[1], x1[1] + np.abs(x1[1]) + np.abs(x2[1]), 1000)
        yz = get_y_with_z_mirror(zs, C_0, self.medium.n_ice, self.__b, self.medium.delta_n, self.medium.z_0, C_1)
        yy = yz[0]
        zz = yz[1]
        ax.plot(yy, zz, '-', label='C0 = {:.3f}'.format(C_0))
        ax.plot(x1[0], x1[1], 'ko')
        ax.plot(x2[0], x2[1], 'd')

    #     ax.plot(zz, yy, '-', label='C0 = {:.3f}'.format(C_0))
    #     ax.plot(x1[1], x1[0], 'ko')
    #     ax.plot(x2[1], x2[0], 'd')
        ax.legend()

    def get_angle_from_C_0(self, C_0, z_pos, angoff=0, in_air=False):
        logC_0 = np.log(C_0 - 1. / self.medium.n_ice)
        return self.get_angle_from_logC_0(logC_0, z_pos, angoff, in_air=in_air)

    def get_angle_from_logC_0(self, logC_0, z_pos, angoff=0, in_air=False):

        '''
        argument angoff is provided so that the function can be used for minimization in get_C_0_from_angle(),
        in which case angoff is the angle for which the C_0 is sought and zero is returned when it is found.

        C_0 has a smallest possible value at 1./self.medium.n_ice . When it approaches this value, very
        small changes in C_0 correspond to a given change in the angle. In order to prevent the root finding
        algorithm from crossing into the invalid range of C_0 at values smaller than 1./self.medium.n_ice,
        the root finding is done with the parameter logC_0 = np.log(C_0 - 1. / self.medium.n_ice), so it is
        not exactly the log of C_0 as the nome of this method implies.
        This is the same parameter transformation that is done for find_solutions()

        input:
            logC_0 = np.log(C_0 - 1. / self.medium.n_ice)
            angoff = angular offset
            z_pos  = z-position from where ray is emitted
        output:
            angle corresponding to C_0, minus offset angoff
        '''

        C_0 = get_C0_from_log(logC_0, self.medium.n_ice)

        dydz = get_y_diff(z_pos, C_0, self.medium.n_ice, self.__b, self.medium.z_0, self.medium.delta_n, in_air=in_air)
#        dydz = self.get_dydz_analytic(C_0, z_pos)
        angle = np.arctan(dydz)


        return angle - angoff

    def get_C_0_from_angle(self, anglaunch, z_pos, in_air=False):

        '''
        Find parameter C0 corresponding to a given launch angle and z-position of a ray.
        The parameter is found by means of a root finding procedure

        output:
            Complete output of optimisation procedure
            (result.x[0] is the C0 value found by optimisation procedure)

        '''

        # C_0 has a smallest possible value at 1./self.medium.n_ice . When it approaches this value, very
        # small changes in C_0 correspond to given change in the angle. In order to prevent the root finding
        # algorithm to cross into the invalid range of C_0 at  values smaller than 1./self.medium.n_ice,
        # the root finding is done with the parameter logC_0_start below. This is the same parameter transformation
        # that is done for find_solutions()

        C_0_start = 2.

        logC_0_start = np.log(C_0_start - 1. / self.medium.n_ice)

#        result = optimize.root(self.get_angle_from_C_0,np.pi/4.,args=(z_pos,anglaunch))
        result = optimize.root(self.get_angle_from_logC_0, logC_0_start, args=(z_pos, anglaunch, in_air))

        # want to return the complete instance of the result class; result value result.x[0] is logC_0,
        # but we want C_0, so replace it in the result class. This may not be good practice but it seems to be
        # more user-friendly than to return the value logC_0
        result.x[0] = copy.copy(get_C0_from_log(result.x[0], self.medium.n_ice))

        return result

#     def get_dydz_analytic(self, C_0, z_pos):
#         '''
#         Implementation of derivative dy/dz obtained from the analytic expresion for y(z)
#         Returns dy/dz for a given z-position and C_0
#         '''
#
#         gamma = self.get_gamma(z_pos)
#
#         b = self.__b
#         c = self.medium.n_ice ** 2 - C_0 ** -2
#         root = np.abs(gamma ** 2 - gamma * b + c)
#         logargument = gamma / (2 * c ** 0.5 * (root) ** 0.5 - b * gamma + 2 * c)
#
#         dydz = 1/(C_0*np.sqrt(c))*(1 - np.sqrt(c)/np.sqrt(root)*(2*gamma-b)*logargument + b*logargument)
#
#         return dydz

    def get_z_from_n(self, n):
        '''
        get z from given n - equation from get_n solved for z
        '''

        return np.log((self.medium.n_ice - n) / self.medium.delta_n) * self.medium.z_0

    def get_surf_skim_angle(self, x1):

        '''
        For a given position x1 = [x,z] and depth profile self.n(), find the angle at which a beam must be
        emitted to "skim the surface", i.e. arrive horizontally (angle = 90 deg) at the surface;
        This is used to find the refraction zone.

        Returns
        -------
        C0crit: float
            C0 of critical angle
        thcrit: float
            critical angle
        '''

        nlaunch = n(x1[1], self.medium.n_ice, self.medium.delta_n, self.medium.z_0)
        # by definition, z of critical angle is at surface, i.e. z=0
        zcrit = 0.
        nsurf = n(zcrit, self.medium.n_ice, self.medium.delta_n, self.medium.z_0)

        sinthcrit = nsurf / nlaunch
        if sinthcrit <= 1:
            # ray goes from point with high optical thickness to point with lower optical thickness,
            # i.e. ray bending is towards horizontal
            thcrit = np.arcsin(sinthcrit)
            C0result = self.get_C_0_from_angle(thcrit, x1[1])
            C0crit = C0result.x[0]
        else:
            # ray goes from point with low optical thickness to point with higher optical thickness,
            # i.e. ray bending is towards vertical, no solution
            thcrit = None
            C0crit = None
            self.__logger.warning(' No solution for critical angle for z = {}!'.format(x1[1]))
        self.__logger.info(' critical angle for z = {} is {} !'.format(x1[1], thcrit / units.deg))
        self.__logger.info(' C0 for critical angle is {}'.format(C0crit))

        return C0crit, thcrit

    def is_in_refraction_zone(self, x1, x2, C0crit=None, plot=False):
        '''
        Find if receiver at x2 is in the refraction zone of emitter at x1. The refraction zone
        is the oposite of the shadow zone.

        If the C0 of the critical angle, C0crit, is provided, it will not be calculated. This is useful
        in case find_solutions() is called and C0crit is calculated in the process of determining the
        initial value for the minimization procedure.

        Returns True if x2 is in the refraction zone of x1 - note that the inverse statement is not
        necessarily true, i.e. when False is returned, it is possible that x2 is in the refraction
        zone nonetheless

        TODO:
        Why does the reference point not seem to lie exactly on the mirrored path?
        Instead of returning True/False, it might be useful to return  ycheck - x2[0] (in case x2[0]>ycrit),
        which gives some idea of how close the receiver is to the refraction zone. This could be used to
        define a "gray zone' and a 'far zone', in which the receiver is most definitely in the shadow zone
        '''

        refraction = False

        if C0crit == None:
            C0crit, thcrit = self.get_surf_skim_angle(x1)
        # z_crit = 0 and hence gamma_crit = delta_n by definition
        gcrit = self.medium.delta_n
        # the y-value where the ray hits z=0
        ycrit = get_y(gcrit, C0crit, self.get_C_1(x1, C0crit), self.medium.n_ice, self.__b, self.medium.z_0)

        if plot:
            import matplotlib.pyplot as plt
            plt.figure('in_refraction_zone')
            plt.grid(True)
            plt.plot(ycrit, 0, 'ro', label='turning point')
            yarray, zarray = self.get_path(x1, [ycrit, 0], C0crit)
            plt.plot(yarray, zarray, 'ko-', markersize=4, label='path')
            plt.plot(x1[0], x1[1], 'C1d', label='emitter')
            plt.plot(x2[0], x2[1], 'go', label='receiver')

        if x2[0] <= ycrit:
            # not in shadow zone
            refraction = True
            self.__logger.debug(' is_in_refraction_zone(): y-position of receiver smaller than ycrit!')
        else:
            # start horizontal ray at (y,z) = (ycrit,0)
            # experimentally this was found to give slightly different results than mirroring the array at the critical angle.
            # theoretically this is not quite unterstood
            C0check = self.get_C_0_from_angle(np.pi / 2., 0)
            C0check = C0check.x[0]
            gcheck = get_gamma(x2[1], self.medium.delta_n, self.medium.z_0)
            ycheck = -get_y(gcheck, C0check, self.get_C_1([ycrit, 0], C0check), self.medium.n_ice, self.__b, self.medium.z_0) + 2 * ycrit

            if x2[0] < ycheck:
                refraction = True
            if plot:
                yarraymirr = -yarray + 2 * ycrit
                plt.plot(yarraymirr, zarray, 'mx-', markersize=4, label='mirrored path')
                # the reference point does not seem to lie exactly on the mirrored path but instead
                # ~1cm inside the path (i.e. towards the emmitter) which I do not understand.
                plt.plot(ycheck, x2[1], 'b+', label='reference point')

        if plot:
            plt.legend(fontsize='x-small')

        return refraction

    def get_tof_for_straight_line(self, x1, x2):
        '''
        Calculate the time of flight for a hypothatical ray travelling from x1 to x2 in a straight line.
        Such an array in general is not a solution consistant with Fermat's principle. It is however
        useful as a reference time or approximation for signals not explicable with geometric optics.

        output:
            time of flight for a ray travelling straight from x1 to x2
        '''

        dx = x2[0] - x1[0]
        dz = x2[1] - x1[1]
        n_ice = self.medium.n_ice
        delta_n = self.medium.delta_n
        z_0 = self.medium.z_0

        if dz > 0:
            return 1. / constants.c * np.sqrt((dx / dz) ** 2 + 1) * (
            n_ice * dz - delta_n * z_0 * (np.exp(x2[1] / z_0) - np.exp(x1[1] / z_0))
            )
        else:
            return n(x2[1], self.medium.n_ice, self.medium.delta_n, self.medium.z_0) / constants.c * dx

    def get_surface_pulse(self, x1, x2, infirn=False, angle='critical', chdraw=None, label=None):

        '''
        Calculate the time for a ray to travel from x1 to the surface and arriving at the surface
        with (a) critical angle or (b) Brewster angle, propagating in a straight line along the surface
        in air (n=1) in the firn at z=0 (n=n(0, self.medium.n_ice, self.medium.delta_n, self.medium.z_0)) and then reaching the receiver by returning into the
        firn just at the point to reach the receiver at x2, entering the firn at the surface at the same
        angle it reached the surface from x1..

        Parameters
        ----------
        x1, x2: arrays
            Arrays with x and z positions of emitter x1 and receiver x2
        infirn: Boolean.
            Set to True if surface ray travels in the firn, set to False (default) if it travels
            in air.
        angle:  String
            specifying angle at which ray reaches/leaves the surface. Can be 'Brewster' or 'critical'
            If neither of these is chosen, a warning is printed and angle is set to 'critical'
        chdraw: string or None
            If None, do not draw the path of the ray. If the ray should be drawn, a string consistent with
            the matplotlib.pyplot library has to be specified, e.g. 'r:' to draw a dotted red line.
            It is assumed that an appropriate figure on which to draw the ray has been created and set as
            current figure by the user before calling this method.
        label:  Label for plot
        '''

        draw = False
        if chdraw != None:
            draw = True

        if infirn == False:
            nlayer = 1.  # index of refraction at surface, default is n=1 for air
        else:
            nlayer = n(0, self.medium.n_ice, self.medium.delta_n, self.medium.z_0)

        if angle == 'critical':
            # sin(th)=1,
            nxsin = 1.
        elif angle == 'Brewster':
            nxsin = np.sin(np.arctan(1. / n(0, self.medium.n_ice, self.medium.delta_n, self.medium.z_0))) * n(0, self.medium.n_ice, self.medium.delta_n, self.medium.z_0)
        else:
            self.__logger.warning(' unknown input angle=={}, using critical angle!!!'.format(angle))
            nxsin = 1.

        zsurf = 0
        gamma = get_gamma(zsurf, self.medium.delta_n, self.medium.z_0)

        # find emission angle for starting point x1 to hit the surface at the specified angle

        # look at time and distance it takes for the signal to travel from the emitter to the surface
        # and from the surface to the receiver
        tice = 0
        sice = 0
        for x in [x1, x2]:
            sinthemit = nxsin / n(x[1], self.medium.n_ice, self.medium.delta_n, self.medium.z_0)
            th_emit = np.arcsin(sinthemit)
            C0result = self.get_C_0_from_angle(th_emit, x[1])
            C0_emit = C0result.x[0]

            self.__logger.info(' emission angle for position {},{} is theta_emit= {}'.format(x[0], x[1], th_emit / units.deg))

            # x-coordinate where ray reaches surface; is always bigger than the x-position of the emitter
            # (i.e. ray travels "to the right")
            xsurf = get_y(gamma, C0_emit, self.get_C_1(x, C0_emit), self.medium.n_ice, self.__b, self.medium.z_0)
            sice += xsurf - x[0]
            self.__logger.info(' air pulse starting at x={}, z={} reaches surface at x={}'.format(x[0], x[1], xsurf))
            ttosurf = self.get_travel_time_analytic(x, [xsurf, zsurf], C0_emit)
            tice += ttosurf
            self.__logger.info(' travel time is {} ns.'.format(ttosurf / units.ns))

            if draw:
                import matplotlib.pyplot as plt
                z = np.linspace(x[1], zsurf, 1000, endpoint=True)
                y = get_y(get_gamma(z, self.medium.delta_n, self.medium.z_0), C0_emit, self.medium.n_ice, self.__b, self.medium.z_0, C_1=self.get_C_1(x, C0_emit))
                if x == x1:
                    ysurf = [y[-1]]
                else:
                    y = -y + 2 * x2[0]
                    ysurf.append(y[-1])
                    plt.plot(ysurf, [0, 0], chdraw, label=label)

                plt.plot(y, z, chdraw)

        self.__logger.info(' time, distance travelled to and from surface: {}, {} '.format(tice, sice))

        sair = abs(x2[0] - x1[0]) - sice
        tair = sair * nlayer / constants.c
        self.__logger.info(' time, distance travelled at surface: {}, {}'.format(tair, sair))
        ttot = tice + tair
        if sair < 0:
            ttot = None
        return ttot

    def angular_diff(self, x_refl, z_refl, pulser_pos, receiver_pos, ipulssol, irxsol):

        '''
        This is a helper function to find a ray that is subject to specular reflection (no transmission) at the
        ice-water interface, e.g. for Moore's Bay. For a (virtual) emitter positioned at [x_refl,z_refl], it finds the
        emission angles such that the rays hit positions pulser_pos and receiver_pos. ipulssol = 0 or 1, respectively
        means that pulser_pos is hit directly or by means of reflection at the surface, respectively.
        irxsol is the equivalent parameter for receiver_pos.

        angular_diff can be used as the function to find a root of in the following fashion:

        result = optimize.root(raytr.angular_diff, x0=x_refl_start, args=(z_refl,pulser_pos,receiver_pos,ipulssol,irxsol))

        Then get the final value by x_refl = result.x[0] (z_refl is fixed)

        The idea is to treat the reflection point as a virtual emitter and then find the x-position at the predefined
        depth z_refl, for which the emisssion angles to pulser_pos and receiver_pos are the same (i.e. the output
        of angular_diff is zero). The x-position would be output "result" ofoptimize.root() above.

        Returns
        -------
        result: float
            Is zero if the angles (w.r.t. the vertical) of rays emitted from [x_refl,z_refl] to
            positions pulser_pos and receiver_pos are the same or greater than zero, if this is not the case.
            For exact defintion, see "result" in code below
        '''

        # treat position of reflection as emitter and Rx/Tx as receivers
        pos_rx = [
            [receiver_pos[0], receiver_pos[1]],
            [x_refl - (pulser_pos[0] - x_refl), pulser_pos[1]]
        ]
        beta0 = None
        beta1 = None
        # solution for receiver
        solution0 = self.find_solutions([x_refl, z_refl], pos_rx[0], plot=False)
        if solution0 != []:
            C0rx = solution0[irxsol]['C0']
            beta0 = self.get_launch_angle([x_refl, z_refl], C0rx)
        # solution for pulser
        solution1 = self.find_solutions([x_refl, z_refl], pos_rx[1], plot=False)
        if solution1 != []:
            C0puls = solution1[ipulssol]['C0']
            beta1 = self.get_launch_angle([x_refl, z_refl], C0puls)

        if beta0 != None and beta1 != None:
            result = (np.tan(beta0) - np.tan(beta1)) ** 2
        else:
            result = np.inf  # set to infinity

        return result
