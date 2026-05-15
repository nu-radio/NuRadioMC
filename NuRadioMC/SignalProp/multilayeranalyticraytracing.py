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
to have a look at the companion note to this module ``MultilayerAnalyticRayTracting.md`` (will be added in the future) and the appendix C of
"NuRadioMC: Simulating the Radio Emission of Neutrinos from Interaction to Detector“ (2020). https://doi.org/10.1140/epjc/s10052-020-7612-8.
This appendix describes the implementation of the previously used single layer analytic raytracer.


Notes
-----
Coordinates are given as (y, z) with units of meters:

* y : horizontal distance
* z : vertical coordinate

z = 0 corresponds to the ice surface.

The ray parameter ``C0`` determines the curvature of the trajectory.
It can be seen as C0 = 1/(n(z)*sin(theta)) where n(z) is the refractive index and theta is the angle relative to the horizontal at the current depth z.

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
from scipy import optimize
from operator import itemgetter
#from numba import njit
#from numba.typed import List
from functools import lru_cache
from NuRadioMC.SignalProp.propagation import solution_types, solution_types_revert
from NuRadioMC.utilities import attenuation

from NuRadioReco.utilities import units, geometryUtilities, constants
#from NuRadioMC.utilities import attenuation as attenuation_util, medium as medium_util
from NuRadioMC.SignalProp.propagation_base_class import ray_tracing_base

from math import sqrt, log, sin

import logging
logger = logging.getLogger("NuRadioMC.analytic_ray_tracing")


# ------------------------------
# Layer definitions
# ------------------------------
LAYERS_SINGLE = [{
    "n_ice": 1.78,
    "delta_n": 0.51,
    "z_0": 37.25,
    "z_min": -3000.0,
    "z_max": 0.0,
    "region": "single",
    "region_name": "SingleModel"
}]

LAYERS_FIRN = [
    {
        "z_min": -14.9,
        "z_max": 0.0,
        "n_ice": 1.78,
        "delta_n": 0.502,
        "z_0": 30.8,
        "region": "firn",
        "region_name": "Firn"
    },
    {
        "z_min": -3000.0,
        "z_max": -14.9,
        "n_ice": 1.78,
        "delta_n": 0.446,   # converted from shifted exponential from the definition in greenland_firn by multiplying delta_n with exp(-z_shift/z_0) to adapt to our case. 
                            # However, it might be worth to look into easier model defintions to simplify the boundary C0 in the minmizing process
        "z_0": 40.9,
        "region": "ice",
        "region_name": "Ice"
    }]

LAYERS = [
    {
        "z_min": -14.9,
        "z_max": 0.0,
        "n_ice": 1.51188,
        "delta_n": 0.271579,
        "z_0": 1/0.114553,
        "region": "snow",
        "region_name": "Snow"
    },
    {
        "z_min": -80.5,
        "z_max": -14.9,
        "n_ice": 1.89957,
        "delta_n": 0.529715,
        "z_0": 1/0.0129175,
        "region": "firn",
        "region_name": "Firn"
    },
    {
        "z_min": -3000.0,
        "z_max": -80.5,
        "n_ice": 1.77468,
        "delta_n": 1.41573,
        "z_0": 1/0.0387882,
        "region": "bubbly_ice",
        "region_name": "Ice"
    }]

LAYERS_AIR = [

        {
        "z_min": 0.0,
        "z_max": np.inf,
        "n_ice": 1.00027,
        "delta_n": 2.7e-4,
        "z_0": -8000.0,
        "region": "air",
        "region_name": "Air"
    },
    {
        "z_min": -14.9,
        "z_max": 0.0,
        "n_ice": 1.51188,
        "delta_n": 0.271579,
        "z_0": 1/0.114553,
        "region": "snow",
        "region_name": "Snow"
    },
    {
        "z_min": -80.5,
        "z_max": -14.9,
        "n_ice": 1.89957,
        "delta_n": 0.529715,
        "z_0": 1/0.0129175,
        "region": "firn",
        "region_name": "Firn"
    },
    {
        "z_min": -3000.0,
        "z_max": -80.5,
        "n_ice": 1.77468,
        "delta_n": 1.41573,
        "z_0": 1/0.0387882,
        "region": "bubbly_ice",
        "region_name": "Ice"
    }]

LAYERS_TEST = [
    {
        "z_min": 0.0,
        "z_max": np.inf,
        "n_ice": 1.0,
        "delta_n": 0.000001,
        "z_0": -500.0,
        "region": "air",
        "region_name": "Air"
    },
    {
        "z_min": -14.9,
        "z_max": 0.0,
        "n_ice": 1.51188,
        "delta_n": 0.271579,
        "z_0": 1/0.114553,
        "region": "snow",
        "region_name": "Snow"
    },
    {
        "z_min": -80.5,
        "z_max": -14.9,
        "n_ice": 1.89957,
        "delta_n": 0.529715,
        "z_0": 1/0.0129175,
        "region": "firn",
        "region_name": "Firn"
    },
    {
        "z_min": -300.0,
        "z_max": -80.5,
        "n_ice": 1.77468,
        "delta_n": 1.41573,
        "z_0": 1/0.0387882,
        "region": "upper_ice",
        "region_name": "Upper Ice"
    },
    {
        "z_min": -3000.0,
        "z_max": -300.0,
        "n_ice": 1.9468,
        "delta_n": 1.41573,
        "z_0": 1/0.0387882,
        "region": "lower_ice",
        "region_name": "Lower Ice"
    }]

def layers_to_arrays(layers):
    """
    Convert layer definitions from dictionaries to NumPy arrays.

    The Numba implementation of the ray tracing solver requires all
    layer parameters to be stored in contiguous arrays rather than
    Python dictionaries. This helper function performs that conversion.

    Parameters
    ----------
    layers : list of dict
        List of layer definitions.

    Returns
    -------
    tuple of ndarray
        Arrays describing the layer parameters:

        z_min : ndarray
            Lower depth boundary of each layer.

        z_max : ndarray
            Upper depth boundary of each layer.

        n_ice : ndarray
            Asymptotic refractive index in each layer.

        delta_n : ndarray
            Refractive index contrast.

        z0 : ndarray
            Exponential scale depth. 
    """
    n = len(layers)
    z_min = np.zeros(n)
    z_max = np.zeros(n)
    n_ice = np.zeros(n)
    delta_n = np.zeros(n)
    z0 = np.zeros(n)

    for i,L in enumerate(layers):
        z_min[i] = L["z_min"]
        z_max[i] = L["z_max"]
        n_ice[i] = L["n_ice"]
        delta_n[i] = L["delta_n"]
        z0[i] = L["z_0"]

    return z_min, z_max, n_ice, delta_n, z0


def init_layer_arrays(z_min, z_max, n_ice, delta_n, z0):
    """
    Validate and return layer arrays.

    All inputs must be array-like with same length.

    Parameters
    ----------
        tuple of ndarray
        Arrays describing the layer parameters:

        z_min : ndarray
            Lower depth boundary of each layer.

        z_max : ndarray
            Upper depth boundary of each layer.

        n_ice : ndarray
            Asymptotic refractive index in each layer.

        delta_n : ndarray
            Refractive index contrast.

        z0 : ndarray
            Exponential scale depth. 
    """
    z_min = np.asarray(z_min, dtype=float)
    z_max = np.asarray(z_max, dtype=float)
    n_ice = np.asarray(n_ice, dtype=float)
    delta_n = np.asarray(delta_n, dtype=float)
    z0 = np.asarray(z0, dtype=float)

    n = len(z_min)

    if not (len(z_max) == len(n_ice) == len(delta_n) == len(z0) == n):
        raise ValueError("All input arrays must have the same length")

    return z_min, z_max, n_ice, delta_n, z0

#@njit(cache = True)
def get_layer_index(z, z_min, z_max):
    """
    Determine the layer index corresponding to a given depth.

    Parameters
    ----------
    z : float
        Depth coordinate.

    z_min, z_max : ndarray
        Arrays containing the lower and upper boundaries of each layer.

    Returns
    -------
    int
        Index of the layer containing the given depth.

    Notes
    -----
    Returns ``-1`` if the depth lies outside the defined layer ranges.
    """
    z = float(z)
    for i in range(len(z_min)):

        if z >= z_min[i] and z <= z_max[i]:
            return i
        
    return -1

#@njit(cache = True)
def analytic_F(z, C0, n_ice, delta_n, z0):
    """
    Evaluate the analytic ray integral F(z) for an exponential index profile.

    This function represents the analytic solution of the horizontal
    ray displacement in a medium with exponential refractive index

        n(z) = n_ice - delta_n * exp(z / z0)

    Parameters
    ----------
    z : float
        Depth coordinate.

    C0 : float
        Ray parameter (inverse horizontal slowness).

    n_ice : float
        Asymptotic refractive index of the layer.

    delta_n : float
        Refractive index contrast.

    z0 : float
        Exponential scale depth.

    Returns
    -------
    float
        Value of the analytic integral F(z).

    Notes
    -----
    The horizontal coordinate of the ray trajectory is given by

        y(z) = F(z) + C1

    where C1 is a layer-dependent offset constant. 
    The offsets are defined from the boundary conditions (e.g. x_start) and can be calculated with 
    the calculate_offsets function.
    """
    z = float(z)
    C0 = float(C0)
    n_ice = float(n_ice)
    delta_n = float(delta_n)
    z0 = float(z0)
    b = 2.0 * n_ice
    n = n_ice - delta_n*np.exp(z / z0)
    c = max(abs(n_ice*n_ice - 1.0/(C0*C0)),1e-14)

    # F only valid for positive c
    if c < 0 :
        return np.nan
    
    
    gamma = delta_n * np.exp(z / z0)
    root = np.abs(gamma*gamma - gamma*b + c)
    
    logargument = gamma / (2.0*np.sqrt(c)*np.sqrt(root) - b*gamma + 2.0*c)

    val = z0 * (n_ice*n_ice*C0*C0 - 1.0)**-0.5 * np.log(logargument)

    return float(np.real(val))

#@njit(cache = True)
def compute_offsets(C0, y_start, z_start, layers, get_intersection_point = False):
    """
    Compute horizontal offset constants for all layers.

    The ray trajectory is expressed as

        y(z) = F(z) + C1

    where the constant ``C1`` differs between layers. This function
    determines the offsets required to ensure continuity of the
    trajectory across layer boundaries.

    Parameters
    ----------
    C0 : float
        Ray parameter.

    y_start, z_start : float
        Starting position of the ray.

    layers : tuple of ndarray
        Layer parameter arrays.

    Returns
    -------
    C1 : ndarray
        Offset constants for each layer.

    idx_start : int
        Index of the layer containing the starting depth.

    Notes
    -----
    Offsets are propagated upward through the layer stack to enforce
    continuity of the ray path going through the chosen starting point.
    """
    z_min, z_max, n_ice, delta_n, z0 = layers
    n_layers = len(z_min)
    C1 = np.zeros(n_layers)
    C0 = float(C0)
    y_start = float(y_start)
    z_start = float(z_start)
    idx_start = -1

    for i in range(n_layers):

        if z_start >= z_min[i] and z_start <= z_max[i]:
            idx_start = i
            break
    
    F_start = analytic_F(z_start, C0, n_ice[idx_start], delta_n[idx_start], z0[idx_start])
    C1[idx_start] = float(y_start - F_start)

    if get_intersection_point is True:
        ybs = np.zeros(n_layers-1) 
        zbs = np.zeros(n_layers-1)

    for i in range(idx_start - 1, -1, -1):

        zb = float(z_min[i])
        F_deep = analytic_F(zb, C0, n_ice[i+1], delta_n[i+1], z0[i+1])
        yb = float(F_deep + C1[i+1])

        if get_intersection_point is True:
            ybs[i]=yb
            zbs[i]=zb

        F_shallow = analytic_F(zb, C0, n_ice[i], delta_n[i], z0[i])
        C1[i] = float(yb - F_shallow)

    if get_intersection_point is True:
        return C1, idx_start, ybs, zbs
    else:
        return C1, idx_start, np.zeros(n_layers -1), np.zeros(n_layers -1)

#@njit(cache = True)
def build_y_field(C0, z_array, layers, C1):
    """
    Evaluate the horizontal ray trajectory y(z).

    Parameters
    ----------
    C0 : float
        Ray parameter.

    z_array : ndarray
        Depth coordinates at which the trajectory should be evaluated.

    layers : tuple of ndarray
        Layer parameter arrays.

    C1 : ndarray
        Offset constants for each layer.

    Returns
    -------
    y : ndarray
        Horizontal coordinates corresponding to ``z_array``.

    layer_idx : ndarray
        Layer index for each depth.
    """
    z_min, z_max, n_ice, delta_n, z0 = layers
    n = len(z_array)
    y = np.zeros(n)
    layer_idx = np.zeros(n, dtype=np.int64)
    C0 = float(C0)

    for j in range(n):

        z = float(z_array[j])
        idx = get_layer_index(z, z_min, z_max)
        layer_idx[j] = idx
        F = analytic_F(z, C0, n_ice[idx], delta_n[idx], z0[idx])
        y[j] = float(F + C1[idx])

    return y, layer_idx

#@njit(cache = True)
def evaluate_y(C0, C1, z, layers):
    """
    Evaluate the horizontal ray coordinate at a given depth.

    Parameters
    ----------
    C0 : float
        Ray parameter.

    C1 : ndarray
        Offset constants for each layer.

    z : float
        Depth coordinate at which the trajectory should be evaluated.

    layers : tuple of ndarray
        Layer parameter arrays.

    Returns
    -------
    float
        Horizontal coordinate y(z).

    Notes
    -----
    The ray trajectory within a layer is described by

        y(z) = F(z) + C1

    where F(z) is the analytic ray integral and C1 is a
    layer-dependent offset chosen to ensure continuity
    across layer boundaries.
    """
    z = float(z)
    C0 = float(C0)
    z_min, z_max, n_ice, delta_n, z0 = layers
    idx = get_layer_index(z, z_min, z_max)
    F = analytic_F(z, C0, n_ice[idx], delta_n[idx], z0[idx])

    return float(F + C1[idx])


#@njit(cache=True)
def find_z_turn(C0, layers):
    """
    Determine the depth of the ray turning point.

    A turning point occurs where the refractive index satisfies

        n(z) = 1 / C0

    which corresponds to horizontal propagation of the ray.

    Parameters
    ----------
    C0 : float
        Ray parameter.

    layers : tuple of ndarray
        Layer parameter arrays.

    Returns
    -------
    float
        Depth coordinate of the turning point.

    Notes
    -----
    If no turning point exists within the medium, the function returns
    the maximum depth boundary of the defined layers.
    """
    eps = 1e-12

    z_min, z_max, n_ice, delta_n, z0 = layers
    target_n = 1.0 / float(C0)

    best_z = np.inf
    found = False

    for i in range(len(z_min)):
        val = (n_ice[i] - target_n) / delta_n[i]

        # must be positive for log
        if val <= 0.0:
            continue

        z = z0[i] * np.log(val)

        if (z_min[i] - eps) <= z <= (z_max[i]):
            if z < best_z:
                best_z = z
                found = True

    if found:
        return best_z

    return np.max(z_max)

#@njit(cache = True)
def get_turning_point(C0, y_start, z_start, layers, C1=None,
                      downgoing=False, with_air=False):
    """
    Compute the coordinates of the turning point (y, z) of a ray with C0 starting from x1=(y_start,z_start).

    Parameters
    ----------
    C0 : float
        Ray parameter.

    y_start, z_start : float
        Starting coordinates of the ray.

    layers : tuple of ndarray
        Layer parameter arrays.

    C1 : ndarray, optional
        Precomputed layer offsets.

    downgoing : bool, optional
        Flag indicating whether the original ray propagates downward.

    with_air : bool, optional
        Flag indicating whether the propagation includes an air layer.

    Returns
    -------
    y_turn : float
        Horizontal coordinate of the turning point.

    z_turn : float
        Depth coordinate of the turning point.

    Notes
    -----
    To avoid numerical issues that appeared when z_turn is set to 0.0 we shift the surface reflection slightly downwards.
    Otherwise this could cause the evaluation at z=0.0 with the layer parameters of the air layer and break the logic.
    If the ray reaches the surface and ``with_air`` is False, the
    turning point is clamped to the surface depth.
    """

    if C1 is None:
        # Compute offsets once
        C1, _, _, _ = compute_offsets(C0, y_start, z_start, layers)
    
    # Find depth of turning point
    z_turn = find_z_turn(C0, layers)
    
    numerical_safety_offset = 1e-12
    z_surface = 0.0 - numerical_safety_offset

    if (z_turn >= z_surface) and not with_air:
            z_turn = z_surface
    
    # Evaluate horizontal coordinate at turning point
    y_turn = evaluate_y(C0, C1, z_turn, layers)
    
    return y_turn, z_turn

#@njit(cache = True)
def get_delta_y(C0, y1, z1, y2, z2, layers, C0range,
                downgoing, with_air):
    """
    Compute horizontal mismatch between a ray and a target point.

    This function evaluates how far a ray with parameter ``C0`` deviates
    from the desired receiver position. It serves as the objective
    function for root-finding during ray solution searches.

    Parameters
    ----------
    C0 : float
        Ray parameter.

    y1, z1 : float
        Starting position.

    y2, z2 : float
        Target position.

    layers : tuple of ndarray
        Layer parameter arrays.

    C0range : tuple
        Allowed range of ray parameters.

    downgoing : bool
        Flag indicating reversed geometry.

    with_air : bool
        Flag indicating propagation through air.

    Returns
    -------
    float
        Horizontal difference between predicted and target position.

    Notes
    -----
    The sign of the returned value determines which side of the
    receiver the ray endpoint lies on and is therefore used by
    root-finding algorithms.
    """
    C0 = float(C0)
    y1 = float(y1)
    z1 = float(z1)
    y2 = float(y2)
    z2 = float(z2)
    z_min, z_max, n_ice, delta_n, z0 = layers

    if C0range[0] == -1.0 and C0range[1] == -1.0:
        C0range = (1. / n_ice[-1], np.inf)

    if C0 < C0range[0] or C0 > C0range[1]:
        return -np.inf
    
    C1, _ , _, _= compute_offsets(C0, y1, z1, layers)
    y_turn, z_turn = get_turning_point(C0,y1,z1,layers,C1,downgoing,with_air)

    
    if (y_turn is not None) and (z_turn is not None):

        if z_turn <= z2:
            dz = z_turn - z2
            dy = y_turn - y2
            diff = np.sqrt(dz*dz + dy*dy) + 10.0 * np.abs(dz)
            return -diff
        
        elif y_turn >= y2:
            y_fit = evaluate_y(C0, C1, z2, layers)
            return y2 - y_fit
        
        else:
            y_raw = evaluate_y(C0, C1, z2, layers)
            y_fit = 2.0*y_turn - y_raw
            return -(y2 - y_fit)
        
    elif (y_turn is None) and (z_turn is None): 
        y_fit = evaluate_y(C0, C1, z2, layers)
        return y2 - y_fit

#@njit(cache = True)
def get_refractive_index(z, layers):
    """
    Evaluate the refractive index at a given depth.

    Parameters
    ----------
    z : float
        Depth coordinate.

    layers : tuple of ndarray
        Layer parameter arrays.

    Returns
    -------
    float
        Refractive index n(z).

    Notes
    -----
    Within each layer the refractive index follows

        n(z) = n_ice - delta_n * exp(z / z0)
    """
    z_min, z_max, n_ice, delta_n, z0 = layers
    idx = get_layer_index(z, z_min, z_max)

    return n_ice[idx] - delta_n[idx] * np.exp(z / z0[idx])

#@njit(cache = True)
def get_C0_from_theta(z_start, theta, layers):
    """
    Convert a launch angle to the corresponding ray parameter.

    Parameters
    ----------
    z_start : float
        Starting depth of the ray.

    theta : float
        Launch angle in radians.

    layers : tuple of ndarray
        Layer parameter arrays.

    Returns
    -------
    float
        Ray parameter C0.
    """
    # Convert launch angle to ray parameter
    n_start = get_refractive_index(z_start, layers)
    p = n_start * np.cos(theta)

    if p == 0.0:
        return 1e12  # avoid division by zero
    
    return 1.0 / p

#@njit(cache = True)
def get_skim_angle(y1, z1, zskim, layers):
    """
    Compute the critical launch angle for a ray that skims a given depth.

    The resulting ray reaches the specified depth with a horizontal
    propagation angle.

    Parameters
    ----------
    y1, z1 : float
        Starting coordinates of the ray.

    zskim : float
        Depth of the plane to skim.

    layers : tuple of ndarray
        Layer parameter arrays.

    Returns
    -------
    C0crit : float
        Ray parameter corresponding to the critical launch angle.

    thcrit : float
        Critical launch angle in radians.

    Notes
    -----
    The critical angle is determined from the refractive index
    contrast between the launch depth and the skim depth.
    """
    nlaunch = get_refractive_index(z1, layers)
    nsurf = get_refractive_index(zskim, layers)
    
    sinthcrit = min(nsurf / nlaunch, 0.999999999)
    if sinthcrit <= 1.0:
        thcrit = np.arcsin(sinthcrit)
        C0crit = get_C0_from_theta(z1, thcrit, layers)
    else:
        thcrit = 1e-12  # nearly zero angle
        C0crit = -1.0

    return C0crit, thcrit


def get_C0_from_log_scalar(logC0, n_ice):
    """
    Convert the logarithmic optimization parameter to a ray parameter.

    Parameters
    ----------
    logC0 : float
        Optimization parameter used during root finding.

    n_ice : float
        Refractive index of deep ice.

    Returns
    -------
    float
        Ray parameter C0.

    Notes
    -----
    The transformation

        C0 = exp(logC0) + 1 / n_ice

    ensures that C0 remains larger than the minimum allowed value
    during optimization.
    """
    return float(np.exp(logC0) + 1. / n_ice)

def get_C0_from_log(logC0, n_ice):

    if isinstance(logC0, np.ndarray):
        logC0 = logC0[0]

    return float(get_C0_from_log_scalar(logC0, n_ice))

def obj_delta_y_sqr(logC0, y1, z1, y2, z2, layers, n_deep,
                    downgoing, with_air):
    """
    Objective function used for root-finding during ray solution search.

    Parameters
    ----------
    logC0 : float
        Logarithmic optimization parameter.

    y1, z1 : float
        Starting coordinates.

    y2, z2 : float
        Target coordinates.

    layers : tuple of ndarray
        Layer parameter arrays.

    n_deep : float
        Refractive index in deep ice.

    downgoing : bool
        Flag indicating reversed geometry.

    with_air : bool
        Flag indicating propagation through air.

    Returns
    -------
    float
        Squared horizontal mismatch between ray endpoint
        and target position.
    """
    C0 = get_C0_from_log(logC0, n_deep)
    dy = get_delta_y(C0, y1, z1, y2, z2, layers, (-1., -1.),downgoing,with_air)

    if not np.isfinite(dy):
        return 1e30
    
    return dy*dy

def obj_delta_y(logC0, y1, z1, y2, z2, layers, n_deep,
                downgoing, with_air):
    """
    Objective function returning the horizontal mismatch of a ray.

    This function is used by root-finding algorithms to determine
    ray parameters that connect two points.

    Parameters
    ----------
    logC0 : float
        Logarithmic optimization parameter.

    y1, z1 : float
        Starting coordinates.

    y2, z2 : float
        Target coordinates.

    layers : tuple of ndarray
        Layer parameter arrays.

    n_deep : float
        Refractive index of deep ice.

    downgoing : bool
        Flag indicating reversed geometry.

    with_air : bool
        Flag indicating propagation through air.

    Returns
    -------
    float
        Horizontal difference between predicted ray position
        and the target point.
    """
    C0 = get_C0_from_log(logC0, n_deep)
    dy = get_delta_y(C0, y1, z1, y2, z2, layers, (-1., -1.),downgoing,with_air)

    if not np.isfinite(dy):
            return 1e30
    
    return dy

# To keep it numba compatible we not pass the soltuon_types_revert objects directly (or so... works like this):
DIRECT = solution_types_revert['direct']
REFLECTED = solution_types_revert['reflected']
REFRACTED = solution_types_revert['refracted']

#@njit(cache=True)
def determine_solution_type(y1, z1, y2, z2, C0, layers, downgoing=False, with_air=False):
    """
    Determine the physical type of a ray tracing solution.

    This function classifies a ray trajectory based on the location of
    its turning point relative to the emitter and receiver.
    Since it is used in the find_solutions function and we are getting the flipped coordinates here, 
    we just take the information for downgoing and with_air as well, makes this function a bit stupid, sorry

    Parameters
    ----------
    y1, z1 : float
        Horizontal and depth coordinates of the ray origin.

    y2, z2 : float
        Horizontal and depth coordinates of the receiver.

    C0 : float
        Ray parameter controlling the curvature of the trajectory.

    layers : tuple of ndarray
        Layer parameter arrays describing the refractive index profile.

    downgoing : bool
        Flag indicating that the geometry corresponds to a ray entering
        the medium from above (e.g. from the air).

    with_air : bool
        Flag indicating whether propagation through the air layer is
        considered.

    Returns
    -------
    int
        Integer identifier of the ray solution type. The returned value
        corresponds to an entry in ``solution_types_revert``.

        Possible solution classes include

        ``direct``
            The ray reaches the receiver before encountering a turning
            point.

        ``refracted``
            The ray reaches a turning point within the medium and then
            propagates back toward the receiver.

        ``reflected``
            The ray reaches the surface (z ≈ 0) and is reflected
            downward.

        ``from_air``
            Special case where the ray originates from the air and
            propagates downward into the medium.

    Notes
    -----
    The classification is determined using the turning point of the ray
    trajectory, obtained from :func:`get_turning_point`.

    The logic proceeds as follows:

    1. If the ray originates from air (``with_air`` and ``downgoing``),
       the solution is classified as direct, since there are only direct rays in this case.

    2. If the receiver lies before the turning point in horizontal
       distance (``y2 < y_turn``), the ray is a direct solution.

    3. If the turning point occurs at or above the surface
       (``z_turn ≥ 0``), the ray is classified as a surface-reflected
       solution.

    4. Otherwise, the ray turns within the medium and is classified as
       a refracted solution.
    """

    y_turn, z_turn = get_turning_point(
        C0,
        y1, z1,
        layers,None,downgoing,with_air
    )

    if with_air: # and downgoing:
        # from air we only find direct solutions (very sloppy check, I know)
        return DIRECT
    
    if y2 < y_turn:
        # receiver reached before turning point -> direct ray
        return DIRECT

    if z_turn >= -1e-12:
        # turning point above ice -> reflection of upwards going ray
        return REFLECTED
    
    return REFRACTED

def find_solutions(x1, x2, layers,tol=1e-12):
    """
    Find all valid ray tracing solutions between two points.

    The function searches for ray parameters C0 that connect the start
    and end positions using numerical root finding.

    Parameters
    ----------
    x1 : tuple
        Start coordinates (y, z).

    x2 : tuple
        End coordinates (y, z).

    layers : list of dict or tuple of ndarray
        Layer definitions.

    tol : float, optional
        Root-finding tolerance.

    Returns
    -------
    list of dict
        List of ray solutions. Each solution contains

        ``type`` : int
            Solution type identifier.

        ``C0`` : float
            Ray parameter.

        ``D`` : float
            Logarithmic parameter used during optimization.

        ``x1`` : tuple
            Starting coordinate.

    Notes
    -----
    Possible solution types include

    * direct rays
    * refracted rays
    * surface-reflected rays
    * rays originating from above the ice surface

    Additional internal flags allow handling of

    * downward-going geometries
    * propagation involving air layers.

    Examples
    --------
    >>> find_solutions((0,-50), (1,0), LAYERS)
    [{'type': 1, 'C0': 0.5, 'D': 0.1, 'x1': (0,-50)}]
    """
    

    if isinstance(layers, list):
        layers = layers_to_arrays(layers)
    

    results = []
    C0s = []
    z_min, z_max, n_ice, delta_n, z0 = layers

    y1, z1 = float(x1[0]), float(x1[1])
    y2, z2 = float(x2[0]), float(x2[1])

    # We only need to find upwards going solutions because of the horizontal invariance of n(z)
    # To find the path from a x1 to a deeper x2 we just have to swap the z values and search from
    # x1' = (y1,z2) to x2' = (y2,z1) instead which makes this all a bit simpler

    with_air = False
    if (z1 > 0.0) or (z2 > 0.0):
        with_air = True

    downgoing = False
    if z1 > z2:
        z1, z2 = z2, z1
        downgoing = True

    n_deep = n_ice[-1]

    theta_straight = np.arctan(max((z2-z1),1e-14)/(y2-y1))

    if theta_straight < np.pi/4 and not with_air: 
        theta_straight = np.pi/4

    
    air_solution_found = False

    if not air_solution_found:
        _, theta_skim = get_skim_angle(
            y1, z1,
            z2,
            layers
                )

        if not np.isfinite(theta_skim):
            theta_skim = np.arctan(z1/y1)


        C0skim = get_C0_from_theta(
            z1,
            np.abs(theta_skim),
            layers
        )

        C0straight = get_C0_from_theta(
            z1,
            np.abs(theta_straight),
            layers
        )

        n_z = get_refractive_index(z1,layers)
        logC0straight = np.log(max(C0straight - 1./n_deep, 1e-12))
        logC0skim_nz = np.log(max(1/n_z - 1./n_deep, 1e-12))
        logC0skim = np.log(max(C0skim- 1./n_deep, 1e-12))

        result = optimize.root(obj_delta_y_sqr, x0=logC0straight, args=(y1,z1,y2,z2,layers, n_deep,downgoing,with_air), tol=tol)

        if(result.fun < 1e-7):
            if(np.round(result.x[0], 3) not in np.round(C0s, 3)):
                C_0 = get_C0_from_log(result.x[0],n_deep)
                C_1, _, _, _ = compute_offsets(C_0,y1, z1, layers)
                C0s.append(C_0)
                solution_type = determine_solution_type(y1,z1,y2,z2, C_0, layers,downgoing,with_air)
                
                results.append({'type': solution_type,
                                'C0': C_0,
                                'C1': C_1,
                                'reflection': 0,
                                'reflection_case': 0,
                                'D' : result.x[0],
                                'x1': x1,
                                'flag' : 1})
        else:
    
            result = optimize.root(obj_delta_y_sqr, x0=logC0skim, args=(y1,z1,y2,z2,layers, n_deep,downgoing,with_air), tol=tol)
            if(result.fun < 1e-7):
                if(np.round(result.x[0], 3) not in np.round(C0s, 3)):
                    C_0 = get_C0_from_log(result.x[0],n_deep)
                    C_1, _, _, _ = compute_offsets(C_0,y1, z1, layers)
                    C0s.append(C_0)
                    solution_type = determine_solution_type(y1,z1,y2,z2, C_0, layers,downgoing,with_air)
                    
                    results.append({'type': solution_type,
                                    'C0': C_0,
                                    'C1': C_1,
                                    'reflection': 0,
                                    'reflection_case': 0,
                                    'D' : result.x[0],
                                    'x1': x1,
                                    'flag' : 1})
        
        # check if another solution with higher logC0 exists
        if result.x[0] is None: 
            result_x = logC0skim_nz
        else:
            result_x = result.x[0]

        logC0_start = result_x + 0.00001
        
        if with_air:
            C0cross_min = 1.0
            logC0_start = np.log(max(C0cross_min - 1./n_deep, 1e-12))

        logC0_stop = 100.0

        delta_start = obj_delta_y(
            logC0_start,
            y1, z1, y2, z2,
            layers,
            n_deep,downgoing,with_air
            )

        delta_stop = obj_delta_y(
            logC0_stop,
            y1, z1, y2, z2,
            layers,
            n_deep,downgoing,with_air
            )

        if(np.sign(delta_start) != np.sign(delta_stop)):

            result2 = optimize.brentq(obj_delta_y, logC0_start, logC0_stop, args=(y1,z1,y2,z2,layers, n_deep,downgoing,with_air))

            if(np.round(result2, 3) not in np.round(C0s, 3)):
                C_0 = get_C0_from_log(result2,n_deep)
                C_1, _, _, _ = compute_offsets(C_0,y1, z1, layers)
                C0s.append(C_0)
                solution_type = determine_solution_type(y1,z1,y2,z2, C_0, layers,downgoing,with_air)

                results.append({'type': solution_type,
                                'C0': C_0,
                                'C1': C_1,
                                'reflection': 0,
                                'reflection_case': 0,
                                'D' : result2,
                                'x1': x1,
                                'flag' : 3})
        
        theta_min =  1e-5
        C0theta_min = get_C0_from_theta(
            z1,
            theta_min,
            layers
            )
        if C0theta_min <= 1/n_deep:
            C0theta_min = 1/n_deep + 1e-12  # small buffer to avoid log(0)

        logC0_start = max(np.log(C0theta_min - 1. / n_deep),-100)
        
        logC0_stop = result_x - 0.00001

        delta_start = obj_delta_y(
            logC0_start,
            y1, z1, y2, z2,
            layers,
            n_deep,downgoing,with_air
            )

        delta_stop = obj_delta_y(
                logC0_stop,
                y1, z1, y2, z2,
                layers,
                n_deep,downgoing,with_air
                )

        if(np.sign(delta_start) != np.sign(delta_stop)):

            result3 = optimize.brentq(obj_delta_y, logC0_start, logC0_stop, args=(y1,z1,y2,z2, layers, n_deep,downgoing,with_air))

            if(np.round(result3, 3) not in np.round(C0s, 3)):
                C_0 = get_C0_from_log(result3, n_deep)
                C_1, _, _, _ = compute_offsets(C_0,y1, z1, layers)
                C0s.append(C_0)
                solution_type = determine_solution_type(y1,z1,y2,z2, C_0, layers,downgoing,with_air)

                results.append({'type': solution_type,
                                'C0': C_0,
                                'C1': C_1,
                                'reflection': 0,
                                'reflection_case': 0,
                                'D' : result3,
                                'x1': x1,
                                'flag' : 4})
        
    return sorted(results, key=itemgetter('type', 'C0'))

def get_path(C0, x1, x2, layers, n_points=2000, return_turning_point = False, get_segments = False):
    """
    Compute the analytic ray trajectory between two points.

    Parameters
    ----------
    C0 : float
        Ray parameter.

    x1 : tuple
        Starting coordinate (y, z).

    x2 : tuple
        Target coordinate (y, z).

    layers : list of dict or tuple of ndarray
        Layer definitions.

    n_points : int, optional
        Number of sampling points used to build the forward branch.

    return_turning_point: bool, optional
        Whether to return the turning point coordintes or not. Default: False. Function returns 4 elements instead of 2 when True!

    Returns
    -------
    y_path : ndarray
        Horizontal coordinates of the ray path.

    z_path : ndarray
        Depth coordinates of the ray path.

    y_turn : float
        Y (horizontal displacement) coordinate of the turning point.

    z_turn : float
        Z (depth) coordinate of the turning point.

    Notes
    -----
    If a turning point occurs the trajectory is mirrored around the
    turning point to generate the refracted portion of the path.

    The resulting path is truncated once the horizontal coordinate
    reaches the receiver position.
    """
    if isinstance(layers, list):
        layers = layers_to_arrays(layers)
    y1, z1 = x1
    y2, z2 = x2

    with_air = False
    if (z1 > 0.0) or (z2 > 0.0):
        with_air = True

    downgoing = False
    if z1 > z2:
        downgoing = True
        z1, z2 = z2, z1

    if get_segments is True:
        C1, _, yb, zb = compute_offsets(C0, y1, z1, layers, get_intersection_point=True)
    else:
        C1, _, _, _ = compute_offsets(C0, y1, z1, layers)

    y_turn, z_turn = get_turning_point(C0,y1,z1,layers,C1,downgoing,with_air)

    if z_turn <= z1 or with_air or y_turn > y2 or y_turn is None or z_turn is None:
        z_forward = np.linspace(z1, z2, n_points)
        y_forward, _ = build_y_field(C0, z_forward, layers, C1)
        y_path, z_path = y_forward, z_forward
    else:
        z_forward = np.linspace(z1, z_turn, n_points)
        y_forward, _ = build_y_field(C0, z_forward, layers, C1)
        
        y_mirror = 2*y_turn - y_forward
        z_up = z_forward[::-1]
        y_up = y_mirror[::-1]
        y_path = np.concatenate([y_forward, y_up])
        z_path = np.concatenate([z_forward, z_up])

    # cut to receiver
    dy = y_path - y2
    cross = np.where(np.diff(np.sign(dy)) != 0)[0]

    if len(cross) > 0:
        i = cross[0]
        t = (y2 - y_path[i]) / (y_path[i+1] - y_path[i])
        z_hit = z_path[i] + t * (z_path[i+1] - z_path[i])
        y_path = np.concatenate([y_path[:i+1], [y2]])
        z_path = np.concatenate([z_path[:i+1], [z_hit]])

    if downgoing:
        y_half = y2 - (y2 - y1)/2
        y_path = 2*y_half - y_path

    if return_turning_point is True:
        if get_segments is True:
            return y_path, z_path, y_turn, z_turn, yb, zb
        else:
            return y_path, z_path, y_turn, z_turn
    else: 
        if get_segments is True:
            return y_path, z_path, yb, zb
        else:
            return y_path, z_path


#@njit(cache=True)
def get_path_segments(C0, x1, x2, layers):
    """
    Construct piecewise ray path segments in a multilayer medium.

    The ray path between two points is decomposed into segments bounded by
    layer interfaces and, if present, the turning point of the ray. Each
    segment lies entirely within a single layer with an exponential refractive
    index profile and is labeled by its propagation direction.

    Segments are defined in depth coordinates and are suitable for subsequent
    analytic evaluation of path length and travel time.

    Parameters
    ----------
    C0 : float
        Ray parameter (inverse horizontal slowness), defined as
        C0 = 1 / beta.

    x1 : tuple of float
        Start point (y, z) in Cartesian coordinates.

    x2 : tuple of float
        End point (y, z) in Cartesian coordinates.

    layers : tuple of ndarray
        Medium definition consisting of:
        (z_min, z_max, n_ice, delta_n, z0)

        z_min, z_max : ndarray
            Lower and upper depth boundaries of each layer.

        n_ice : ndarray
            Base refractive index per layer.

        delta_n : ndarray
            Amplitude of exponential refractive index variation.

        z0 : ndarray
            Exponential scale depth per layer.

    Returns
    -------
    segments : numba.typed.List of tuples
        List of path segments. Each segment is defined as:
        (z_start, z_end, C0, layer_idx, direction)

        z_start : float
            Starting depth of the segment.

        z_end : float
            Ending depth of the segment.

        C0 : float
            Ray parameter (passed through for convenience).

        layer_idx : int
            Index of the layer in which the segment lies.

        direction : int
            Propagation direction:
            - UPGOING (1): increasing z
            - DOWNGOING (2): decreasing z
    """


    z_min, z_max, n_ice, delta_n, z0 = layers

    y1, z1 = float(x1[0]), float(x1[1])
    y2, z2 = float(x2[0]), float(x2[1])

    with_air = False
    if (z1 > 0.0) or (z2 > 0.0):
        with_air = True

    downgoing = False
    if z1 > z2:
        z1, z2 = z2, z1
        downgoing = True

    solution_type = determine_solution_type(y1,z1,y2,z2, C0, layers,downgoing,with_air)

    # Compute offsets and get layer intersections while we are at it
    C1, idx_start, yb, zb = compute_offsets(C0, y1, z1, layers, get_intersection_point=True)
    # Compute turning point
    y_turn, z_turn = get_turning_point(C0, y1, z1, layers, C1, downgoing, False)

    if solution_type != 1: # Refracted/reflected rays: go up to the turning point and down again to x2

        # Upwards part from z1 to z_turn
        points_up = List()
        points_up.append(z1)

        for i in range(len(zb)):
            z_b = zb[i]
            if z1 < z_b < z_turn:
                points_up.append(z_b)

        points_up.append(z_turn)

        points_up.sort()

        # Downwards part from z_turn to z2
        points_down = List()
        points_down.append(z_turn)

        for i in range(len(zb)):
            z_b = zb[i]
            if z2 < z_b < z_turn:
                points_down.append(z_b)

        points_down.append(z2) # Include endpoint

        points_down.sort()
        points_down.reverse()
    
    else: # Direct path: upwards going from x1 to x2
        points_up = List()
        points_up.append(z1)

        for i in range(len(zb)):
            z_b = zb[i]
            if z1 < z_b < z2:
                points_up.append(z_b)

        points_up.append(z2) # Include endpoint

        points_up.sort()

    # Build segments from edge points
    segments = List()

    # Upgoing segments
    for i in range(len(points_up)-1):
        z_start = points_up[i]
        z_end = points_up[i+1]
        z_mid = 0.5 * (z_start + z_end)
        idx = get_layer_index(z_mid, z_min, z_max)
        segments.append((z_start, z_end, C0, idx, 1)) # Set upgoing flag to 1

    if solution_type != DIRECT:

        # Downgoing segments
        for i in range(len(points_down)-1):
            z_start = points_down[i]
            z_end = points_down[i+1]
            z_mid = 0.5 * (z_start + z_end)
            idx = get_layer_index(z_mid, z_min, z_max)
            segments.append((z_start, z_end, C0, idx, 0)) # Set upgoing flag to 0

    return segments

#@njit(cache=True)
def get_path_length_analytic(C0, x1, x2, layers):
    """
    Compute total analytic ray path length in a multilayer medium.

    The total path length is obtained by summing analytic contributions from
    individual segments returned by `get_path_segments`. Each segment lies in a
    layer with exponential refractive index profile:

        n(z) = n_ice - delta_n * exp(z / z0)

    The path length within each segment is evaluated using a closed-form
    solution derived from the ray equation.
    The math was taken from Appendix C.5 of https://doi.org/10.1140/epjc/s10052-020-7612-8 
    but without the detour over the angle to get beta but using beta = 1/C0 which should be equivalent (according to my understanding -> know where to search if stuff)

    Parameters
    ----------
    C0 : float
        Ray parameter (inverse horizontal slowness).

    x1 : tuple of float
        Start point (y, z).

    x2 : tuple of float
        End point (y, z).

    layers : tuple of ndarray
        Medium definition consisting of:
        (z_min, z_max, n_ice_arr, delta_n_arr, z0_arr)

    Returns
    -------
    total_s : float
        Total geometric path length of the ray.
    """


    z_min, z_max, n_ice_arr, delta_n_arr, z0_arr = layers    
    total_s = 0.0

    segments = get_path_segments(C0,x1,x2,layers)

    for seg in segments:
        z1, z2, C0, idx, upgoing = seg

        n_ice = n_ice_arr[idx]
        delta_n = delta_n_arr[idx]
        z0 = z0_arr[idx]

        
        beta = 1.0 / C0 
        alpha = n_ice**2 - beta**2

        
        def gamma(z):
            n_z = n_ice - delta_n * np.exp(z / z0)
            g = n_z**2 - beta**2
            return max(g,1e-14)
        
        EPS = 1e-10

        def l1(z):
            n_z = n_ice - delta_n * np.exp(z / z0)
            val = sqrt(alpha * gamma(z)) + n_ice * n_z - beta**2
            return abs(val)

        def l2(z):
            val = sqrt(gamma(z)) + (n_ice - delta_n * np.exp(z / z0))
            return abs(val)

        def get_s(z):
            return n_ice / sqrt(alpha) * (z - z0 * log(l1(z))) + z0 * log(l2(z))

        if upgoing==1:
            s_seg = get_s(z2) - get_s(z1)
        else:
            s_seg = get_s(z1) - get_s(z2)

        total_s += s_seg

    return total_s

#@njit
def get_launch_angle(C0, x1, x2, layers):
    """
    Compute the ray launch angle at the starting point.

    The launch angle is determined from the local refractive index and the ray
    parameter C0 using Snell's law in a continuously varying medium.

    Parameters
    ----------
    C0 : float
        Ray parameter (inverse horizontal slowness).

    x1 : tuple of float
        Start point (y, z).

    x2 : tuple of float
        End point (y, z).

    layers : tuple of ndarray
        Medium definition.

    Returns
    -------
    angle : float
        Launch angle in radians, measured with respect to the vertical. 
    """

    y1, z1 = float(x1[0]), float(x1[1])
    y2, z2 = float(x2[0]), float(x2[1])

    with_air = False
    if (z1 > 0.0) or (z2 > 0.0):
        with_air = True

    downgoing = False
    if z1 > z2:
        z1, z2 = z2, z1
        downgoing = True

    solution_type = determine_solution_type(y1,z1,y2,z2, C0, layers,downgoing,with_air)
    n = get_refractive_index(x1[1],layers)

    if solution_type == DIRECT and downgoing: 
        angle = np.pi - np.arcsin(1/(n*C0))
    else:
        angle = np.arcsin(1/(n*C0))

    return angle

#@njit
def get_receiving_angle(C0, x1, x2, layers):
    """
    Compute the ray receiving angle at the endpoint.

    The receiving angle is determined from the local refractive index at the
    endpoint and the ray parameter C0, accounting for ray geometry and solution
    type (direct or refracted).

    Parameters
    ----------
    C0 : float
        Ray parameter (inverse horizontal slowness).

    x1 : tuple of float
        Start point (y, z).

    x2 : tuple of float
        End point (y, z).

    layers : tuple of ndarray
        Medium definition.

    Returns
    -------
    angle : float
        Receiving angle in radians, measured with respect to the vertical.

    """
    
    z_min, z_max, n_ice, delta_n, z0 = layers

    y1, z1 = float(x1[0]), float(x1[1])
    y2, z2 = float(x2[0]), float(x2[1])

    with_air = False
    if (z1 > 0.0) or (z2 > 0.0):
        with_air = True

    downgoing = False
    if z1 > z2:
        z1, z2 = z2, z1
        downgoing = True

    solution_type = determine_solution_type(y1,z1,y2,z2, C0, layers,downgoing,with_air)
    n = get_refractive_index(x2[1],layers)

    if solution_type == DIRECT and not downgoing:
        angle = np.pi - np.arcsin(1/(n*C0))
    else:
        angle = np.arcsin(1/(n*C0))
    return angle

def get_launch_vector(C0, x1, x2, layers):
    """
    Compute the launch direction vector of the ray.

    The launch vector is constructed from the launch angle at the
    starting point and expressed in Cartesian (y, z) coordinates.
    The angle is measured with respect to the horizontal, so the
    vector components are given by (cos(theta), sin(theta)).

    Parameters
    ----------
    C0 : float
        Ray parameter (inverse horizontal slowness).

    x1 : tuple of float
        Start point (y, z).

    x2 : tuple of float
        End point (y, z).

    layers : tuple of ndarray
        Medium definition.

    Returns
    -------
    ndarray of shape (2,)
        Launch direction vector (vy, vz). The vector is normalized and
        points along the ray at the emission point.
    """
    angle = get_launch_angle(C0, x1, x2, layers)

    vy = np.cos(angle)
    vz = np.sin(angle)

    return np.array((vy, vz))

def get_receiving_vector(C0, x1, x2, layers):
    """
    Compute the receiving direction vector of the ray.

    The receiving vector is constructed from the receiving angle at
    the endpoint and expressed in Cartesian (y, z) coordinates. The
    vector points toward the receiver, i.e. opposite to the local ray
    propagation direction, such that the vertical component is inverted.

    Parameters
    ----------
    C0 : float
        Ray parameter (inverse horizontal slowness).

    x1 : tuple of float
        Start point (y, z).

    x2 : tuple of float
        End point (y, z).

    layers : tuple of ndarray
        Medium definition.

    Returns
    -------
    ndarray of shape (2,)
        Receiving direction vector (vy, vz). The vector is normalized
        and points toward the receiver at the arrival point.
    """
    angle = get_receiving_angle(C0, x1, x2, layers)

    vy = np.cos(angle)
    vz = -np.sin(angle)

    return np.array((vy, vz))

def get_reflection_angle(C0, x1, x2, layers):
    """
    Compute the surface reflection angle of a ray solution.

    For surface-reflected solutions, the incidence angle is evaluated
    just below the surface (z ≈ 0) and the reflection angle is defined
    as twice the incidence angle. This corresponds to the angle between
    the incoming and reflected ray directions for specular reflection
    at a horizontal interface.

    Parameters
    ----------
    C0 : float
        Ray parameter (inverse horizontal slowness).

    x1 : tuple of float
        Start point (y, z).

    x2 : tuple of float
        End point (y, z).

    layers : tuple of ndarray
        Medium definition.

    Returns
    -------
    float or None
        Reflection angle in radians for reflected solutions. Returns
        None if the ray solution is not surface-reflected.

    Notes
    -----
    The solution type is determined using :func:`determine_solution_type`.
    The incidence angle is evaluated at z = -1e-12 to avoid numerical
    issues at the surface boundary.
    """

    y1, z1 = float(x1[0]), float(x1[1])
    y2, z2 = float(x2[0]), float(x2[1])

    with_air = False
    if (z1 > 0.0) or (z2 > 0.0):
        with_air = True

    downgoing = False
    if z1 > z2:
        z1, z2 = z2, z1
        downgoing = True

    solution_type = determine_solution_type(
        y1, z1, y2, z2, C0, layers, downgoing, with_air
    )

    if solution_type != REFLECTED:
        return None
    
    # evaluate just below surface
    x_surface = (y1, -1e-12)

    incidence_angle = get_launch_angle(C0, x_surface, x2, layers)

    return 2.0 * incidence_angle

#@njit(cache=True)
def get_travel_time_analytic(C0, x1, x2, layers):
    """
    Compute total analytic ray path length in a multilayer medium.

    The total path length is obtained by summing analytic contributions from
    individual segments returned by `get_path_segments`. Each segment lies in a
    layer with exponential refractive index profile:

        n(z) = n_ice - delta_n * exp(z / z0)

    The path length within each segment is evaluated using a closed-form
    solution derived from the ray equation. 
    The math is taken from Appendix C.5 of https://doi.org/10.1140/epjc/s10052-020-7612-8 
    but without the detour over the angle to get beta but using beta = 1/C0 which should be equivalent (according to my understanding)

    Parameters
    ----------
    C0 : float
        Ray parameter (inverse horizontal slowness).

    x1 : tuple of float
        Start point (y, z).

    x2 : tuple of float
        End point (y, z).

    layers : tuple of ndarray
        Medium definition consisting of:
        (z_min, z_max, n_ice_arr, delta_n_arr, z0_arr)

    Returns
    -------
    total_s : float
        Total geometric path length of the ray.
    """

    z_min, z_max, n_ice_arr, delta_n_arr, z0_arr = layers    
    total_t = 0.0

    segments = get_path_segments(C0,x1,x2,layers)

    for seg in segments:
        z1, z2, C0, idx, upgoing = seg

        n_ice = n_ice_arr[idx]
        delta_n = delta_n_arr[idx]
        z0 = z0_arr[idx]

        
        beta = 1.0 / C0 
        alpha = n_ice**2 - beta**2

        
        def gamma(z):
            n_z = n_ice - delta_n * np.exp(z / z0)
            g = n_z**2 - beta**2
            return max(g,1e-14)


        def l1(z):
            n_z = n_ice - delta_n * np.exp(z / z0)
            val = sqrt(alpha * gamma(z)) + n_ice * n_z - beta**2
            return abs(val) 

        def l2(z):
            val = sqrt(gamma(z)) + (n_ice - delta_n * np.exp(z / z0))
            return abs(val) 
        
        def get_t(z):
            return ( z0 * ( sqrt(gamma(z)) + n_ice * log(l2(z)) - log(l1(z)) * (n_ice**2 / sqrt(alpha)) ) + z * (n_ice**2 / sqrt(alpha)) ) / constants.c

        if upgoing==1:
            t_seg = get_t(z2) - get_t(z1)
        else:
            t_seg = get_t(z1) - get_t(z2)

        total_t += t_seg

    return total_t


def get_frequencies_for_attenuation(
        frequency,
        n_frequencies_integration=32,
        max_detector_freq=None):
    """
    Construct a reduced set of frequencies for attenuation integration.

    The function selects a sparse set of frequencies spanning the non-zero
    entries of the input frequency array. If ``max_detector_freq`` is provided,
    the frequency range is split into a dense region up to the detector
    bandwidth and a sparser region above it.

    Parameters
    ----------
    frequency : array_like
        Frequency array. Only positive values are considered.
    n_frequencies_integration : int, optional
        Maximum number of frequencies used for attenuation integration.
        Default is 32.
    max_detector_freq : float or None, optional
        Maximum detector frequency. If provided, frequencies up to this value
        are sampled densely, while higher frequencies are sampled more
        sparsely. If None, a single evenly spaced grid is returned.
        Default is None.

    Returns
    -------
    ndarray
        Array of frequencies used for attenuation integration. Returns an empty
        array if no positive frequencies are provided.

    Notes
    -----
    This function is implemented as an alternative to analyticraytracing.ray_tracing_2D.__get_frequencies_for_attenuation to test functionality without class structure. 
    Should be omitted once the class structure is implemented and the other function can be used.
    """

    mask = frequency > 0
    freq_nonzero = frequency[mask]

    if len(freq_nonzero) == 0:
        return np.array([])

    fmin = freq_nonzero.min()
    fmax = freq_nonzero.max()

    n = min(n_frequencies_integration, len(freq_nonzero))

    # simple case
    if max_detector_freq is None:
        return np.linspace(fmin, fmax, n)

    # split detector / above detector
    det_mask = freq_nonzero <= max_detector_freq

    if np.sum(det_mask) < 2:
        return np.linspace(fmin, fmax, n)

    f_det = np.linspace(
        freq_nonzero[det_mask].min(),
        freq_nonzero[det_mask].max(),
        n
    )

    # upper half sparse
    if np.sum(~det_mask) > 1:
        f_hi = np.linspace(
            freq_nonzero[~det_mask].min(),
            freq_nonzero[~det_mask].max(),
            n // 2
        )
        return np.concatenate((f_det, f_hi))

    return f_det

#@njit(cache=True, fastmath=True)
def ds_dz_layer(z, C0, idx, layers):
    """
    Compute differential path length factor ds/dz for a layered refractive index.

    This function evaluates the geometrical factor

        ds/dz = n / sqrt(n^2 - beta^2)

    for a refractive index profile of the form

        n(z) = n_ice - delta_n * exp(z / z0)

    within a given layer.

    Parameters
    ----------
    z : ndarray
        Depth coordinates at which to evaluate the differential path length.
    C0 : float
        Ray tracing constant. The parameter ``beta`` is defined as ``1 / C0``.
    idx : int
        Layer index selecting parameters from ``layers``.
    layers : tuple of ndarray
        Layer parameter arrays in the form
        ``(z_min, z_max, n_ice_arr, delta_n_arr, z0_arr)``.

    Returns
    -------
    ndarray
        Differential path length factor ``ds/dz`` evaluated at ``z``.

    Notes
    -----
    A small floor is applied to the denominator to avoid numerical
    singularities near turning points.
    """

    z_min, z_max, n_ice_arr, delta_n_arr, z0_arr = layers

    n_ice = n_ice_arr[idx]
    delta_n = delta_n_arr[idx]
    z0 = z0_arr[idx]

    beta = 1.0 / C0

    n = n_ice - delta_n * np.exp(z / z0)

    gamma = n * n - beta * beta

    # avoid turning-point singularity
    gamma = np.maximum(gamma, 1e-14)

    return n / np.sqrt(gamma)

def get_attenuation_along_path(
        C0,
        x1,
        x2,
        layers,
        frequency,
        freqs=None,
        attenuation_model="GL3",
        dz=10 * units.m,
        refine=True,
        n_frequencies_integration=32,
        max_detector_freq=None
        ):
    """
    Compute frequency-dependent attenuation along a ray path.

    The attenuation is calculated by integrating the inverse attenuation
    length along ray segments within ice layers. The computation is performed
    on a reduced set of frequencies and interpolated to the full frequency
    array for efficiency.

    Parameters
    ----------
    x1 : array_like
        Starting position of the ray.
    x2 : array_like
        End position of the ray.
    C0 : float
        Ray tracing constant defining the trajectory.
    layers : tuple
        Layered medium definition passed to the ray tracing and refractive
        index evaluation routines.
    frequency : ndarray
        Frequencies at which the attenuation factor is evaluated.
    freqs : ndarray
        Coarser frequencies that were calculated from ``frequency`` using the ``__get_frequencies_for_attenuation`` function. If not provided, this will be calculated using the adapted function above. Default is None.
    attenuation_model : str, optional
        Name of the attenuation model passed to
        ``attenuation.get_attenuation_length``. Default is "GL3".
    dz : float, optional
        Step size in depth for numerical integration. Default is 10 m.
    n_frequencies_integration : int, optional
        Number of frequencies used for sparse attenuation integration.
        Default is 32.
    max_detector_freq : float or None, optional
        Maximum detector frequency used to bias frequency sampling toward the
        detector band. Default is None.

    Returns
    -------
    ndarray
        Frequency-dependent attenuation factor along the full path.

    Notes
    -----
    - Only segments below the surface (z < 0) contribute to attenuation.
    - The integral is evaluated as

      exp(-∫ ds / L(f, z))

      where ``L`` is the attenuation length.
    - The exponent is clipped to avoid overflow in the exponential.
    - Sparse-frequency attenuation is interpolated to the full frequency grid.
    """

    # We can again use our segment function to get monotonous segments contained into one layer
    segments = get_path_segments(C0, x1, x2, layers)

    turning_z = None
    z_receiver = x2[1]

    for i in range(len(segments) - 1):
        _, z2_prev, _, _, up1 = segments[i]
        _, _, _, _, up2 = segments[i + 1]

        # upgoing -> downgoing transition
        if up1 == 1 and up2 == 0:
            turning_z = z2_prev
            break
    if refine:
        dz_fine = dz / 2.0          # finer resolution near turning point
        turning_window = 10 * dz      # refine within this distance
        receiver_window = 10 * dz

        dz_very_fine = dz / 200.0          # finer resolution near turning point
        turning_window_fine = 3 * dz      # refine within this distance
        receiver_window_fine = 3 * dz

    attenuation_factor = np.ones_like(frequency)

    if freqs is None:
        # Get sparser frequencies that we use for calculation, can then interpolate for finer results afterwards
        freqs = get_frequencies_for_attenuation(
            frequency,
            n_frequencies_integration,
            max_detector_freq
        )

    if len(freqs) == 0:
        return attenuation_factor

    mask = frequency > 0

    for seg in segments:

        z1, z2, C0, idx, direction = seg

        # Skip air segments (above z=0.0) since we assume the attenuation to be neglegible there
        if z1 >= 0 and z2 >= 0:
            continue

        z1 = min(z1, 0.0)
        z2 = min(z2, 0.0)

        z_edges = [z1]
        z = z1

        direction = 1 if z2 > z1 else -1

        while (z < z2 if direction > 0 else z > z2):
            
            if refine:
                use_fine = False
                use_very_fine = False

                # refine near turning point
                if turning_z is not None:
                    if abs(z - turning_z) < turning_window:
                        use_fine = True
                    if abs(z - turning_z) < turning_window_fine:
                        use_very_fine = True

                # refine near receiver
                if abs(z - z_receiver) < receiver_window:
                    use_fine = True
                
                if abs(z - z_receiver) < receiver_window_fine:
                    use_very_fine = True

                if use_very_fine:
                    dz_local = dz_very_fine
                elif use_fine:
                    dz_local = dz_fine
                else:
                    dz_local = dz
            else:
                dz_local = dz


            z_next = z + direction * dz_local

            # avoid overshoot
            if (direction > 0 and z_next > z2) or (direction < 0 and z_next < z2):
                z_next = z2

            z_edges.append(z_next)
            z = z_next

        z_edges = np.array(z_edges)

        dz_actual = np.abs(np.diff(z_edges))
        z_mid = z_edges[:-1] + 0.5 * np.diff(z_edges)

        # Get ds_dz factor used in the integral
        ds_dz = ds_dz_layer(z_mid, C0, idx, layers)
        
        # Compute sparse frequencies
        attenuation_sparse = np.empty_like(freqs)

        for i, f in enumerate(freqs):

            L = attenuation.get_attenuation_length(
                z_mid,
                f,
                attenuation_model
            )

            exponent = np.sum((ds_dz * dz_actual) / L)

            if exponent > 700.0:
                exponent = 700.0

            attenuation_sparse[i] = np.exp(-exponent)

        attenuation_segment = np.ones_like(frequency)

        attenuation_segment[mask] = np.interp(
            frequency[mask],
            freqs,
            attenuation_sparse
        )

        # Overall attenuation: each individual attenuation contribution reduces the signal 
        # -> multiply factor of each integration segment
        attenuation_factor *= attenuation_segment

    return attenuation_factor

#@njit(cache=True)
def get_focusing_factor(C0, x1, x2, layers):
    """
    Analytic solution to calculate the focusing factor

    This was adapted from analyticraytracing.py and evaluates the path integrals taken from Sjoerd Bouma's PhD thesis. 
    The segments to evaulate are again calculated with the get_path_segments function, analogously to how it is done in the other functions here.

    Parameters
    ----------
    C0 : float
        Ray parameter (inverse horizontal slowness).

    x1 : tuple of float
        Start point (y, z).

    x2 : tuple of float
        End point (y, z).

    layers : tuple of ndarray
        Medium definition consisting of:
        (z_min, z_max, n_ice_arr, delta_n_arr, z0_arr)

    Returns
    -------
    focusing_factor : float
        Total focusing factor of the path

    """

    z_min, z_max, n_ice_arr, delta_n_arr, z0_arr = layers    
    beta = 1.0 / C0

    y1, z1 = float(x1[0]), float(x1[1])
    y2, z2 = float(x2[0]), float(x2[1])

    with_air = False
    if (z1 >= 0.0) or (z2 >= 0.0):
        with_air = True

    downgoing = False
    if z1 > z2:
        z1, z2 = z2, z1
        downgoing = True

    solution_type = determine_solution_type(y1, z1, y2, z2, C0, layers, downgoing, with_air)

    if solution_type == REFRACTED:
        return np.nan
    
    segments = get_path_segments(C0, x1, x2, layers)



    w_phi = 0.0
    w_theta = 0.0

    for seg in segments:
        z1, z2, C0, idx, direction = seg

        # Skip air segments (above z=0.0) since we assume the attenuation to be neglegible there
        if z1 >= 0 and z2 >= 0:
            continue

        n_ice = n_ice_arr[idx]
        delta_n = delta_n_arr[idx]
        z0 = z0_arr[idx]

        alpha = n_ice**2 - beta**2

        def n_of_z(z):
            return n_ice - delta_n * np.exp(z / z0)

        def gamma(z):
            g = n_of_z(z)**2 - beta**2
            return max(g, 1e-14)   # avoid instability

        def phi_F(z):
            val = np.sqrt(alpha * gamma(z)) + n_ice * n_of_z(z) - beta**2
            return (1.0 / np.sqrt(alpha)) * (
                z - z0 * np.log(abs(val))
            )

        def theta_F(z):
            val = np.sqrt(alpha * gamma(z)) + n_ice * n_of_z(z) - beta**2

            return (
                n_ice**2 * z / (alpha**1.5)
                + z0 * (n_ice * n_of_z(z) + beta**2) / (alpha * np.sqrt(gamma(z)))
                - n_ice**2 * z0 / (alpha**1.5) * np.log(abs(val))
            )

        if direction == 1:
            w_phi += phi_F(z2) - phi_F(z1)
            w_theta += theta_F(z2) - theta_F(z1)
        else:
            w_phi += phi_F(z1) - phi_F(z2)
            w_theta += theta_F(z1) - theta_F(z2)

    


    if x1[1] > 0:
        launch_angle = get_launch_angle(C0, x1, (0,-0.001), layers)
        n1 = get_refractive_index(-0.001, layers)
    else:
        launch_angle = get_launch_angle(C0, x1, x2, layers)
        n1 = get_refractive_index(x1[1], layers)

    if x2[1] > 0:
        receive_angle = get_receiving_angle(C0, (0,-0.001), x2, layers)
        n2 = get_refractive_index(-0.001, layers)
    else:
        receive_angle = get_receiving_angle(C0, x1, x2, layers)
        n2 = get_refractive_index(x2[1], layers)

    if x1[1] > 0:
        s = get_path_length_analytic(C0, (0, -0.001), x2, layers)
    elif x2[1] > 0:
        s = get_path_length_analytic(C0, x1, (0, -0.001), layers)
    else:
        s = get_path_length_analytic(C0, x1, x2, layers)

    f_inv_sq = (
        n1 * n2
        * abs(np.cos(launch_angle) * np.cos(receive_angle))
        * (w_theta * w_phi / (s**2))
        )
    
    return np.sqrt(1 / f_inv_sq)


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
                 compile_numba=False):
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

        numba_available = False
        List = list # fallback for get_path_segments function

        try:
            from numba import jit, njit
            from numba.typed import List
            numba_available = True
            logger.status("Numba version of raytracer is available")
        except ImportError:
            logger.warning("Numba is not available")
            numba_available = False
            

        if compile_numba:
            if numba_available:
                global get_layer_index, analytic_F, compute_offsets, build_y_field, evaluate_y, find_z_turn
                global get_turning_point, get_delta_y, get_refractive_index, get_C0_from_theta, get_skim_angle
                global determine_solution_type, get_path_segments, get_path_length_analytic, get_launch_angle
                global get_receiving_angle, get_reflection_angle, get_travel_time_analytic, ds_dz_layer, get_focusing_factor
                try:
                    get_layer_index = jit(get_layer_index, nopython=True, cache=True)
                    analytic_F = jit(analytic_F, nopython=True, cache=True)
                    compute_offsets = jit(compute_offsets, nopython=True, cache=True)
                    build_y_field = jit(build_y_field, nopython=True, cache=True)
                    evaluate_y = jit(evaluate_y, nopython=True, cache=True)
                    find_z_turn = jit(find_z_turn, nopython=True, cache=True)
                    get_turning_point = jit(get_turning_point, nopython=True, cache=True)
                    get_delta_y = jit(get_delta_y, nopython=True, cache=True)
                    get_refractive_index = jit(get_refractive_index, nopython=True, cache=True)
                    get_C0_from_theta = jit(get_C0_from_theta, nopython=True, cache=True)
                    get_skim_angle = jit(get_skim_angle, nopython=True, cache=True)
                    determine_solution_type = jit(determine_solution_type, nopython=True, cache=True)
                    get_path_segments = jit(get_path_segments, nopython=True, cache=True)
                    get_path_length_analytic = jit(get_path_length_analytic, nopython=True, cache=True)
                    get_launch_angle = jit(get_launch_angle, nopython=True, cache=True)
                    get_receiving_angle = jit(get_receiving_angle, nopython=True, cache=True)
                    get_reflection_angle = jit(get_reflection_angle, nopython=True, cache=True)
                    get_travel_time_analytic = jit(get_travel_time_analytic, nopython=True, cache=True)
                    ds_dz_layer = jit(ds_dz_layer, nopython=True, cache=True)
                    get_focusing_factor = jit(get_focusing_factor, nopython=True, cache=True)
                    self.use_cpp = False
                except Exception:
                    self.__logger.warning("Error in compiling methods using jit - proceeding without numba")
                    compile_numba = False

    @property
    def _layers_arr(self):
        return self.medium.get_layers_array

    @log_timing()
    def determine_solution_type(self, x1, x2, C0):

        y1, z1 = x1
        y2, z2 = x2

        with_air = False
        if (z1 > 0.0) or (z2 > 0.0):
            with_air = True

        downgoing = False
        if z1 > z2:
            z1, z2 = z2, z1
            downgoing = True

        
        solution_type = determine_solution_type(y1, z1, y2, z2, C0, self._layers_arr, downgoing, with_air)
        
        self._logger.info(
            "solution_type | x1=%s x2=%s C0=%s type=%s",
            x1,
            x2,
            C0,
            solution_type
            )
        
        return solution_type
    
    @log_timing()
    def find_solutions(self, x1, x2, plot=False, *_, **__):

        solutions = find_solutions(x1, x2, self._layers_arr)

        self._logger.info(
            "find_solutions | x1=%s x2=%s solutions=%s",
            x1,
            x2,
            solutions
            )

        return solutions

    @log_timing()
    def get_travel_time_analytic(self, x1, x2, C0, *_, **__):

        travel_time = get_travel_time_analytic(C0, x1, x2, self._layers_arr)

        self._logger.info(
            "get_travel_time_analytic | x1=%s x2=%s C0=%s travel_time=%s",
            x1,
            x2,
            C0,
            travel_time
            )

        return travel_time

    @log_timing()
    def get_path_length_analytic(self, x1, x2, C0, *_, **__):

        path_length = get_path_length_analytic(C0, x1, x2, self._layers_arr)

        self._logger.info(
            "get_path_length_analytic | x1=%s x2=%s C0=%s path_length=%s",
            x1,
            x2,
            C0,
            path_length
            )

        return path_length

    def get_launch_vector(self, x1, x2, C0):
        return get_launch_vector(C0, x1, x2, self._layers_arr)
    
    def get_receive_vector(self, x1, x2, C0):
        return get_receiving_vector(C0, x1, x2, self._layers_arr)
    
    @log_timing()
    def get_launch_angle(self, x1, C0, *_, **__):

        launch_angle = get_launch_angle(C0, x1, x1, self._layers_arr)

        self._logger.info(
            "get_launch_angle | x1=%s C0=%s launch_angle=%s",
            x1,
            C0,
            launch_angle
            )

        return launch_angle
    
    @log_timing()
    def get_receive_angle(self, x1, x2, C0, *_, **__):

        receive_angle = get_receiving_angle(C0, x1, x2, self._layers_arr)

        self._logger.info(
            "get_receive_angle | x1=%s x2=%s C0=%s receive_angle=%s",
            x1,
            x2,
            C0,
            receive_angle
            )

        return receive_angle

    @log_timing()
    def get_reflection_angle(self, x1, x2, C0, *_, **__):

        reflection_angle = get_reflection_angle(C0, x1, x2, self._layers_arr)

        self._logger.info(
            "get_reflection_angle | x1=%s x2=%s C0=%s reflection_angle=%s",
            x1,
            x2,
            C0,
            reflection_angle
            )

        return reflection_angle


    def get_path_reflections(self, x1, x2, C0, npoints=1000,*_, **__):
        return get_path(C0, x1, x2, self._layers_arr, npoints)
    
    def get_path_segments(self, x1, x2, C0, *_, **__):
        return get_path_segments(C0, x1, x2, self._layers_arr)
    
    def get_turning_point(self, x1, C0):
        with_air = False
        if x1[1] > 0.0 : with_air = True
        return get_turning_point(x1[0], x1[1], C0, self._layers_arr, with_air=with_air)

    @log_timing()
    def get_focusing_analytic(self, x1, x2, C0, *_, **__):

        focusing_factor = get_focusing_factor(C0, x1, x2, self._layers_arr)

        self._logger.info(
            "get_focusing_analytic | x1=%s x2=%s C0=%s focusing_factor=%s",
            x1,
            x2,
            C0,
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
    
    @log_timing()
    def get_attenuation_along_path(self, x1, x2, C0, frequency, max_detector_frequency=None, *_, **__):
        
        attenuation_model =  self.attenuation_model
        dz = self.dz
        freqs = self.__get_frequencies_for_attenuation(frequency, max_detector_frequency)
        
        attenuation_factor = get_attenuation_along_path(C0, x1, x2, self._layers_arr, frequency, freqs, attenuation_model, dz)
        
        self._logger.info(
            "get_attenuation_along_path | x1=%s x2=%s C0=%s attenuation_factor=%s",
            x1,
            x2,
            C0,
            attenuation_factor
            )

        return attenuation_factor
    

def get_inice_quantities(pos, theta_air, layers):   
    """
    Compute the in-ice propagation quantities for a ray entering the surface
    at a given air incidence angle.

    This function uses the analytic multilayer ray tracer to determine the
    horizontal displacement and travel time of a refracted ray propagating
    from the ice surface to a receiver position.

    Parameters
    ----------
    pos : array-like of float
        Receiver position given as (x, y, z) in meters.
        Only the y and z coordinates are used internally since the
        ray tracing is performed in a 2D plane.

    theta_air : float
        Zenith angle of the incoming ray in air in radians.
        Defined with respect to the surface normal.

    layers : tuple of np.ndarray
        Layered refractive-index model as returned by
        layers_to_arrays().

    Returns
    -------
    horizontal_offset : float
        Horizontal distance in meters between the receiver position and the
        corresponding surface intersection point of the refracted ray within
        the 2D propagation plane.

    travel_time : float
        Signal propagation time in seconds from the surface to the receiver
        through the ice.

    Notes
    -----
    The ray parameter is determined from Snell's law 
    and the analytic multilayer ray tracing formalism is used to evaluate
    the corresponding trajectory and travel time.
    """

    n_air = get_refractive_index(0.0001, layers)

    p = n_air * np.sin(theta_air)
    C0 = 1.0 / p

    x1 = (pos[1],pos[2])
    x2 = (0, 0)
    travel_time = get_travel_time_analytic(C0, x1, x2, layers)

    C1, _, _, _ = compute_offsets(C0, pos[1], pos[2], layers)

    horizontal_offset =  evaluate_y(C0, C1, 0, layers) - evaluate_y(C0, C1, pos[2], layers)

    return horizontal_offset, travel_time

def get_time_difference_plane_wave_analytic(pos1, pos2, theta_air, phi_air, layers):
    """
    Compute the relative arrival time of a plane wave between two receivers.

    The incoming signal is modeled as a plane wave arriving from air with
    zenith angle theta_air and azimuth angle phi_air. The total
    arrival-time difference is calculated from:

    1. The difference in refracted in-ice propagation times.
    2. The difference in air propagation lengths before the rays enter the
       ice surface.

    The in-ice propagation is evaluated analytically using the multilayer
    ray tracer.

    Parameters
    ----------
    pos1 : array-like of float
        Position of the first receiver as (x, y, z) in meters.

    pos2 : array-like of float
        Position of the second receiver as (x, y, z) in meters.

    theta_air : float
        Zenith angle of the incoming plane wave in air in radians.
        Defined with respect to the surface normal.

    phi_air : float
        Azimuth angle of the incoming plane wave in radians.

    layers : tuple of np.ndarray
        Layered refractive-index model as returned by
        layers_to_arrays().

    Returns
    -------
    delta_t : float
        Relative signal arrival time in seconds

    Notes
    -----
    For each receiver, the analytic ray tracer determines:

    - the in-ice travel time
    - the horizontal surface displacement between the receiver and the
      corresponding surface intersection point of the refracted ray

    The surface intersection points are reconstructed from the incoming wave vector and 
    the horizontal displacements yielded with the analytic raytracer.

    The difference in air propagation length is then obtained from the
    projection of the separation vector between both surface intersection
    points onto the full 3D propagation direction.
    """

    # horizontal propagation direction of incoming signal, projected onto surface
    src_hvec = -np.array([
        np.sin(phi_air),
        np.cos(phi_air)
    ])

    n_air = get_refractive_index(0.0001, layers)

    # in-ice raytracing solution for given surface angle and antenna depth
    # r: radial distance the ray covers on the way from the surface to the antenna depth
    # t: travel time along this path in ice

    r1, t1 = get_inice_quantities(pos1, theta_air, layers)
    r2, t2 = get_inice_quantities(pos2, theta_air, layers)

    # receiver positions projected onto surface ()
    R1 = np.array([pos1[0], pos1[1]])
    R2 = np.array([pos2[0], pos2[1]])

    # actual surface intersection points
    P1 = R1 + r1 * src_hvec
    P2 = R2 + r2 * src_hvec

    print(f"P1: {P1}")
    print(f"P2: {P2}")
    
    # vector connecting both surface intersection points
    dP = P1 - P2

    # projection of this connection vector onto src_hvec
    # surface_shift = dP if incoming wave is along dP
    # surface_shift = 0 if incoming wave is perpendicular to dP
    
    # difference in air travel length from projection of the surface_shift onto the actual 3D incoming ray vector
    # same as multiplying by sin(theta)

    surface_shift = np.inner(src_hvec, dP)
    delta_L_air = surface_shift * np.sin(theta_air)

    # convert to travel time
    delta_t_air = n_air * delta_L_air*units.m / constants.c

    # total relative arrival time
    delta_t = (t1 - t2) + delta_t_air

    return delta_t