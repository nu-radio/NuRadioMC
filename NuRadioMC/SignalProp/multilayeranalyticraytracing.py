"""
Ray tracing module for layered exponential refractive index profiles.

This module implements analytic ray tracing in glacial ice or another medium where the
refractive index can be described as layers of exponentials where each layer follows 

n(z) = n_ice - delta_n * exp(z / z_0)

This also suppports propagation across layer boundaries where n(z) is not continuous, so also air-to-ice tracing.
We can also model the refractive index of air as an exponential layer of course.
Internal layer reflections inside the ice are not yet implemented, but should be easily available in the future.

To find the ray solutions we needed to implement a number of functions
that you can use to :

* evaluate refractive index profiles
* compute analytic ray trajectories
* determine turning points
* solve for ray parameters connecting two points
* classify solutions (direct, refracted, reflected)

...if they are in our nice layer format that expects smooth boundaries of the n(z) definition that we apply

The implementation is an expansion of the previous analytic ray
tracing solver used in NuRadioMC . In order to enable compilation with
``numba.njit(nopython=True)``, the layer definitions are internally
converted from dictionary-based objects to arrays. 
For more insight into the physics behind this approach and remarks on the solving strategy it is recommended 
to have a look at the companion note to this module ``MultilayerAnalyticRayTracting.md`` and the appendix C of
"NuRadioMC: Simulating the Radio Emission of Neutrinos from Interaction to Detector“ (2020). https://doi.org/10.1140/epjc/s10052-020-7612-8.
This appendix describes the implementation of the previously used single layer analytic raytracer.


Examples
--------
Typical workflow:

1. Define the ice layers (e.g. ``LAYERS``).
2. Call :func:`find_solutions` to determine valid ray parameters between
   two points.
3. For each solution, compute the ray path using :func:`get_path`.

Notes
-----
Coordinates are given as (y, z) with units of meters:

* y : horizontal distance
* z : depth (negative downward)

z = 0 corresponds to the surface.

The ray parameter ``C0`` represents the **inverse horizontal slowness**
of the ray and determines the curvature of the trajectory.
It can also be defined as C0 = 1/(n(z)*sin(theta)) where theta is the angle relative to the horizontal.

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
    Surface-to-deep refractive index contrast.

z_0 : float
    Exponential scale depth controlling the transition of the index profile.

region : str
    Internal identifier of the physical region.

region_name : str
    Human-readable name of the region.

Internally these definitions are converted to arrays using
:func:`layers_to_arrays` in order to support Numba compilation.

Contact: hannes.warnhofer@desy.de
Created 2026
Developed for use in NuRadioMC ray tracing.
"""

import numpy as np
from scipy import optimize
from operator import itemgetter
from numba import njit
from functools import lru_cache
from NuRadioMC.SignalProp.propagation import solution_types, solution_types_revert

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
    }
]

LAYERS_AIR = [
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
        "z_min": -3000.0,
        "z_max": -80.5,
        "n_ice": 1.77468,
        "delta_n": 1.41573,
        "z_0": 1/0.0387882,
        "region": "bubbly_ice",
        "region_name": "Ice"
    }
]


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


@njit(cache = True)
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

@njit(cache = True)
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
    c = n_ice*n_ice - 1.0/(C0*C0)

    # F only valid for positive c
    if c <= 0 :
        return np.nan
    
    gamma = delta_n * np.exp(z / z0)
    root = np.abs(gamma*gamma - gamma*b + c)
    
    logargument = gamma / (2.0*np.sqrt(c)*np.sqrt(root) - b*gamma + 2.0*c)
    
    val = z0 * (n_ice*n_ice*C0*C0 - 1.0)**-0.5 * np.log(logargument)
    return float(np.real(val))

@njit(cache = True)
def compute_offsets(C0, y_start, z_start, layers):
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

    for i in range(idx_start - 1, -1, -1):
        zb = float(z_min[i])
        F_deep = analytic_F(zb, C0, n_ice[i+1], delta_n[i+1], z0[i+1])
        yb = float(F_deep + C1[i+1])
        F_shallow = analytic_F(zb, C0, n_ice[i], delta_n[i], z0[i])
        C1[i] = float(yb - F_shallow)

    return C1, idx_start

@njit(cache = True)
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
        #print(f"F: {F}")
        #print(f"C1[{idx}]={C1[idx]}")
        y[j] = float(F + C1[idx])
        #print(f"y[{j}]={y[j]}")
    return y, layer_idx

@njit(cache = True)
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

@njit(cache = True)
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
    C0 = float(C0)
    z_min, z_max, n_ice, delta_n, z0 = layers
    target_n = 1.0 / C0
    n_layers = len(z_min)
    for i in range(n_layers):
        n_min = n_ice[i] - delta_n[i] * np.exp(z_min[i] / z0[i])
        n_max = n_ice[i] - delta_n[i] * np.exp(z_max[i] / z0[i])
        if n_min >= target_n >= n_max:
            val = (n_ice[i] - target_n) / delta_n[i]

            if val <= 0:
                #return z_max[-1] + 1000.0
                return np.max(z_max)
            
            z = z0[i] * np.log(val)

            if np.isnan(z):
                #return z_max[-1] + 1000.0
                return np.max(z_max)
            return z
            #return float(z0[i] * np.log((n_ice[i] - target_n) / delta_n[i]))
    return np.max(z_max)

@njit(cache = True)
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
        C1, _ = compute_offsets(C0, y_start, z_start, layers)
    
    # Find depth of turning point
    z_turn = find_z_turn(C0, layers)
    
    numerical_safety_offset = 1e-6
    z_surface = 0.0 - numerical_safety_offset

    if (z_turn >= z_surface) and not with_air:
            z_turn = z_surface
    
    # Evaluate horizontal coordinate at turning point
    y_turn = evaluate_y(C0, C1, z_turn, layers)
    
    return y_turn, z_turn

@njit(cache = True)
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
    C1, _ = compute_offsets(C0, y1, z1, layers)
    #z_turn = float(find_z_turn(C0, layers))
    #y_turn = float(evaluate_y(C0, C1, z_turn, layers))
    y_turn, z_turn = get_turning_point(C0,y1,z1,layers,C1,downgoing,with_air)
    
    if (y_turn is not None) and (z_turn is not None):
        if z_turn < z2:
            dz = z_turn - z2
            dy = y_turn - y2
            diff = np.sqrt(dz*dz + dy*dy) + 10.0 * np.abs(dz)
            return -diff
        elif y_turn > y2:
            y_fit = evaluate_y(C0, C1, z2, layers)
            return y2 - y_fit
        else:
            y_raw = evaluate_y(C0, C1, z2, layers)
            y_fit = 2.0*y_turn - y_raw
            return -(y2 - y_fit)
    else: 
        y_fit = evaluate_y(C0, C1, z2, layers)
        return y2 - y_fit

@njit(cache = True)
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

@njit(cache = True)
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

@njit(cache = True)
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
    #sinthcrit = nsurf / nlaunch
    sinthcrit = min(nsurf / nlaunch, 0.999999)
    if sinthcrit <= 1.0:
        thcrit = np.arcsin(sinthcrit)
        C0crit = get_C0_from_theta(z1, thcrit, layers)
    else:
        thcrit = 1e-12  # nearly zero angle
        C0crit = -1.0
    return C0crit, thcrit

# ------------------------------
# Optimization helpers
# ------------------------------
def get_C0_from_log(logC0, n_ice):
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


def determine_solution_type(y1, z1, y2, z2, C0, layers, downgoing, with_air):
    """
    Determine the physical type of a ray tracing solution.

    This function classifies a ray trajectory based on the location of
    its turning point relative to the emitter and receiver.

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
       the solution is classified as ``from_air``.

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

    if with_air is True and downgoing is True:
        return solution_types_revert['from_air']
        #return 0

    if y2 < y_turn:
        # receiver reached before turning point -> direct ray
        return solution_types_revert['direct']

    if z_turn >= -1e-6:
        return solution_types_revert['reflected']
    

    return solution_types_revert['refracted']

def find_solutions(x1, x2, layers,tol=1e-6):
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

    ## Here something is still wrong
    ## theta skim goes to inf for too horizontal geometries when z1 is on the same height as z2.

    theta_straight = np.arctan((z2-z1)/(y2-y1))
    #print(f"theta_straight: {theta_straight}")

    _, theta_skim = get_skim_angle(
        y1, z1,
        z2,
        layers
        )

    if not np.isfinite(theta_skim):
        theta_skim = theta_straight - 0.1
    #print(f"theta_skim: {theta_skim}")
    

    C0skim = get_C0_from_theta(
        z1,
        np.abs(theta_skim),
        layers
    )
    #print(f"C0skim: {C0skim}")

    C0straight = get_C0_from_theta(
        z1,
        np.abs(theta_straight),
        layers
    )
    #print(f"C0straight: {C0straight}")
    
    logC0straight = np.log(max(C0straight - 1./n_deep, 1e-12))
    logC0skim = np.log(max(C0skim - 1./n_deep, 1e-12))
    #print(f"logC0skim: {logC0skim}")
    print(f"-------------------------------------------------------")
    print(f"Original x1 and x2: {x1} and {x2}. With air: {with_air}. Downgoing: {downgoing}")
    print(f"Searching for Ray-Tracing Solutions from x1 ({y1},{z1}) to x2 ({y2},{z2})...")
    result = optimize.root(obj_delta_y_sqr, x0=logC0straight, args=(y1,z1,y2,z2,layers, n_deep,downgoing,with_air), tol=tol)
    print(f"result of root otimization with C0 {get_C0_from_log(result.x[0],n_deep)}: {result}")
    if(result.fun < 1e-7):
        if(np.round(result.x[0], 3) not in np.round(C0s, 3)):
            C_0 = get_C0_from_log(result.x[0],n_deep)
            C0s.append(C_0)
            solution_type = determine_solution_type(y1,z1,y2,z2, C_0, layers,downgoing,with_air)
            
            results.append({'type': solution_type,
                            'C0': C_0,
                            'D' : result.x[0],
                            'x1': x1})
    else:
        # or maybe just see again what this brings us and keep it if it's new
        result = optimize.root(obj_delta_y_sqr, x0=logC0skim, args=(y1,z1,y2,z2,layers, n_deep,downgoing,with_air), tol=tol)
        if(result.fun < 1e-7):
            if(np.round(result.x[0], 5) not in np.round(C0s, 5)):
                C_0 = get_C0_from_log(result.x[0],n_deep)
                C0s.append(C_0)
                solution_type = determine_solution_type(y1,z1,y2,z2, C_0, layers,downgoing,with_air)
                
                results.append({'type': solution_type,
                                'C0': C_0,
                                'D' : result.x[0],
                                'x1': x1})
                
            '''else:
            logC0_stop = 100.
            delta_start = obj_delta_y(
                logC0skim,
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
            
            print("delta_start: ", delta_start)
            print("delta_stop: ", delta_stop)

            if(np.sign(delta_start) != np.sign(delta_stop)):
                result = optimize.brentq(obj_delta_y, logC0skim, logC0_stop, args=(y1,z1,y2,z2,layers, n_deep,downgoing,with_air))
                if(np.round(result, 5) not in np.round(C0s, 5)):
                    C_0 = get_C0_from_log(result,n_deep)
                    C0s.append(C_0)
                    solution_type = determine_solution_type(y1,z1,y2,z2, C_0, layers,downgoing,with_air)

                    results.append({'type': solution_type,
                                    'C0': C_0,
                                    'D' : result2,
                                    'x1': x1})'''

    # check if another solution with higher logC0 exists
    logC0_start = result.x[0] + 0.00001
    #logC0_start = 0.0
    if with_air:
        C0cross_min = 1.0
        logC0_start = np.log(max(C0cross_min - 1./n_deep, 1e-12))

    logC0_stop = 100.0

    delta_test = obj_delta_y(
        -10.,
        y1, z1, y2, z2,
        layers,
        n_deep,downgoing,with_air
        )

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
    
    #print("with_air: ", with_air)
    #print("downgoing: ", downgoing)
    #print("logC0_start: ", logC0_start)
    #print("logC0_stop: ", logC0_stop)
    #print("delta_start: ", delta_start)
    #print("delta_stop: ", delta_stop)
    #print("delta_test: ", delta_test)

    if(np.sign(delta_start) != np.sign(delta_stop)):

        result2 = optimize.brentq(obj_delta_y, logC0_start, logC0_stop, args=(y1,z1,y2,z2,layers, n_deep,downgoing,with_air))

        if(np.round(result2, 5) not in np.round(C0s, 5)):
            C_0 = get_C0_from_log(result2,n_deep)
            C0s.append(C_0)
            solution_type = determine_solution_type(y1,z1,y2,z2, C_0, layers,downgoing,with_air)

            results.append({'type': solution_type,
                            'C0': C_0,
                            'D' : result2,
                            'x1': x1})
    else:
        print("no solution with logC0 > {:.3f} exists".format(result.x[0]))

    
    theta_min =  1e-5
    C0theta_min = get_C0_from_theta(
        z1,
        theta_min,
        layers
        )
    if C0theta_min <= 1/n_deep:
        C0theta_min = 1/n_deep + 1e-12  # small buffer to avoid log(0)

    logC0_start = max(np.log(C0theta_min - 1. / n_deep),-100)
    #print('logC0_start: ',logC0_start)
    
    
    logC0_stop = result.x[0] - 0.00001
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
        print("solution with logC0 < {:.3f} exists".format(result.x[0]))
        result3 = optimize.brentq(obj_delta_y, logC0_start, logC0_stop, args=(y1,z1,y2,z2, layers, n_deep,downgoing,with_air))


        if(np.round(result3, 5) not in np.round(C0s, 5)):
            C_0 = get_C0_from_log(result3, n_deep)
            C0s.append(C_0)
            solution_type = determine_solution_type(y1,z1,y2,z2, C_0, layers,downgoing,with_air)

            print("found {} solution C0 = {:.2f}".format(solution_types[solution_type], C_0))
            results.append({'type': solution_type,
                            'C0': C_0,
                            'D' : result3,
                            'x1': x1})
    else:
        print("no solution with logC0 < {:.3f} exists".format(result.x[0]))


    print(f"Solution found for x1 ({y1},{z1}) to x2 ({y2},{z2}): {results}")
    print(f"-------------------------------------------------------")
    return sorted(results, key=itemgetter('type', 'C0'))

# ------------------------------
# Path builder
# ------------------------------
def get_path(C0, x1, x2, layers, n_points=2000):
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

    Returns
    -------
    y_path : ndarray
        Horizontal coordinates of the ray path.

    z_path : ndarray
        Depth coordinates of the ray path.

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

    C1, _ = compute_offsets(C0, y1, z1, layers)
    print(f"C1 in get_path call: {C1}")
    y_turn, z_turn = get_turning_point(C0,y1,z1,layers,C1,downgoing,with_air)

    if z_turn <= z1 or with_air or y_turn > y2 or y_turn is None or z_turn is None:
        z_forward = np.linspace(z1, z2, n_points)
        y_forward, _ = build_y_field(C0, z_forward, layers, C1)
        y_path, z_path = y_forward, z_forward
    else:
        z_forward = np.linspace(z1, z_turn, n_points)
        y_forward, _ = build_y_field(C0, z_forward, layers, C1)
        
        print(f"z_turn: {z_turn}, y_turn: {y_turn}")
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

    return y_path, z_path