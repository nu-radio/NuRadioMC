
import numpy as np
from scipy import integrate
from operator import itemgetter
#from numba import njit
#from numba.typed import List
from functools import lru_cache
from NuRadioMC.SignalProp.propagation import solution_types, solution_types_revert
from NuRadioMC.utilities import attenuation

from NuRadioMC.SignalProp.AnalyticRayTracingImpl.maybenumba import njit

from NuRadioReco.utilities import units, geometryUtilities, constants
#from NuRadioMC.utilities import attenuation as attenuation_util, medium as medium_util
from NuRadioMC.SignalProp.propagation_base_class import ray_tracing_base

from math import sqrt, log, sin

from NuRadioMC.SignalProp.AnalyticRayTracingImpl.MultilayerAnalyticRayTracing.corefunctions import layers_to_arrays, compute_offsets, get_turning_point, get_layer_index, build_y_field, get_n_1D, determine_solution_type

NumbaList = list # fallback for get_path_segments function


DIRECT = solution_types_revert['direct']
REFLECTED = solution_types_revert['reflected']
REFRACTED = solution_types_revert['refracted']

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


@njit(cache=True)
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
        points_up = NumbaList()
        points_up.append(z1)

        for i in range(len(zb)):
            z_b = zb[i]
            if z1 < z_b < z_turn:
                points_up.append(z_b)

        points_up.append(z_turn)

        points_up.sort()

        # Downwards part from z_turn to z2
        points_down = NumbaList()
        points_down.append(z_turn)

        for i in range(len(zb)):
            z_b = zb[i]
            if z2 < z_b < z_turn:
                points_down.append(z_b)

        points_down.append(z2) # Include endpoint

        points_down.sort()
        points_down.reverse()

    else: # Direct path: upwards going from x1 to x2
        points_up = NumbaList()
        points_up.append(z1)

        for i in range(len(zb)):
            z_b = zb[i]
            if z1 < z_b < z2:
                points_up.append(z_b)

        points_up.append(z2) # Include endpoint

        points_up.sort()

    # Build segments from edge points
    segments = NumbaList()

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

@njit(cache=True)
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
        alpha = max(n_ice**2 - beta**2, 1e-15)


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

@njit(cache=True)
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
    n = get_n_1D(x1[1],layers)

    if solution_type == DIRECT and downgoing:
        angle = np.pi - np.arcsin(1/(n*C0))
    else:
        angle = np.arcsin(1/(n*C0))

    return angle

@njit(cache=True)
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
    n = get_n_1D(x2[1],layers)

    if solution_type == DIRECT and not downgoing:
        angle = np.pi - np.arcsin(1/(n*C0))
    else:
        angle = np.arcsin(1/(n*C0))
    return angle

@njit(cache=True)
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

@njit(cache=True)
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

@njit(cache=True)
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

@njit(cache=True)
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
        alpha = max(n_ice**2 - beta**2, 1e-15)


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

@njit(cache=True, fastmath=True)
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
        dz_fine = dz / 20.0          # finer resolution near turning point
        turning_window = 10 * dz      # refine within this distance
        receiver_window = 10 * dz

        dz_very_fine = dz / 500.0          # finer resolution near turning point
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

def get_attenuation_along_path_new(
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
    integration_window_size = 20 * units.m  # e.g., 20 meters

    # We can again use our segment function to get monotonous segments contained into one layer
    segments = get_path_segments(C0, x1, x2, layers)

    y1, z1 = float(x1[0]), float(x1[1])
    y2, z2 = float(x2[0]), float(x2[1])

    with_air = False
    if (z1 > 0.0) or (z2 > 0.0):
        with_air = True

    downgoing = False
    if z1 > z2:
        z1, z2 = z2, z1
        downgoing = True


    C1, _ , _, _= compute_offsets(C0, y1, z1, layers)
    y_turn, z_turn = get_turning_point(C0,y1,z1,layers,C1,downgoing,with_air)

    turning_z = z_turn
    z_receiver = x2[1]

    for i in range(len(segments) - 1):
        _, z2_prev, _, _, up1 = segments[i]
        _, _, _, _, up2 = segments[i + 1]

        # upgoing -> downgoing transition
        if up1 == 1 and up2 == 0:
            turning_z = z2_prev
            break

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

        # Skip air segments
        if z1 >= 0 and z2 >= 0:
            continue

        z1 = min(z1, 0.0)
        z2 = min(z2, 0.0)

        # Determine if this segment overlaps with the integration window around the turning point
        lower_bound = turning_z - integration_window_size / 2
        upper_bound = turning_z + integration_window_size / 2

        is_near_turning = (
            turning_z is not None and
            not (z2 < lower_bound or z1 > upper_bound)
        )

        if is_near_turning:
            def integrand(z, f):
                L = attenuation.get_attenuation_length(z, f, attenuation_model)
                ds_dz_val = ds_dz_layer(z, C0, idx, layers)
                return ds_dz_val / L

            attenuation_sparse = []
            for f in freqs:
                exponent, _ = integrate.quad(
                    lambda z: integrand(z, f),
                    z1, z2,
                    epsabs=1e-4, epsrel=1e-2,
                    points=[turning_z]  # Force evaluation at turning point
                )
                if exponent > 700.0:
                    exponent = 700.0
                attenuation_sparse.append(np.exp(-exponent))

            attenuation_sparse = np.array(attenuation_sparse)
        else:
            z_edges = [z1]
            z = z1
            while (z < z2 if direction > 0 else z > z2):
                dz_local = dz
                z_next = z + direction * dz_local
                if (direction > 0 and z_next > z2) or (direction < 0 and z_next < z2):
                    z_next = z2
                z_edges.append(z_next)
                z = z_next

            z_edges = np.array(z_edges)
            dz_actual = np.abs(np.diff(z_edges))
            z_mid = z_edges[:-1] + 0.5 * np.diff(z_edges)
            ds_dz_vals = ds_dz_layer(z_mid, C0, idx, layers)

            attenuation_sparse = np.zeros_like(freqs)
            for i, f in enumerate(freqs):
                L = attenuation.get_attenuation_length(z_mid, f, attenuation_model)
                exponent = np.sum(ds_dz_vals * dz_actual / L)
                if exponent > 700.0:
                    exponent = 700.0
                attenuation_sparse[i] = np.exp(-exponent)

        # Interpolate sparse results onto full frequency grid
        attenuation_segment = np.ones_like(frequency)
        attenuation_segment[mask] = np.interp(
            frequency[mask],
            freqs,
            attenuation_sparse
        )
        attenuation_factor *= attenuation_segment

    return attenuation_factor

@njit(cache=True)
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
        launch_angle = get_launch_angle(C0, x1, (0,-0.0001), layers)
        n1 = get_n_1D(-0.0001, layers)
    else:
        launch_angle = get_launch_angle(C0, x1, x2, layers)
        n1 = get_n_1D(x1[1], layers)

    if x2[1] > 0:
        receive_angle = get_receiving_angle(C0, (0,-0.0001), x2, layers)
        n2 = get_n_1D(-0.0001, layers)
    else:
        receive_angle = get_receiving_angle(C0, x1, x2, layers)
        n2 = get_n_1D(x2[1], layers)

    if x1[1] > 0:
        s = get_path_length_analytic(C0, (0, -0.0001), x2, layers)
    elif x2[1] > 0:
        s = get_path_length_analytic(C0, x1, (0, -0.0001), layers)
    else:
        s = get_path_length_analytic(C0, x1, x2, layers)

    f_inv_sq = (
        n1 * n2
        * abs(np.cos(launch_angle) * np.cos(receive_angle))
        * (w_theta * w_phi / (s**2))
        )

    return np.sqrt(1 / f_inv_sq)


def get_path_length_numerical(C0, x1, x2, layers):
    """
    Numerically compute the ray path length in a multi-layer medium.

    Parameters
    ----------
    C0 : float
        Ray parameter (inverse horizontal slowness).
    x1 : tuple of float
        Start point (y, z).
    x2 : tuple of float
        End point (y, z).
    layers : tuple of ndarray
        Medium definition: (z_min, z_max, n_ice_arr, delta_n_arr, z0_arr)

    Returns
    -------
    total_s : float
        Total geometric path length.
    """
    z_min, z_max, n_ice_arr, delta_n_arr, z0_arr = layers
    total_s = 0.0

    segments = get_path_segments(C0, x1, x2, layers)

    for seg in segments:
        z1, z2, C0, idx, upgoing = seg
        n_ice = n_ice_arr[idx]
        delta_n = delta_n_arr[idx]
        z0 = z0_arr[idx]

        # Define the refractive index profile for this layer
        def n(z):
            return n_ice - delta_n * np.exp(z / z0)

        # Integrand: ds/dz = sec(theta) = sqrt(1 + (dz/dy)^2)
        def ds(d, C0):
            z = d  # Depth variable
            n_z = n(z)
            gamma = n_z**2 - (1.0 / C0)**2  # beta = 1/C0
            if gamma <= 0:
                return 1e10  # Avoid division by zero (ray turns around)
            cos_theta = np.sqrt(gamma) / n_z
            return 1.0 / cos_theta

        # Handle directionality (upgoing vs downgoing)
        if upgoing:
            s_seg, _ = integrate.quad(ds, z1, z2, args=(C0,), epsabs=1e-4, epsrel=1.49e-08)
        else:
            s_seg, _ = integrate.quad(ds, z2, z1, args=(C0,), epsabs=1e-4, epsrel=1.49e-08)

        total_s += s_seg

    return total_s

def get_travel_time_numerical(C0, x1, x2, layers):
    """
    Numerically compute the ray travel time in a multi-layer medium.

    Parameters
    ----------
    C0 : float
        Ray parameter.
    x1 : tuple of float
        Start point (y, z).
    x2 : tuple of float
        End point (y, z).
    layers : tuple of ndarray
        Medium definition: (z_min, z_max, n_ice_arr, delta_n_arr, z0_arr)

    Returns
    -------
    total_t : float
        Total travel time (in seconds).
    """

    z_min, z_max, n_ice_arr, delta_n_arr, z0_arr = layers
    total_t = 0.0

    segments = get_path_segments(C0, x1, x2, layers)

    for seg in segments:
        z1, z2, C0, idx, upgoing = seg
        n_ice = n_ice_arr[idx]
        delta_n = delta_n_arr[idx]
        z0 = z0_arr[idx]

        def n(z):
            return n_ice - delta_n * np.exp(z / z0)

        # Integrand: dt/dz = n(z)/cos(theta)
        def dt(d, C0):
            z = d
            n_z = n(z)
            gamma = n_z**2 - (1.0 / C0)**2
            if gamma <= 0:
                return 1e10  # Avoid singularities
            cos_theta = np.sqrt(gamma) / n_z
            return n_z / (constants.c * cos_theta)

        if upgoing:
            t_seg, _ = integrate.quad(dt, z1, z2, args=(C0,), epsabs=1e-10, epsrel=1.49e-08)
        else:
            t_seg, _ = integrate.quad(dt, z2, z1, args=(C0,), epsabs=1e-10, epsrel=1.49e-08)

        total_t += t_seg 

    return total_t