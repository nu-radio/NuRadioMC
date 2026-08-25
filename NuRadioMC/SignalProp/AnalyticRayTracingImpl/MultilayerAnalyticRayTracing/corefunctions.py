
import numpy as np

from NuRadioMC.SignalProp.propagation import solution_types_revert

from NuRadioMC.SignalProp.AnalyticRayTracingImpl.maybenumba import njit


from math import sqrt, log, sin


NumbaList = list # fallback for get_path_segments function

#@njit(cache=True)
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

@njit(cache=True)
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

    for i in range(len(z_min)):
        if z_min[i] <= z <= z_max[i]:
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
    c = max(abs(n_ice*n_ice - 1.0/(C0*C0)),1e-14)

    # F only valid for positive c
    if c < 0 :
        return np.nan


    gamma = delta_n * np.exp(z / z0)
    root = max(gamma*gamma - gamma*b + c, 1e-14)

    logargument = gamma / (2.0*np.sqrt(c)*np.sqrt(root) - b*gamma + 2.0*c)

    val = z0 * (n_ice*n_ice*C0*C0 - 1.0)**-0.5 * np.log(abs(logargument))

    return float(np.real(val))

@njit(cache = True)
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
        y[j] = float(F + C1[idx])

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


@njit(cache=True)
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

        if (z_min[i]-eps) <= z <= (z_max[i]):
            if z < best_z:
                best_z = z
                found = True

    if found:
        return best_z

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

@njit(cache = True)
def get_n_1D(z, layers):
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
    n_start = get_n_1D(z_start, layers)
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
    nlaunch = get_n_1D(z1, layers)
    nsurf = get_n_1D(zskim, layers)

    sinthcrit = min(nsurf / nlaunch, 0.99999999)
    if sinthcrit <= 1.0:
        thcrit = np.arcsin(sinthcrit)
        C0crit = get_C0_from_theta(z1, thcrit, layers)
    else:
        thcrit = 1e-12  # nearly zero angle
        C0crit = -1.0

    return C0crit, thcrit


# To keep it numba compatible we not pass the soltuon_types_revert objects directly (or so... works like this):
DIRECT = solution_types_revert['direct']
REFLECTED = solution_types_revert['reflected']
REFRACTED = solution_types_revert['refracted']

@njit(cache=True)
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