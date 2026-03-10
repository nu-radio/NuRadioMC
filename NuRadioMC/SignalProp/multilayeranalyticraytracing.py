"""
Ray tracing module for layered ice refractive index profiles.

This module implements analytic ray tracing in glacial ice or another medium where the
refractive index varies exponentially with depth and may change
between multiple layers. It provides utilities to

* evaluate refractive index profiles
* compute analytic ray trajectories
* determine turning points
* solve for ray parameters connecting two points
* classify solutions (direct, refracted, reflected)

The implementation follows the analytic solution of ray propagation
in exponential media commonly used in radio detection of neutrinos
in glacial ice.

Typical workflow
----------------
1. Define the ice layers (e.g. ``LAYERS``).
2. Use :func:`find_solutions` to determine valid ray parameters between
   two points.
3. Use :func:`get_path` to compute the full ray trajectory.

Coordinates
-----------
Positions are given as (y, z) coordinates:

* y : horizontal distance
* z : depth (negative downward)

The ray parameter ``C0`` corresponds to the inverse horizontal slowness
of the ray and determines the curvature of the trajectory.
Authors
-------
Hannes Warnhofer
    RNO-G/DESY Zeuthen

Contact
-------
hannes.warnhofer@desy.de

Created
-------
2026

Notes
-----
Developed for use in NuRadioMC ray tracing.
"""

import numpy as np
from scipy import optimize
from operator import itemgetter

from NuRadioMC.SignalProp.propagation import solution_types, solution_types_revert

LAYERS_SINGLE = [{
        "n_ice": 1.78,
        #"delta_n": 0.43,
        'delta_n': 0.51,
        #"z_0": 1/0.0132,
        "z_0": 37.25,
        "z_min": -3000.0,
        "z_max": 0.0,
        "region": "single",
        "region_name" : "SingleModel"
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

def get_layer_params(z, layers):
    """
    Return the ice layer parameters for a given depth based on a list of layers.

    The function searches through a list of layer definitions and returns the
    layer whose depth range contains the provided value.

    Parameters
    ----------
    z : float
        Depth coordinate (negative downwards).
    layers : list of dict
        Layer definitions containing `z_min`, `z_max`, `n_ice`, etc.

    Returns
    -------
    dict
        Matching layer dictionary.

    Raises
    ------
    ValueError
        If `z` is outside all defined layers.

    Examples
    --------
    >>> get_layer_params(-50, LAYERS)
    {'z_min': -80.5, 'z_max': -14.9, 'n_ice': 1.89957, ...}
    """

    for layer in layers:
        if layer["z_min"] <= z <= layer["z_max"]:
            return layer
    raise ValueError(f"z={z} is outside the defined layer ranges.")

def get_layer_indices(z_array, layers):
    """
    Determine the layer index for each depth in an array.

    This function assigns each depth to the index of the corresponding
    layer in the provided layer list.

    Parameters
    ----------
    z_array : float or array_like
        Depth(s) to evaluate.
    layers : list of dict
        List of layer definitions.

    Returns
    -------
    int or ndarray of int
        Index(es) of the layer corresponding to each depth.

    Examples
    --------
    >>> get_layer_indices([-10, -50, -100], LAYERS)
    array([0, 1, 2])
    """
    scalar_input = np.isscalar(z_array)

    z_array = np.atleast_1d(z_array)
    layer_idx = np.zeros_like(z_array, dtype=int)

    for i, L in enumerate(layers):
        mask = (z_array > L["z_min"]) & (z_array <= L["z_max"])
        layer_idx[mask] = i

    if scalar_input:
        return int(layer_idx[0])
    return layer_idx

def get_refractive_index(z, layers):
    """
    Compute the refractive index profile n(z).

    The refractive index is evaluated using an exponential parameterization
    within each layer.

    Parameters
    ----------
    z : float or array_like
        Depth coordinate(s).
    layers : list of dict
        Layer definitions containing ``n_ice``, ``delta_n``, and ``z_0``.

    Returns
    -------
    float or ndarray
        Refractive index evaluated at the provided depth(s).

    Notes
    -----
    The refractive index in each layer follows

    n(z) = n_ice - delta_n * exp(z / z_0)

    Examples
    --------
    >>> get_refractive_index(-50, LAYERS)
    1.6
    """
    
    z = np.asarray(z)

    # determine layer index for each z
    layer_idx = get_layer_indices(z, layers)

    # allocate output
    n = np.zeros_like(z, dtype=float)

    # compute per layer
    for i, L in enumerate(layers):
        mask = layer_idx == i
        if np.any(mask):
            n_ice   = L["n_ice"]
            delta_n = L["delta_n"]
            z_0     = L["z_0"]

            n[mask] = n_ice - delta_n * np.exp(z[mask] / z_0)

    # return scalar if scalar input
    if np.isscalar(z):
        return float(n)

    return n

def analytic_F(z, C_0, layer):
    """
    Compute the analytic ray tracing function F(z) for a given layer.

    This function represents the analytic solution of the ray path
    integral in a medium with an exponential refractive index profile.

    Parameters
    ----------
    z : float or ndarray
        Depth coordinate(s).
    C_0 : float
        Ray parameter (inverse horizontal slowness).
    layer : dict
        Layer definition containing ``n_ice``, ``delta_n``, and ``z_0``.

    Returns
    -------
    float or ndarray
        Value of the analytic function F(z).

    Notes
    -----
    The returned function is used to compute the horizontal coordinate
    of the ray trajectory:

    y(z) = F(z) + C1

    Examples
    --------
    >>> analytic_F(-50, 0.5, LAYERS[1])
    12.345
    """

    n_ice   = layer["n_ice"]
    delta_n = layer["delta_n"]
    z_0     = layer["z_0"]

    b = 2 * n_ice
    c = n_ice**2 - C_0**-2

    gamma = delta_n * np.exp(z / z_0)
    root = np.abs(gamma**2 - gamma*b + c)

    logargument = gamma / (2*np.sqrt(c)*np.sqrt(root) - b*gamma + 2*c)

    val = z_0 * (n_ice**2 * C_0**2 - 1)**-0.5 * np.log(logargument)

    val = np.real(val)
    
    return val

def compute_all_offsets(C0, x_start, layers):
    """
    Compute the horizontal offsets C1 for each layer to ensure continuity of y(z).

    Parameters
    ----------
    C0 : float
        Ray parameter.
    x_start : tuple
        Starting coordinates (y_start, z_start).
    layers : list of dict
        Ice layer definitions.

    Returns
    -------
    ndarray
        Array of offset constants ``C1`` for each layer.

    Examples
    --------
    >>> compute_all_offsets(0.5, (0, -50), LAYERS)
    array([1.2, 0.8, 0.5])

    
    Notes
    -----
    The offsets are propagated both upward and downward from the starting
    layer to enforce continuity of the ray trajectory.
    """

    y_start, z_start = x_start
    n_layers = len(layers)

    C1 = np.zeros(n_layers)

    # ---- find starting layer ----
    idx_start = get_layer_indices(z_start, layers)

    # ---- starting offset ----
    F_start = analytic_F(z_start, C0, layers[idx_start])
    C1[idx_start] = y_start - F_start

    # ---- propagate upward (toward surface, smaller index) ----
    for i in range(idx_start - 1, -1, -1):

        z_boundary = layers[i]["z_min"]  # shared boundary


        # compute y at boundary from deeper layer
        F_prev = analytic_F(z_boundary, C0, layers[i+1])
        y_boundary = F_prev + C1[i+1]



        # compute new offset
        F_new = analytic_F(z_boundary, C0, layers[i])
        C1[i] = y_boundary - F_new

    # ---- propagate downward (toward depth, larger index) ----
    for i in range(idx_start + 1, n_layers):

        z_boundary = layers[i]["z_max"]


        F_prev = analytic_F(z_boundary, C0, layers[i-1])
        y_boundary = F_prev + C1[i-1]


        F_new = analytic_F(z_boundary, C0, layers[i])
        C1[i] = y_boundary - F_new


    return C1


def build_y_field(C0, x_start, z_array, layers, C1=None):
    """
    Compute the horizontal ray trajectory y(z).

    Given a ray parameter and a starting position, this function evaluates
    the horizontal coordinate of the ray at a set of depth values.

    Parameters
    ----------
    C0 : float
        Ray parameter.
    x_start : tuple
        Starting coordinates (y_start, z_start).
    z_array : array_like
        Depth values at which to evaluate the trajectory.
    layers : list of dict
        Layer definitions.
    C1 : ndarray, optional
        Precomputed offsets for each layer. If not provided, they are
        calculated internally.

    Returns
    -------
    tuple
        y : ndarray
            y-coordinates corresponding to `z_array`.
        layer_idx : ndarray
            Layer index for each depth.
        C1 : ndarray
            Offsets used for each layer.

    Examples
    --------
    >>> build_y_field(0.5, (0, -50), [-100, -50, -10], LAYERS)
    (array([1.1, 1.3, 1.5]), array([2, 1, 0]), array([0.5, 0.8, 1.2]))
    """

    z_array = np.asarray(z_array)

    # 1. compute layer index for each z
    layer_idx = get_layer_indices(z_array, layers)

    #print("Layer index distribution:")
    #for i in range(len(layers)):
    #    print(f"  Layer {i}: {np.sum(layer_idx == i)} points")

    # 2. compute offsets
    if C1 is None:
        C1 = compute_all_offsets(C0, x_start, layers)

    # 3. compute y
    y = np.zeros_like(z_array)

    for i, L in enumerate(layers):
        mask = layer_idx == i
        if np.any(mask):
            F_vals = analytic_F(z_array[mask], C0, L)
            y[mask] = F_vals + C1[i]

    return y, layer_idx, C1


def find_z_turn(C0, layers):
    """
    Find the turning point depth where the ray reflects due to total internal refraction.
    The turning point occurs where the refractive index equals
    ``1 / C0``.

    Parameters
    ----------
    C0 : float
        Ray parameter.
    layers : list of dict
        Ice layer definitions.

    Returns
    -------
    float
        Depth of turning point. Returns 0.0 if no turning point exists.

    Examples
    --------
    >>> find_z_turn(0.5, LAYERS)
    -20.0
    """

    target_n = 1.0 / C0
    
    for L in layers:
        def n(z):
            return L["n_ice"] - L["delta_n"] * np.exp(z / L["z_0"])
        
        if n(L["z_min"]) >= target_n >= n(L["z_max"]):
            z_turn = L["z_0"] * np.log(
                (L["n_ice"] - target_n) / L["delta_n"]
            )
            return z_turn
    
    return 0.0 # no turning

def evaluate_y(C0, C1, z, layers):
    """
    Evaluate y(z) at a given depth using precomputed offsets.

    Parameters
    ----------
    C0 : float
        Ray parameter.
    C1 : array
        Offsets per layer.
    z : float
        Depth to evaluate.
    layers : list of dict
        Ice layer definitions.

    Returns
    -------
    float
        y-coordinate at depth `z`.

    Examples
    --------
    >>> evaluate_y(0.5, C1, -50, LAYERS)
    1.23
    """
    idx = get_layer_indices(z, layers)
    F_val = analytic_F(z, C0, layers[int(idx)])
    return F_val + C1[idx]

def get_turning_point(C0, x1, layers, C1=None):
    """
    Compute the coordinates of the turning point (y, z) of a ray with C0 starting from x1.

    Parameters
    ----------
    C0 : float
        Ray parameter.
    x1 : tuple
        Start coordinates (y_start, z_start).
    layers : list of dict
        Ice layer definitions.
    C1 : array, optional
        Precomputed offsets.

    Returns
    -------
    tuple
        y_turn : float
            y-coordinate of turning point.
        z_turn : float
            z-coordinate of turning point.

    Examples
    --------
    >>> get_turning_point(0.5, (0, -50), LAYERS)
    (1.23, -20.0)
    """

    if C1 is None:
        C1 = compute_all_offsets(C0, x1, layers)

    z_turn = find_z_turn(C0, layers)
    if z_turn is not None:
        if z_turn > 0: 
            z_turn = 0
        
        y_turn = evaluate_y(C0, C1, z_turn, layers)
        
    else: 
        y_turn = None

    return y_turn , z_turn


def evaluate_y_with_mirror(C0, C1, z_array, layers):
    """
    Evaluate the ray trajectory including reflection at the turning point.

    If a turning point exists, the trajectory above the turning point
    is mirrored to represent the refracted portion of the ray path.

    Parameters
    ----------
    C0 : float
        Ray parameter.
    C1 : array
        Offsets per layer.
    z_array : array_like
        Depths to evaluate.
    layers : list of dict
        Ice layer definitions.

    Returns
    -------
    ndarray
        y-coordinates including mirrored values after the turning point.

    Examples
    --------
    >>> evaluate_y_with_mirror(0.5, C1, [-50, -10, 0], LAYERS)
    array([1.2, 1.4, 1.6])
    """
    z_array = np.asarray(z_array)
    y = np.zeros_like(z_array, dtype=float)

    # 1. compute turning point (scalar)
    z_turn = find_z_turn(C0, layers)

    if z_turn is None:
        # No turning point: everything is direct
        return evaluate_y(C0, C1, z_array, layers)

    # 2. compute y at turning point
    y_turn = evaluate_y(C0, C1, z_turn, layers)

    # 3. vectorized mirroring logic
    direct_mask = z_array <= z_turn
    reflected_mask = ~direct_mask

    # 3a. direct points
    if np.any(direct_mask):
        y[direct_mask] = evaluate_y(C0, C1, z_array[direct_mask], layers)

    # 3b. mirrored/reflected points
    if np.any(reflected_mask):
        z_mirror = 2*z_turn - z_array[reflected_mask]
        y[reflected_mask] = 2*y_turn - evaluate_y(C0, C1, z_mirror, layers)

    return y


def get_delta_y(C0, x1, x2, layers, C0range=(-1.0,-1.0)):
    """
    Compute the horizontal difference between the ray trajectory and a target point.

    This function evaluates how far the analytic ray path deviates from
    the desired target position ``x2``. It is used as the objective function
    when solving for valid ray tracing solutions.

    Parameters
    ----------
    C0 : float
        Ray parameter.
    x1 : tuple
        Start coordinates (y_start, z_start).
    x2 : tuple
        Target coordinates (y_target, z_target).
    layers : list of dict
        Ice layer definitions.
    C0range : tuple, optional
        Allowed range of C0 values. Default is (-1.0, -1.0) (automatic).

    Returns
    -------
    float
        Difference in y between ray and target. Returns -inf if C0 is out of range.

    Examples
    --------
    >>> get_delta_y(0.5, (0,-50), (1,0), LAYERS)
    0.123
    """
    C_0_first = C0

    if C0range[0] == -1.0 and C0range[1] == -1.0:
        C0range = (1. / get_layer_params(-2000,layers)['n_ice'], np.inf)
    else:
        C0range = (float(C0range[0]), float(C0range[1]))
    Corange_array = np.array(C0range ,  dtype=np.float64)
    if((C_0_first < Corange_array[0]) or(C_0_first > Corange_array[1])):
        return -np.inf
    

    # determine y translation first
    C1  = compute_all_offsets(C0,x1,layers)

    # for a given c_0, 3 cases are possible to reach the y position of x2
    # 1) direct ray, i.e., before the turning point
    # 2) refracted ray, i.e. after the turning point but not touching the surface
    # 3) reflected ray, i.e. after the ray reaches the surface

    y_turn, z_turn = get_turning_point(C0, x1, layers, C1)
    if z_turn is not None:
        if(z_turn < x2[1]):  # turning points is deeper that x2 positions, can't reach target
            # the minimizer has problems finding the minimum if inf is returned here. Therefore, we return the distance
            # between the turning point and the target point + 10 x the distance between the z position of the turning points
            # and the target position. This results in a objective function that has the solutions as the only minima and
            # is smooth in C_0

            diff = ((z_turn - x2[1]) ** 2 + (y_turn - x2[0]) ** 2) ** 0.5 + 10 * np.abs(z_turn - x2[1])
            return -diff

        if(y_turn > x2[0]):  # we always propagate from left to right
            # direct ray

            y2_fit = evaluate_y(C_0_first,C1,x2[1],layers)
            diff = (x2[0] - y2_fit)

            return diff
        else:
            # now it's a bit more complicated. we need to transform the coordinates to
            # be on the mirrored part of the function

            z_mirrored = x2[1]
            y2_raw = evaluate_y(C_0_first,C1,z_mirrored,layers)
            y2_fit = 2 * y_turn - y2_raw
            diff = (x2[0] - y2_fit)

            return -1 * diff
        


def get_C0_from_log(logC0,n_ice):
    """
    Transform the optimization parameter from log-space to C0.

    This transformation improves numerical stability when fitting ray
    parameters.

    
    Parameters
    ----------
    logC0 : float
        Logarithmic fit parameter.
    n_ice : float
        Refractive index of deep ice.

    Returns
    -------
    float
        Linear C0 value.

    Examples
    --------
    >>> get_C0_from_log(0.1, 1.78)
    1.789
    """
    return np.exp(logC0) + 1. / n_ice

def get_C0_from_theta(z_start, layers, theta):
    """
    Compute the ray parameter C0 from a launch angle.

    Parameters
    ----------
    z_start : float
        Start depth.
    layers : list of dict
        Ice layer definitions.
    theta : float
        Launch angle in radians.

    Returns
    -------
    float
        Corresponding C0 value.

    Examples
    --------
    >>> get_C0_from_theta(-50, LAYERS, np.pi/6)
    0.57
    """
    n_start = get_refractive_index([z_start], layers)
    p = n_start * np.sin(np.pi/2-theta)
    C0 = 1/p

    #if not np.isinf(C0):
    #    C0 = n_start - 1

    return C0
    
def get_skim_angle(x1, layers, zskim = 0.0):

    """
    Compute the launch angle required for a ray to skim a certain depth.

    The ray arrives horizontally at the plane at zskim (90° angle).

    Parameters
    ----------
    x1 : tuple
        Start coordinates (y, z).
    layers : list of dict
        Ice layer definitions.
    zskim : float, optional
        Depth of surface to skim (default 0.0).

    Returns
    -------
    tuple
        C0crit : float
            C0 of critical angle.
        thcrit : float
            Critical angle in radians.

    Examples
    --------
    >>> get_skim_angle((0, -50), LAYERS)
    (0.5, 1.57)
    """

    nlaunch = get_refractive_index([x1[1]],layers)
    
    nsurf = get_refractive_index([zskim],layers)

    sinthcrit = nsurf / nlaunch

    if sinthcrit <= 1:
        # ray goes from point with high optical thickness to point with lower optical thickness,
        # i.e. ray bending is towards horizontal
        thcrit = np.arcsin(sinthcrit)
        C0crit = get_C0_from_theta(x1[1],layers,thcrit)
    else:
        # ray goes from point with low optical thickness to point with higher optical thickness,
        # i.e. ray bending is towards vertical, no solution. returning small angle.
        thcrit = np.pi/1e12
        C0crit = None


    return C0crit, thcrit


def obj_delta_y_sqr(logC_0, x1, x2, layers, n_deep):
    """
    Objective function used in root finding for ray solutions.

    This function returns the squared horizontal mismatch between the
    predicted ray endpoint and the target point.

    Parameters
    ----------
    logC_0 : float
        Logarithmic ray parameter used by the optimizer.
    x1 : ndarray
        Starting position.
    x2 : ndarray
        Target position.
    layers : list of dict
        Layer definitions.
    n_deep : float
        Refractive index in deep ice.
    reflection : int, optional
        Reflection configuration flag.
    reflection_case : int, optional
        Reflection mode.

    Returns
    -------
    float
        Squared difference between ray and target position.
    """
    C_0 = get_C0_from_log(logC_0, n_deep)
    return get_delta_y(C_0, x1, x2, layers, (-1.0,-1.0)) ** 2

def obj_delta_y(logC_0, x1, x2, layers, n_deep):
    """
    Objective function returning the horizontal mismatch of the ray path.

    This function is used during root finding to determine ray
    parameters that connect two points.

    Parameters
    ----------
    logC_0 : float
        Logarithmic ray parameter.
    x1 : ndarray
        Starting position.
    x2 : ndarray
        Target position.
    layers : list of dict
        Layer definitions.
    n_deep : float
        Deep ice refractive index.

    Returns
    -------
    float
        Horizontal difference between the ray trajectory and target point.
    """
    C_0 = get_C0_from_log(logC_0, n_deep)
    return get_delta_y(C_0, x1, x2, layers, (-1.0,-1.0))

def determine_solution_type(x1, x2, C0, layers):
    """
    Determine the physical type of a ray tracing solution.

    
    Parameters
    ----------
    x1 : array_like
        Start coordinates (y, z).
    x2 : array_like
        End coordinates (y, z).
    C0 : float
        Ray parameter.
    layers : list of dict
        Ice layer definitions.

    Returns
    -------
    int
        Identifier for the solution type:

        * 1 : direct
        * 2 : refracted
        * 3 : reflected

    Examples
    --------
    >>> determine_solution_type((0,-50), (1,0), 0.5, LAYERS)
    1
    """
    y_turn, z_turn = get_turning_point(C0, x1, layers)
    if(x2[0] < y_turn):
        return solution_types_revert['direct']
    else:
        if(z_turn == 0):
            return solution_types_revert['reflected']
        else:
            return solution_types_revert['refracted']

def find_solutions(x1, x2, layers):
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
    layers : list of dict
        Ice layer definitions.

    Returns
    -------
    list of dict
            List of solutions. Each entry contains

            - ``type`` : solution type
            - ``C0`` : ray parameter
            - ``D`` : logarithmic parameter used in optimization
            - ``x1`` : starting position



    Examples
    --------
    >>> find_solutions((0,-50), (1,0), LAYERS)
    [{'type': 1, 'C0': 0.5, 'D': 0.1, 'x1': (0,-50)}]
    """

    # calculate optimal start value. The objective function becomes infinity if the turning point is below the z
    # position of the observer. We calculate the corresponding value so that the minimization starts at one edge
    # of the objective function
    # c = self.__b ** 2 / 4 - (0.5 * self.__b - np.exp(x2[1] / self.medium.z_0) * self.medium.n_ice) ** 2
    # C_0_start = (1 / (self.medium.n_ice ** 2 - c)) ** 0.5
    # R.L. March 15, 2019: This initial condition does not find a solution for e.g.:
    # emitter  at [-400.0*units.m,-732.0*units.m], receiver at [0., -2.0*units.m]

    tol = 1e-6
    results = []
    C0s = []

    n_deep = get_layer_params(-2000,layers)['n_ice']



    ## Here something is still wrong
    ## theta skim goes to inf for too horizontal geometries when z1 is on the same height as z2.

    _, theta_skim = get_skim_angle(x1,layers, x2[1])

    C0skim = get_C0_from_theta(x1[1],layers,theta_skim)
    #print(f"theta_skim: {theta_skim} ----> C0skim: {C0skim}")

    logC0skim = np.log(C0skim-1./n_deep)

    #obj_delta_y_sqr = obj_delta_y_square
    result = optimize.root(obj_delta_y_sqr, x0=logC0skim, args=(np.array(x1), np.array(x2),layers, n_deep), tol=tol)
    print(f"result of root otimization with C0 {get_C0_from_log(result.x[0],n_deep)}: {result}")
    if(result.fun < 1e-7):
        if(np.round(result.x[0], 3) not in np.round(C0s, 3)):
            C_0 = get_C0_from_log(result.x[0],n_deep)
            C0s.append(C_0)
            solution_type = determine_solution_type(x1, x2, C_0, layers)
            
            results.append({'type': solution_type,
                            'C0': C_0,
                            'D' : result.x[0],
                            'x1': x1})

    # check if another solution with higher logC0 exists
    logC0_start = result.x[0] + 0.0001
    logC0_stop = 100
    delta_start = obj_delta_y(logC0_start, x1, x2,layers, n_deep)
    delta_stop = obj_delta_y(logC0_stop, x1, x2, layers, n_deep)

    if(np.sign(delta_start) != np.sign(delta_stop)):

        result2 = optimize.brentq(obj_delta_y, logC0_start, logC0_stop, args=(x1, x2, layers, n_deep))

        if(np.round(result2, 3) not in np.round(C0s, 3)):
            C_0 = get_C0_from_log(result2,n_deep)
            C0s.append(C_0)
            solution_type = determine_solution_type(x1, x2, C_0, layers)

            results.append({'type': solution_type,
                            'C0': C_0,
                            'D' : result2,
                            'x1': x1})
    else:
        print("no solution with logC0 > {:.3f} exists".format(result.x[0]))

    
    theta_min =  1e-4
    C0theta_min = get_C0_from_theta(x1[1],layers,theta_min)
    #print(f"C0start from theta_min {np.rad2deg(theta_min):.4f} deg: {C0theta_min}")
    logC0_start = np.log(C0theta_min - 1. / n_deep)
    #logC0_start = -100.
    #print("logC0_Start: ",logC0_start)
    
    
    logC0_stop = result.x[0] - 0.0001
    delta_start = obj_delta_y(logC0_start, x1, x2, layers, n_deep)
    delta_stop = obj_delta_y(logC0_stop, x1, x2, layers, n_deep)
    if(np.sign(delta_start) != np.sign(delta_stop)):
        print("solution with logC0 < {:.3f} exists".format(result.x[0]))
        result3 = optimize.brentq(obj_delta_y, logC0_start, logC0_stop, args=(x1, x2, layers, n_deep))


        if(np.round(result3, 5) not in np.round(C0s, 5)):
            C_0 = get_C0_from_log(result3, n_deep)
            C0s.append(C_0)
            solution_type = determine_solution_type(x1, x2, C_0, layers)

            print("found {} solution C0 = {:.2f}".format(solution_types[solution_type], C_0))
            results.append({'type': solution_type,
                            'C0': C_0,
                            'D' : result3,
                            'x1': x1})
    else:
        print("no solution with logC0 < {:.3f} exists".format(result.x[0]))


    return sorted(results, key=itemgetter('type', 'C0'))


def get_path(C0, x1, x2, layers, n_points=2000):
    """
    Compute the analytic ray trajectory between two points.

    This function constructs the ray path for a given ray parameter ``C0``.
    If a turning point exists (where the ray bends upward due to the
    refractive index gradient), the trajectory is mirrored to generate
    the refracted branch.

    Parameters
    ----------
    C0 : float
        Ray parameter controlling the curvature of the trajectory.
    x1 : tuple
        Starting coordinate ``(y, z)``.
    x2 : tuple
        Target coordinate ``(y, z)``.
    layers : list of dict
        Layer definitions describing the refractive index profile.
    n_points : int, optional
        Number of points used for the forward integration branch.
        Default is 2000.

    Returns
    -------
    y_path : ndarray
        Horizontal coordinates of the ray path.
    z_path : ndarray
        Depth coordinates of the ray path.

    Notes
    -----
    If a turning point occurs, the function constructs the refracted branch
    by mirroring the forward trajectory around the turning point.

    The returned trajectory stops once the horizontal coordinate reaches
    the receiver position.

    Examples
    --------
    >>> y_path, z_path = get_path(
    ...     C0=0.5,
    ...     x1=(0, -500),
    ...     x2=(100, -50),
    ...     layers=LAYERS
    ... )
    >>> len(y_path)
    4000
    """

    y1, z1 = x1
    y2, z2 = x2

    z_turn = find_z_turn(C0, layers)

    # ---------- build forward branch ----------
    if z_turn is None:
        z_forward = np.linspace(z1, z2, n_points)
    else:
        z_forward = np.linspace(z1, z_turn, n_points)

    y_forward, _, _ = build_y_field(C0, x1, z_forward, layers)

    # ---------- direct ray ----------
    if z_turn is None:
        y_path = y_forward
        z_path = z_forward

    # ---------- turning ray ----------
    else:

        y_turn, _, _ = build_y_field(C0, x1, np.array([z_turn]), layers)

        y_mirror = 2*y_turn - y_forward

        z_up = z_forward[::-1]
        y_up = y_mirror[::-1]

        y_path = np.concatenate([y_forward, y_up])
        z_path = np.concatenate([z_forward, z_up])

    # ---------- stop at receiver y2 ----------
    dy = y_path - y2
    cross = np.where(np.diff(np.sign(dy)) != 0)[0]

    if len(cross) == 0:
        return y_path, z_path

    i = cross[0]

    # linear interpolation to exact endpoint
    t = (y2 - y_path[i]) / (y_path[i+1] - y_path[i])
    z_hit = z_path[i] + t * (z_path[i+1] - z_path[i])

    y_path = np.concatenate([y_path[:i+1], [y2]])
    z_path = np.concatenate([z_path[:i+1], [z_hit]])

    return y_path, z_path
