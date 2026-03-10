
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

def get_layer_params_old(z):

    if z > -14.9:
        # Snow
        return {
            "n_ice": 1.51188,
            "delta_n": 0.271579,
            "z_0": 1/0.114553,
            "z_min": -14.9,
            "z_max": 0.0,
            "region": "snow",
            "region_name" : "Snow"
        }

    elif z > -80.5:
        # Firn
        return {
            "n_ice": 1.89957,
            "delta_n": 0.529715,
            "z_0": 1/0.0129175,
            "z_min": -80.5,
            "z_max": -14.9,
            "region": "firn",
            "region_name" : "Firn"
        }

    else:
        # Bubbly ice
        return {
            "n_ice": 1.77468,
            "delta_n": 1.41573,
            "z_0": 1/0.0387882,
            "z_min": -1500.0,
            "z_max": -80.5,
            "region": "bubbly_ice",
            "region_name" : "Ice"
        }

def get_layer_params(z, layers):
    """
    Return the layer definition corresponding to a given depth.

    The function searches through a list of layer dictionaries and returns
    the one whose depth interval contains the input depth.

    Args:
        z (float):
            Depth value in meters.
        layers (list of dict):
            List of layer definitions containing `z_min` and `z_max`.

    Returns:
        dict:
            The layer dictionary that contains the given depth.

    Raises:
        ValueError:
            If the depth is outside all defined layer ranges.

    Example:
        >>> layer = get_layer_params(-50, LAYERS)
        >>> layer["region"]
        'firn'
    """

    for layer in layers:
        if layer["z_min"] <= z <= layer["z_max"]:
            return layer
    raise ValueError(f"z={z} is outside the defined layer ranges.")

def get_layer_indices(z_array, layers):
    """
    Determine which layer each depth value belongs to.

    The function supports both scalar and array inputs and returns the
    corresponding layer indices.

    Args:
        z_array (float or array-like):
            Depth value(s).
        layers (list of dict):
            List of layer definitions.

    Returns:
        int or numpy.ndarray:
            Layer index (or array of indices) corresponding to each depth.

    Example:
        >>> get_layer_indices([-10, -100], LAYERS)
        array([0, 2])
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
    Compute the refractive index n(z) for a given depth or array of depths.

    The refractive index is calculated using an exponential model within
    each layer.

    Args:
        z (float or array-like):
            Depth value(s).
        layers (list of dict):
            List of layer definitions.

    Returns:
        float or numpy.ndarray:
            Refractive index value(s) corresponding to the input depth(s).

    Example:
        >>> get_refractive_index(-100, LAYERS)
        1.77
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
    Evaluate the analytic ray tracing integral F(z) for a given layer.

    This function computes the analytic solution used to reconstruct the
    horizontal ray trajectory.

    Args:
        z (float or array-like):
            Depth value(s).
        C_0 (float):
            Ray parameter controlling the trajectory curvature.
        layer (dict):
            Layer definition containing refractive index parameters.

    Returns:
        float or numpy.ndarray:
            Value of the analytic function F(z).
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
    Compute integration offsets for all layers for a given ray parameter.

    The offsets ensure that the analytic ray solution remains continuous
    when crossing layer boundaries.

    Args:
        C0 (float):
            Ray parameter.
        x_start (tuple):
            Starting position `(y, z)` of the ray.
        layers (list of dict):
            Layer definitions.

    Returns:
        numpy.ndarray:
            Array of offset values `C1` for each layer.
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
    Compute the horizontal ray coordinate y(z) for a set of depths.

    The function evaluates the analytic ray solution across multiple
    layers and applies the appropriate layer offsets.

    Args:
        C0 (float):
            Ray parameter.
        x_start (tuple):
            Starting coordinate `(y, z)`.
        z_array (array-like):
            Depth values where the trajectory should be evaluated.
        layers (list of dict):
            Layer definitions.
        C1 (array-like, optional):
            Precomputed layer offsets.

    Returns:
        tuple:
            (y, layer_idx, C1)

            y : numpy.ndarray
                Horizontal coordinates corresponding to `z_array`.

            layer_idx : numpy.ndarray
                Layer index for each depth value.

            C1 : numpy.ndarray
                Offset values used for each layer.
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
    Determine the turning point depth of a ray.

    The turning point occurs where the refractive index satisfies
    n(z) = 1 / C0.

    Args:
        C0 (float):
            Ray parameter.
        layers (list of dict):
            Layer definitions.

    Returns:
        float:
            Depth of the turning point. Returns 0.0 if no turning point exists.
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
    Evaluate the ray trajectory y(z) at a given depth.

    Args:
        C0 (float):
            Ray parameter.
        C1 (array-like):
            Layer offsets.
        z (float):
            Depth value.
        layers (list of dict):
            Layer definitions.

    Returns:
        float:
            Horizontal ray coordinate y(z).
    """
    idx = get_layer_indices(z, layers)
    F_val = analytic_F(z, C0, layers[int(idx)])
    return F_val + C1[idx]

def get_turning_point(C0, x1, layers, C1=None):
    """
    Compute the horizontal and vertical coordinates of the ray turning point.

    Args:
        C0 (float):
            Ray parameter.
        x1 (tuple):
            Starting coordinate `(y, z)`.
        layers (list of dict):
            Layer definitions.
        C1 (array-like, optional):
            Precomputed layer offsets.

    Returns:
        tuple:
            (y_turn, z_turn)

            y_turn : float
                Horizontal coordinate of the turning point.

            z_turn : float
                Depth of the turning point.
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
    Evaluate y(z) while accounting for ray reflection at the turning point.

    If the ray passes the turning point, the trajectory is mirrored to
    represent the refracted/reflected path.

    Args:
        C0 (float):
            Ray parameter.
        C1 (array-like):
            Layer offsets.
        z_array (array-like):
            Depth values.
        layers (list of dict):
            Layer definitions.

    Returns:
        numpy.ndarray:
            Horizontal ray coordinates corresponding to the input depths.
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
    the desired endpoint.

    Args:
        C0 (float):
            Ray parameter.
        x1 (tuple):
            Starting coordinate `(y, z)`.
        x2 (tuple):
            Target coordinate `(y, z)`.
        layers (list of dict):
            Layer definitions.
        C0range (tuple, optional):
            Allowed range for C0.

    Returns:
        float:
            Difference between predicted and target horizontal position.
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

    Args:
        logC0 (float):
            Logarithmic optimization parameter.
        n_ice (float):
            Refractive index in deep ice.

    Returns:
        float:
            Ray parameter C0.
    """
    return np.exp(logC0) + 1. / n_ice

def get_C0_from_theta(z_start, layers, theta):
    """
    Compute the ray parameter C0 from a launch angle.

    Args:
        z_start (float):
            Launch depth.
        layers (list of dict):
            Layer definitions.
        theta (float):
            Launch angle in radians.

    Returns:
        float:
            Ray parameter C0.
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

    Args:
        x1 (tuple):
            Starting position `(y, z)`.
        layers (list of dict):
            Layer definitions.
        zskim (float, optional):
            Depth used for the calculation.

    Returns:
        tuple:
            (C0crit, thcrit)

            C0crit : float
                Ray parameter corresponding to the critical angle.

            thcrit : float
                Critical launch angle in radians.
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


def obj_delta_y_sqr( logC_0, x1, x2, layers, n_deep):
    """
    Objective function used in root finding for ray solutions.

    This function returns the squared horizontal mismatch between the
    predicted ray endpoint and the target point.

    Args:
        logC_0 (float):
            Optimization parameter in log-space.
        x1 (array-like):
            Start coordinate.
        x2 (array-like):
            End coordinate.
        layers (list of dict):
            Layer definitions.
        n_deep (float):
            Deep ice refractive index.

    Returns:
        float:
            Squared horizontal difference.
    """
    C_0 = get_C0_from_log(logC_0, n_deep)
    return get_delta_y(C_0, x1, x2, layers, (-1.0,-1.0)) ** 2

def obj_delta_y( logC_0, x1, x2, layers, n_deep):
    """
Objective function returning the horizontal mismatch of the ray path.

Args:
    logC_0 (float):
        Optimization parameter in log-space.
    x1 (array-like):
        Start coordinate.
    x2 (array-like):
        End coordinate.
    layers (list of dict):
        Layer definitions.
    n_deep (float):
        Deep ice refractive index.

Returns:
    float:
        Horizontal difference between predicted and target position.
"""
    C_0 = get_C0_from_log(logC_0, n_deep)
    return get_delta_y(C_0, x1, x2, layers, (-1.0,-1.0))

def determine_solution_type(x1, x2, C0, layers):
    """
    Determine the physical type of a ray tracing solution.

    Args:
        x1 (array-like):
            Start coordinate `(y, z)`.
        x2 (array-like):
            End coordinate `(y, z)`.
        C0 (float):
            Ray parameter.
        layers (list of dict):
            Layer definitions.

    Returns:
        int:
            Solution type identifier:

            * 1 — direct ray
            * 2 — refracted ray
            * 3 — reflected ray
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

    Args:
        x1 (tuple):
            Start coordinate `(y, z)`.
        x2 (tuple):
            End coordinate `(y, z)`.
        layers (list of dict):
            Layer definitions.

    Returns:
        list of dict:
            List of ray solutions containing:

            type : int
                Solution type (direct, refracted, reflected)

            C0 : float
                Ray parameter

            D : float
                Optimization parameter

            x1 : tuple
                Start coordinate
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
    Compute the ray trajectory (y,z) between two points using analytic ray tracing.

    This function builds the ray path for a given ray parameter C0,
    accounting for turning points and mirroring if the ray bends back
    before reaching the target.

    Args:
        C0 (float):
            Ray parameter controlling trajectory curvature.
        x1 (tuple):
            Start coordinate `(y, z)`.
        x2 (tuple):
            End coordinate `(y, z)`.
        layers (list of dict):
            List of layer definitions (each containing `n_ice`, `delta_n`, `z_0`, etc.).
        n_points (int, optional):
            Number of points to use in the forward integration branch. Default is 2000.

    Returns:
        tuple:
            y_path : numpy.ndarray
                Horizontal coordinates of the ray path.
            z_path : numpy.ndarray
                Depth coordinates of the ray path.

    Example:
        >>> y_path, z_path = get_path(C0=0.5, x1=(0, -500), x2=(100, -50), layers=LAYERS)
        >>> len(y_path)
        4000  # forward + mirrored points if turning occurs
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

        y_mirror = mirror(y_forward, y_turn)

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
