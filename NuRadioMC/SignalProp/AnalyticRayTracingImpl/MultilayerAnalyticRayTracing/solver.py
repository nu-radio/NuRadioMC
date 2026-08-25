import numpy as np
from scipy import optimize
from operator import itemgetter

from NuRadioMC.SignalProp.AnalyticRayTracingImpl.MultilayerAnalyticRayTracing.corefunctions import layers_to_arrays, compute_offsets, get_delta_y, get_n_1D, get_c0_from_theta, get_skim_angle, determine_solution_type



def get_c0_from_log_scalar(logc0, n_ice):
    """
    Convert the logarithmic optimization parameter to a ray parameter.

    Parameters
    ----------
    logc0 : float
        Optimization parameter used during root finding.

    n_ice : float
        Refractive index of deep ice.

    Returns
    -------
    float
        Ray parameter c0.

    Notes
    -----
    The transformation

        c0 = exp(logc0) + 1 / n_ice

    ensures that c0 remains larger than the minimum allowed value
    during optimization.
    """
    return float(np.exp(logc0) + 1. / n_ice)

def get_c0_from_log(logc0, n_ice):

    if isinstance(logc0, np.ndarray):
        logc0 = logc0[0]

    return float(get_c0_from_log_scalar(logc0, n_ice))

def obj_delta_y_sqr(logc0, y1, z1, y2, z2, layers, n_deep,
                    downgoing, with_air):
    """
    Objective function used for root-finding during ray solution search.

    Parameters
    ----------
    logc0 : float
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
    c0 = get_c0_from_log(logc0, n_deep)
    dy = get_delta_y(c0, y1, z1, y2, z2, layers, (-1., -1.),downgoing,with_air)

    if not np.isfinite(dy):
        return 1e30

    return dy*dy

def obj_delta_y(logc0, y1, z1, y2, z2, layers, n_deep,
                downgoing, with_air):
    """
    Objective function returning the horizontal mismatch of a ray.

    This function is used by root-finding algorithms to determine
    ray parameters that connect two points.

    Parameters
    ----------
    logc0 : float
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
    c0 = get_c0_from_log(logc0, n_deep)
    dy = get_delta_y(c0, y1, z1, y2, z2, layers, (-1., -1.),downgoing,with_air)

    if not np.isfinite(dy):
            return 1e30

    return dy


def find_solutions(x1, x2, layers,tol=1e-12):
    """
    Find all valid ray tracing solutions between two points. Stable solutions over layer boundary with non-smooth n(z).

    The function searches for ray parameters c0 that connect the start
    and end positions using numerical root finding.

    This function is still used, after optimizing the solution finding for better solving in the situations where emitter and receiver are on the same depth (z1=z2) always resulted in less stable solving for air-to-ice rays.
    In order to provide a stable solver for all points we use a combination of this function find_solutions and the next function find_solutions_bulk, dependent on the geometry of the situation.

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

        ``c0`` : float
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
    c0s = []
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

    theta_straight = np.arctan2((z2-z1),(y2-y1))

    if theta_straight < np.pi/4 and not with_air:
        theta_straight = np.pi/4

    _, theta_skim = get_skim_angle(
        y1, z1,
        z2,
        layers
            )

    if not np.isfinite(theta_skim):
        theta_skim = np.arctan2(z1,y1)


    c0skim = get_c0_from_theta(
        z1,
        np.abs(theta_skim),
        layers
    )

    c0straight = get_c0_from_theta(
        z1,
        np.abs(theta_straight),
        layers
    )

    n_z = get_n_1D(z1,layers)
    logc0straight = np.log(max(c0straight - 1./n_deep, 1e-12))
    logc0skim_nz = np.log(max(1/n_z - 1./n_deep, 1e-12))
    logc0skim = np.log(max(c0skim- 1./n_deep, 1e-12))

    result = optimize.root(obj_delta_y_sqr, x0=logc0straight, args=(y1,z1,y2,z2,layers, n_deep,downgoing,with_air), tol=tol)
    if(result.fun < 1e-7):
        if(np.round(result.x[0], 3) not in np.round(c0s, 3)):
            c_0 = get_c0_from_log(result.x[0],n_deep)
            c_1, _, _, _ = compute_offsets(c_0,y1, z1, layers)
            c0s.append(c_0)
            solution_type = determine_solution_type(y1,z1,y2,z2, c_0, layers,downgoing,with_air)

            results.append({'type': solution_type,
                            'C0': c_0,
                            'C1': c_1,
                            'reflection': 0,
                            'reflection_case': 0,
                            'D' : result.x[0],
                            'x1': x1,
                            'flag' : 1})
    else:

        result = optimize.root(obj_delta_y_sqr, x0=logc0skim, args=(y1,z1,y2,z2,layers, n_deep,downgoing,with_air), tol=tol)
        if(result.fun < 1e-7):
            if(np.round(result.x[0], 3) not in np.round(c0s, 3)):
                c_0 = get_c0_from_log(result.x[0],n_deep)
                c_1, _, _, _ = compute_offsets(c_0,y1, z1, layers)
                c0s.append(c_0)
                solution_type = determine_solution_type(y1,z1,y2,z2, c_0, layers,downgoing,with_air)

                results.append({'type': solution_type,
                                'C0': c_0,
                                'C1': c_1,
                                'reflection': 0,
                                'reflection_case': 0,
                                'D' : result.x[0],
                                'x1': x1,
                                'flag' : 1})


    # check if another solution with higher logc0 exists
    if result.x[0] is None:
        result_x = logc0skim_nz
    else:
        result_x = result.x[0]

    logc0_start = result_x + 0.00001

    if with_air:
        c0cross_min = 1.0
        logc0_start = np.log(max(c0cross_min - 1./n_deep, 1e-12))

    logc0_stop = 100.0

    delta_test = obj_delta_y(
        -10.,
        y1, z1, y2, z2,
        layers,
        n_deep,downgoing,with_air
        )

    delta_start = obj_delta_y(
        logc0_start,
        y1, z1, y2, z2,
        layers,
        n_deep,downgoing,with_air
        )

    delta_stop = obj_delta_y(
        logc0_stop,
        y1, z1, y2, z2,
        layers,
        n_deep,downgoing,with_air
        )


    if(np.sign(delta_start) != np.sign(delta_stop)):

        result2 = optimize.brentq(obj_delta_y, logc0_start, logc0_stop, args=(y1,z1,y2,z2,layers, n_deep,downgoing,with_air))

        if(np.round(result2, 3) not in np.round(c0s, 3)):
            c_0 = get_c0_from_log(result2,n_deep)
            c_1, _, _, _ = compute_offsets(c_0,y1, z1, layers)
            c0s.append(c_0)
            solution_type = determine_solution_type(y1,z1,y2,z2, c_0, layers,downgoing,with_air)

            results.append({'type': solution_type,
                            'C0': c_0,
                            'C1': c_1,
                            'reflection': 0,
                            'reflection_case': 0,
                            'D' : result2,
                            'x1': x1,
                            'flag' : 3})


    theta_min =  1e-5
    c0theta_min = get_c0_from_theta(
        z1,
        theta_min,
        layers
        )
    if c0theta_min <= 1/n_deep:
        c0theta_min = 1/n_deep + 1e-12  # small buffer to avoid log(0)

    logc0_start = max(np.log(c0theta_min - 1. / n_deep),-100)

    logc0_stop = result_x - 0.00001
    delta_start = obj_delta_y(
        logc0_start,
        y1, z1, y2, z2,
        layers,
        n_deep,downgoing,with_air
        )

    delta_stop = obj_delta_y(
            logc0_stop,
            y1, z1, y2, z2,
            layers,
            n_deep,downgoing,with_air
            )

    if(np.sign(delta_start) != np.sign(delta_stop)):
        result3 = optimize.brentq(obj_delta_y, logc0_start, logc0_stop, args=(y1,z1,y2,z2, layers, n_deep,downgoing,with_air))


        if(np.round(result3, 3) not in np.round(c0s, 3)):
            c_0 = get_c0_from_log(result3, n_deep)
            c_1, _, _, _ = compute_offsets(c_0,y1, z1, layers)
            c0s.append(c_0)
            solution_type = determine_solution_type(y1,z1,y2,z2, c_0, layers,downgoing,with_air)


            results.append({'type': solution_type,
                            'C0': c_0,
                            'C1': c_1,
                            'reflection': 0,
                            'reflection_case': 0,
                            'D' : result3,
                            'x1': x1,
                            'flag' : 4})

    return sorted(results, key=itemgetter('type', 'C0'))

def find_solutions_bulk(x1, x2, layers,tol=1e-12):
    """
    Find all valid ray tracing solutions between two points. Optimized for points below ice with z1 close to z2.

    The function searches for ray parameters c0 that connect the start
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

        ``c0`` : float
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
    c0s = []
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

    #theta_straight = np.arctan2(max((z2-z1),1e-14)/max((y2-y1),1e-14))
    theta_straight = abs(np.arctan2((z2-z1),(y2-y1)))

    if with_air and downgoing:
        theta_straight = np.pi/2 - 0.1

    if theta_straight < np.pi/4:
        theta_straight = np.pi/4

    #if abs(z1-z2) < 2:
    #    theta_straight = np.pi/2


    _, theta_skim = get_skim_angle(
        y1, z1,
        z2,
        layers
            )

    if not np.isfinite(theta_skim):
        theta_skim = abs(np.arctan2(z1,y1))


    c0skim = get_c0_from_theta(
        z1,
        np.abs(theta_skim),
        layers
    )

    c0straight = get_c0_from_theta(
        z1,
        np.abs(theta_straight),
        layers
    )

    c0_sixty = get_c0_from_theta(
        z1,
        np.pi/3,
        layers
    )


    n_z = get_n_1D(z1,layers)


    logc0straight = np.log(abs(c0straight - 1./n_deep))
    logc0skim_nz = np.log(max(abs(1/n_z - 1./n_deep),1e-14))
    logc0skim = np.log(abs(c0skim- 1./n_deep))
    logc0_60 = np.log(abs(c0_sixty - 1./n_deep))

    initial_guesses = [
        logc0straight,
        logc0skim,
        logc0skim_nz,
        logc0_60

    ]

    result = None


    for x0 in initial_guesses:

        if not np.isfinite(x0):
            continue

        try:
            result = optimize.root(obj_delta_y_sqr, x0=x0, args=(y1,z1,y2,z2,layers, n_deep,downgoing,with_air), tol=tol)

            c_0 = get_c0_from_log(result.x[0],n_deep)
            if(result.fun < 1e-7):
                if(np.round(c_0, 3) not in np.round(c0s, 3)):

                    c_1, _, _, _ = compute_offsets(c_0,y1, z1, layers)
                    c0s.append(c_0)
                    solution_type = determine_solution_type(y1,z1,y2,z2, c_0, layers,downgoing,with_air)

                    results.append({'type': solution_type,
                                    'C0': c_0,
                                    'C1': c_1,
                                    'reflection': 0,
                                    'reflection_case': 0,
                                    'D' : result.x[0],
                                    'x1': x1,
                                    'flag' : 50})

        except Exception:
            continue

    # check if another solution with higher logc0 exists
    if result.x[0] is None:
        result_x = logc0skim_nz
    else:
        result_x = result.x[0]

    logc0_start = result_x + 0.00001

    if with_air:
        c0cross_min = 1.0
        logc0_start = np.log(max(c0cross_min - 1./n_deep, 1e-12))

    logc0_stop = 100.0

    delta_start = obj_delta_y(
        logc0_start,
        y1, z1, y2, z2,
        layers,
        n_deep,downgoing,with_air
        )

    delta_stop = obj_delta_y(
        logc0_stop,
        y1, z1, y2, z2,
        layers,
        n_deep,downgoing,with_air
        )

    if(np.sign(delta_start) != np.sign(delta_stop)):

        result2 = optimize.brentq(obj_delta_y, logc0_start, logc0_stop, args=(y1,z1,y2,z2,layers, n_deep,downgoing,with_air))

        if(np.round(result2, 3) not in np.round(c0s, 3)):
            c_0 = get_c0_from_log(result2,n_deep)
            c_1, _, _, _ = compute_offsets(c_0,y1, z1, layers)
            c0s.append(c_0)
            solution_type = determine_solution_type(y1,z1,y2,z2, c_0, layers,downgoing,with_air)

            results.append({'type': solution_type,
                            'C0': c_0,
                            'C1': c_1,
                            'reflection': 0,
                            'reflection_case': 0,
                            'D' : result2,
                            'x1': x1,
                            'flag' : 3})


    theta_min =  1e-5
    c0theta_min = get_c0_from_theta(
        z1,
        theta_min,
        layers
        )
    if c0theta_min <= 1/n_deep:
        c0theta_min = 1/n_deep + 1e-12  # small buffer to avoid log(0)

    logc0_start = max(np.log(c0theta_min - 1. / n_deep),-100)

    logc0_stop = result_x - 0.00001
    delta_start = obj_delta_y(
        logc0_start,
        y1, z1, y2, z2,
        layers,
        n_deep,downgoing,with_air
        )

    delta_stop = obj_delta_y(
            logc0_stop,
            y1, z1, y2, z2,
            layers,
            n_deep,downgoing,with_air
            )

    if(np.sign(delta_start) != np.sign(delta_stop)):
        result3 = optimize.brentq(obj_delta_y, logc0_start, logc0_stop, args=(y1,z1,y2,z2, layers, n_deep,downgoing,with_air))


        if(np.round(result3, 3) not in np.round(c0s, 3)):
            c_0 = get_c0_from_log(result3, n_deep)
            c_1, _, _, _ = compute_offsets(c_0,y1, z1, layers)
            c0s.append(c_0)
            solution_type = determine_solution_type(y1,z1,y2,z2, c_0, layers,downgoing,with_air)


            results.append({'type': solution_type,
                            'C0': c_0,
                            'C1': c_1,
                            'reflection': 0,
                            'reflection_case': 0,
                            'D' : result3,
                            'x1': x1,
                            'flag' : 4})

    return sorted(results, key=itemgetter('type', 'C0'))

def reduce_solutions(results, with_air=False, tol = 1e-3):
    """
    Reduce a list of ray-tracing solutions by removing numerically
    equivalent solutions and limiting the total number of returned
    solutions.

    Solutions are grouped according to their c0 parameter using
    the specified tolerance. Within each group, the solution with the
    lowest flag value is retained. If more than two solutions remain
    after this reduction, only the two solutions with the highest c0
    values are kept. This suppresses unphysical low-c0 solutions
    that can appear due to numerical instabilities.

    Parameters
    ----------
    results : list of dict
        List of solution dictionaries. Each dictionary is expected
        to contain at least the keys 'C0', 'flag', and 'type'.

    tol : float, optional
        Tolerance used to group similar c0 values.
        Default is 1e-3.

    Returns
    -------
    results : list of dict
        Reduced and sorted list of solutions.
    """
    unique_results = {}

    for r in results:

        key = round(r['C0'] / tol)

        # keep result with lower flag
        if key not in unique_results or r['flag'] < unique_results[key]['flag']:
            unique_results[key] = r

    results = sorted(unique_results.values(), key=itemgetter('type', 'C0'))

    if with_air:
        results = max(results, key=itemgetter('C0'))
        return [results]

    if len(results) > 2: # keep results with higher c0 (sometimes numerical issues result in wrong solutions with low c0)
        results = sorted(results, key=itemgetter('C0'))[-2:]

    results = sorted(results, key=itemgetter('type', 'C0'))

    return results
