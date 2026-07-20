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
from NuRadioMC.SignalProp.AnalyticRayTracingImpl.MultilayerAnalyticRayTracing.corefunctions import compute_offsets, evaluate_y, get_n_1D
from NuRadioMC.SignalProp.AnalyticRayTracingImpl.MultilayerAnalyticRayTracing.getrayparameters import get_travel_time_analytic

from NuRadioMC.SignalProp.AnalyticRayTracing.maybenumba import njit


@njit(cache=True)
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

    n_air = get_n_1D(0.0001, layers)

    p = n_air * np.sin(theta_air)
    C0 = 1.0 / p

    x1 = (pos[1],pos[2])
    x2 = (0, 0)
    travel_time = get_travel_time_analytic(C0, x1, x2, layers)

    C1, _, _, _ = compute_offsets(C0, pos[1], pos[2], layers)

    horizontal_offset =  evaluate_y(C0, C1, 0, layers) - evaluate_y(C0, C1, pos[2], layers)

    return horizontal_offset, travel_time

@njit(cache=True)
def get_time_difference_plane_wave_analytic(pos1, pos2, theta_air, phi_air, layers, azimuth_convention = 'nuradio'):
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
    azimuth_convention : string
        Which azimuth convention to use ('nuradio' or 'astropy')
        'nuradio': 0deg east, 90deg north, 180 deg west, 270deg south, 'astropy': 0deg north, 90deg east, 180 deg south, 270deg west,

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
    if azimuth_convention == 'nuradio':
        src_hvec = np.array([
            np.cos(phi_air),
            np.sin(phi_air)
        ])
    elif azimuth_convention == 'astropy':
        src_hvec = np.array([
            np.sin(phi_air),
            np.cos(phi_air)
        ])
    else:
        raise ValueError(f"Azimuth convention '{azimuth_convention}' is not defined. Use 'nuradio' or 'astropy'!")


    n_air = get_n_1D(0.0001, layers)

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

    #print(f"P1: {P1}")
    #print(f"P2: {P2}")

    # vector connecting both surface intersection points
    dP = P2 - P1

    # projection of this connection vector onto src_hvec
    # surface_shift = dP if incoming wave is along dP
    # surface_shift = 0 if incoming wave is perpendicular to dP

    # difference in air travel length from projection of the surface_shift onto the actual 3D incoming ray vector
    # same as multiplying by sin(theta)

    surface_shift = np.dot(src_hvec, dP)
    delta_L_air = surface_shift * np.sin(theta_air)

    # convert to travel time
    delta_t_air = n_air * delta_L_air*units.m / constants.c

    # total relative arrival time
    delta_t = (t1 - t2) + delta_t_air

    return delta_t