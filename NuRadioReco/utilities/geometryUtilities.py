import numpy as np
from scipy import constants
from NuRadioReco.utilities import units
from numpy.lib import scimath as SM
import logging
logger = logging.getLogger('NuRadioReco.geometryUtilities')


def get_time_delay_from_direction(zenith, azimuth, positions, n=None):
    """
    Calculate the time delay between given positions for an arrival direction

    Parameters
    ---------

    zenith: float [rad]
        Zenith angle in convention up = 0
    azimuth: float [rad]
        Azimuth angle in convention East = 0, counter-clock-wise
    positions: array[N x 3]
        Positions on ground

    """
    if(n is None):  # assume propagation through air as default
        n = 1.000293
    shower_axis = np.array([np.sin(zenith) * np.cos(azimuth), np.sin(zenith) * np.sin(azimuth), np.cos(zenith)])

    if positions.ndim == 1:
        times = -(1 / (constants.c / n)) * np.dot(shower_axis, positions) * units.s
    else:
        times = np.zeros(positions.shape[0])
        for i in range(positions.shape[0]):
            times[i] = -(1 / (constants.c / n)) * np.dot(shower_axis, positions[i, :]) * units.s

    return times


def rot_z(angle):
    """Angle helper function"""
    m = np.array([  [np.cos(angle), -1 * np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]
                    ])
    return m


def rot_x(angle):
    """Angle helper function"""
    m = np.array([  [1, 0, 0],
                    [0, np.cos(angle), -1 * np.sin(angle)],
                    [0, np.sin(angle), np.cos(angle)]
                    ])
    return m


def rot_y(angle):
    """Angle helper function"""
    m = np.array([  [np.cos(angle), 0, np.sin(angle)],
                    [0, 1, 0],
                    [-1 * np.sin(angle), 0, np.cos(angle)]
                    ])
    return m


def get_efield_in_spherical_coords(efield, theta, phi):
    """
    Get 3D electric field from cartesian coordinates in spherical coordinates,
    using the arrival directions theta and phi
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    e1 = np.array([st * cp, st * sp, ct])
    e2 = np.array([ct * cp, ct * sp, -st])
    e3 = np.array([-sp, cp, 0])
#     e1 /= linalg.norm(e1)
#     e2 /= linalg.norm(e2)
#     e3 /= linalg.norm(e3)

    transformation_matrix = np.matrix([e1, e2, e3])
#     inverse_transformation_matrix = np.linalg.inv(transformation_matrix)

    efield_2 = np.squeeze(np.asarray(np.dot(transformation_matrix, efield)))
    return efield_2


def get_fresnel_angle(zenith_incoming, n_2=1.3, n_1=1.):
    """ Apply Snell's law for given zenith angle, when a signal travels from n1 to n2 """
    t = n_1 / n_2 * np.sin(zenith_incoming)
    if t > 1:
        logger.debug('Fresnel refraction results in unphysical values, refraction from {n1} to {n2} with incoming angle {zenith:.1f}, returning None'.format(n1=n_1, n2=n_2, zenith=np.rad2deg(zenith_incoming)))
        return None
    else:
        if(zenith_incoming > 0.5 * np.pi):
            return np.pi - np.arcsin(t)
        return np.arcsin(t)


def get_fresnel_t_p(zenith_incoming, n_2=1.3, n_1=1.):
    """  returns the coefficient t which is the ratio of the transmitted wave's
    electric field amplitude to that of the incident wave for parallel polarization (p-wave)
    this polarization corresponds to the eTheta polarization

    parallel and perpendicular refers to the signal's polarization with respect
    to the 'plane of incident' which is defindes as: "the plane of incidence
    is the plane which contains the surface normal and the propagation vector
    of the incoming radiation."
    """
    zenith_outgoing = get_fresnel_angle(zenith_incoming, n_2, n_1)
    t = 2 * n_1 * np.cos(zenith_incoming) / (n_1 * np.cos(zenith_outgoing) + n_2 * np.cos(zenith_incoming))
    return t


def get_fresnel_t_s(zenith_incoming, n_2=1.3, n_1=1.):
    """  returns the coefficient t which is the ratio of the transmitted wave's
    electric field amplitude to that of the incident wave for perpendicular polarization (s-wave)
    this polarization corresponds to the ePhi polarization

    parallel and perpendicular refers to the signal's polarization with respect
    to the 'plane of incident' which is defindes as: "the plane of incidence
    is the plane which contains the surface normal and the propagation vector
    of the incoming radiation."
    """
    zenith_outgoing = get_fresnel_angle(zenith_incoming, n_2, n_1)
    t = 2 * n_1 * np.cos(zenith_incoming) / (n_1 * np.cos(zenith_incoming) + n_2 * np.cos(zenith_outgoing))
    return t


def get_fresnel_r_p(zenith_incoming, n_2=1.3, n_1=1.):
    """  returns the coefficient r which is the ratio of the reflected wave's
    electric field amplitude to that of the incident wave for parallel polarization (p-wave)
    this polarization corresponds to the eTheta polarization

    parallel and perpendicular refers to the signal's polarization with respect
    to the 'plane of incident' which is defindes as: "the plane of incidence
    is the plane which contains the surface normal and the propagation vector
    of the incoming radiation."
    """
    n = n_2/n_1
    return (-n**2 * np.cos(zenith_incoming) + SM.sqrt(n**2 - np.sin(zenith_incoming)**2)) / \
              (n**2 * np.cos(zenith_incoming) + SM.sqrt(n**2 - np.sin(zenith_incoming)**2))


def get_fresnel_r_s(zenith_incoming, n_2=1.3, n_1=1.):
    """  returns the coefficient r which is the ratio of the reflected wave's
    electric field amplitude to that of the incident wave for perpendicular polarization (s-wave)
    this polarization corresponds to the ePhi polarization

    parallel and perpendicular refers to the signal's polarization with respect
    to the 'plane of incident' which is defindes as: "the plane of incidence
    is the plane which contains the surface normal and the propagation vector
    of the incoming radiation."
    """
    n = n_2/n_1
    return (np.cos(zenith_incoming) - SM.sqrt(n**2 - np.sin(zenith_incoming)**2)) / \
              (np.cos(zenith_incoming) + SM.sqrt(n**2 - np.sin(zenith_incoming)**2))

