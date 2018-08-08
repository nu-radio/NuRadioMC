from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from scipy import constants
from NuRadioReco.utilities import units
import NuRadioReco.framework.channel
import logging
logger = logging.getLogger('geometryUtilities')


def get_time_delay_from_direction(zenith, azimuth, positions, n=None):

    if(n is None):  # assume propagation through air as default
        n = 1.000293
    shower_axis = np.array([np.sin(zenith) * np.cos(azimuth), np.sin(zenith) * np.sin(azimuth), np.cos(zenith)])

    if positions.ndim == 1:
        times = -(1 / (constants.c / n)) * np.dot(shower_axis, positions) * units.s
    else:
        times = np.zeros(positions.shape[0])
        for i in xrange(positions.shape[0]):
            times[i] = -(1 / (constants.c / n)) * np.dot(shower_axis, positions[i, :]) * units.s

    return times


# def convert_xyz_to_onsky(vec, azimuth, zenith):
#     """ Rotate the electric field from on site to on sky coordinates """
#     """ Returns vector in E_theta, E_phi """
#     ct = np.cos(zenith)
#     st = np.sin(zenith)
#     cp = np.cos(azimuth)
#     sp = np.sin(azimuth)
#
#     rotation_xyz2pt = np.array([[   ct * cp, ct * sp, -1 * st],
#                                 [-1 * sp, cp, 0],
#                                 [   st * cp, st * sp, ct]])
#
#     vec_n = np.dot(rotation_xyz2pt, vec)
#     return vec_n
#
# def convert_onsky_to_xyz(vec, azimuth, zenith):
#     """ Rotate the electric field from on sky coordinates to on site """
#
#     ct = np.cos(zenith)
#     st = np.sin(zenith)
#     cp = np.cos(azimuth)
#     sp = np.sin(azimuth)
#
#     rotation_pt2xyz = np.array([[   ct * cp, -1.*sp, st * cp],
#                                 [   ct * sp, cp, st * sp],
#                                 [-1 * st, 0    , ct]])
#
#     vec_n = np.dot(rotation_pt2xyz, vec)
#     return vec_n
#
#
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
        logger.warning('Fresnel refraction results in unphysical values, refraction from {n1} to {n2} with incoming angle {zenith:.1f}, returning None'.format(n1=n_1, n2=n_2, zenith=np.rad2deg(zenith_incoming)))
        return None
    else:
        if(zenith_incoming > 0.5 * np.pi):
            return np.pi - np.arcsin(t)
        return np.arcsin(t)


def get_fresnel_t_perpendicular(zenith_incoming, n_2=1.3, n_1=1.):
    """  returns the coefficient t which is the ratio of the transmitted wave's
    electric field amplitude to that of the incident wave for perpendicular polarization (p-wave)

    parallel and perpendicular refers to the signal's polarization with respect
    to the 'plane of incident' which is defindes as: "the plane of incidence
    is the plane which contains the surface normal and the propagation vector
    of the incoming radiation."
    """
    zenith_outgoing = get_fresnel_angle(zenith_incoming, n_2, n_1)
    t = 2 * n_1 * np.cos(zenith_incoming) / (n_1 * np.cos(zenith_outgoing) + n_2 * np.cos(zenith_incoming))
    return t


def get_fresnel_t_parallel(zenith_incoming, n_2=1.3, n_1=1.):
    """  returns the coefficient t which is the ratio of the transmitted wave's
    electric field amplitude to that of the incident wave for parallel polarization (s-wave)

    parallel and perpendicular refers to the signal's polarization with respect
    to the 'plane of incident' which is defindes as: "the plane of incidence
    is the plane which contains the surface normal and the propagation vector
    of the incoming radiation."
    """
    zenith_outgoing = get_fresnel_angle(zenith_incoming, n_2, n_1)
    t = 2 * n_1 * np.cos(zenith_incoming) / (n_1 * np.cos(zenith_incoming) + n_2 * np.cos(zenith_outgoing))
    return t


def get_fresnel_r_perpendicular(zenith_incoming, n_2=1.3, n_1=1.):
    """  returns the coefficient r which is the ratio of the reflected wave's
    electric field amplitude to that of the incident wave for perpendicular polarization (p-wave)

    parallel and perpendicular refers to the signal's polarization with respect
    to the 'plane of incident' which is defindes as: "the plane of incidence
    is the plane which contains the surface normal and the propagation vector
    of the incoming radiation."
    """
    zenith_outgoing = get_fresnel_angle(zenith_incoming, n_2, n_1)
    if(zenith_outgoing is None):  # we have total internal reflection
        return 1
    r = (n_2 * np.cos(zenith_incoming) - n_1 * np.cos(zenith_outgoing)) / (n_2 * np.cos(zenith_incoming) + n_1 * np.cos(zenith_outgoing))
    return r


def get_fresnel_r_parallel(zenith_incoming, n_2=1.3, n_1=1.):
    """  returns the coefficient r which is the ratio of the reflected wave's
    electric field amplitude to that of the incident wave for parallel polarization (s-wave)

    parallel and perpendicular refers to the signal's polarization with respect
    to the 'plane of incident' which is defindes as: "the plane of incidence
    is the plane which contains the surface normal and the propagation vector
    of the incoming radiation."
    """
    zenith_outgoing = get_fresnel_angle(zenith_incoming, n_2, n_1)
    if(zenith_outgoing is None):  # we have total internal reflection
        return 1
    r = (n_1 * np.cos(zenith_incoming) - n_2 * np.cos(zenith_outgoing)) / (n_1 * np.cos(zenith_incoming) + n_2 * np.cos(zenith_outgoing))
    return r

