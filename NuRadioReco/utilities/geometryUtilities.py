import numpy as np
from scipy import constants
from NuRadioReco.utilities import units, ice
from numpy.lib import scimath as SM
import logging
logger = logging.getLogger('NuRadioReco.geometryUtilities')


def get_time_delay_from_direction(zenith, azimuth, positions, n=1.000293):
    """
    Calculate the time delay between given positions for an arrival direction

    Parameters
    ----------

    zenith: float [rad]
        Zenith angle in convention up = 0
    azimuth: float [rad]
        Azimuth angle in convention East = 0, counter-clock-wise
    positions: array[N x 3]
        Positions on ground
    n: float (default: 1.000293)
        Index of reflection of propagation medium. By default, air is assumed

    Returns
    -------
    times: np.array of floats
        Time delays
    """
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
    m = np.array(
        [
            [np.cos(angle), -1 * np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ]
    )
    return m


def rot_x(angle):
    """Angle helper function"""
    m = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), -1 * np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ]
    )
    return m


def rot_y(angle):
    """Angle helper function"""
    m = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-1 * np.sin(angle), 0, np.cos(angle)]
        ]
    )
    return m


def get_efield_in_spherical_coords(efield, theta, phi):
    """
    Get 3D electric field from cartesian coordinates in spherical coordinates,
    using the arrival directions theta and phi

    Parameters
    ----------
    efield: np.array
        Electric field in cartesian coordinates
    theta: float
        Zenith angle of the arriving signal
    phi: float
        Azimuth angle of the arriving signal
    
    Returns
    -------
    np.array
        Electric field in spherical coordinates
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
    """ Apply Snell's law for given zenith angle, when a signal travels from n1 to n2 

    Parameters
    ----------
    zenith_incoming: float
        Zenith angle of the incoming signal
    n_2: float
        Refractive index of the medium the signal is transmitted into
    n_1: float
        Refractive index of the medium the signal is coming from

    Returns
    -------
    float
        Fresnel angle
    """
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
    to the 'plane of incident' which is definded as: "the plane of incidence
    is the plane which contains the surface normal and the propagation vector
    of the incoming radiation."

    Parameters
    ----------
    zenith_incoming: float
        Zenith angle of the incoming signal
    n_2: float
        Refractive index of the medium the signal is transmitted into
    n_1: float
        Refractive index of the medium the signal is coming from

    Returns
    -------
    float
        Fresnel coefficient t for theta (parallel) polarization
    """
    zenith_outgoing = get_fresnel_angle(zenith_incoming, n_2, n_1)
    if(zenith_outgoing is None):    #check for total internal reflection
        return 0
    t = 2 * n_1 * np.cos(zenith_incoming) / (n_1 * np.cos(zenith_outgoing) + n_2 * np.cos(zenith_incoming))
    return t


def get_fresnel_t_s(zenith_incoming, n_2=1.3, n_1=1.):
    """  returns the coefficient t which is the ratio of the transmitted wave's
    electric field amplitude to that of the incident wave for perpendicular polarization (s-wave)
    this polarization corresponds to the ePhi polarization

    parallel and perpendicular refers to the signal's polarization with respect
    to the 'plane of incident' which is definded as: "the plane of incidence
    is the plane which contains the surface normal and the propagation vector
    of the incoming radiation."

    Parameters
    ----------
    zenith_incoming: float
        Zenith angle of the incoming signal
    n_2: float
        Refractive index of the medium the signal is transmitted into
    n_1: float
        Refractive index of the medium the signal is coming from

    Returns
    -------
    float
        Fresnel coefficient t for phi (perpendicular) polarization
    """
    zenith_outgoing = get_fresnel_angle(zenith_incoming, n_2, n_1)
    if(zenith_outgoing is None):    #check for total internal reflection
        return 0
    t = 2 * n_1 * np.cos(zenith_incoming) / (n_1 * np.cos(zenith_incoming) + n_2 * np.cos(zenith_outgoing))
    return t


def get_fresnel_r_p(zenith_incoming, n_2=1.3, n_1=1.):
    """  returns the coefficient r which is the ratio of the reflected wave's
    electric field amplitude to that of the incident wave for parallel polarization (p-wave)
    this polarization corresponds to the eTheta polarization

    parallel and perpendicular refers to the signal's polarization with respect
    to the 'plane of incident' which is definded as: "the plane of incidence
    is the plane which contains the surface normal and the propagation vector
    of the incoming radiation."

    Parameters
    ----------
    zenith_incoming: float
        Zenith angle of the incoming signal
    n_2: float
        Refractive index of the medium the signal is reflected from
    n_1: float
        Refractive index of the medium the signal is coming from

    Returns
    -------
    float
        Fresnel coefficient r for theta (parallel) polarization
    """
    n = n_2 / n_1
    return np.conjugate((n**2 * np.cos(zenith_incoming) - SM.sqrt(n**2 - np.sin(zenith_incoming)**2)) / \
                        (n**2 * np.cos(zenith_incoming) + SM.sqrt(n**2 - np.sin(zenith_incoming)**2)))


def get_fresnel_r_s(zenith_incoming, n_2=1.3, n_1=1.):
    """  returns the coefficient r which is the ratio of the reflected wave's
    electric field amplitude to that of the incident wave for perpendicular polarization (s-wave)
    this polarization corresponds to the ePhi polarization

    parallel and perpendicular refers to the signal's polarization with respect
    to the 'plane of incident' which is definded as: "the plane of incidence
    is the plane which contains the surface normal and the propagation vector
    of the incoming radiation."

    Parameters
    ----------
    zenith_incoming: float
        Zenith angle of the incoming signal
    n_2: float
        Refractive index of the medium the signal is reflected from
    n_1: float
        Refractive index of the medium the signal is coming from

    Returns
    -------
    float
        Fresnel coefficient r for phi (perpendicular) polarization
    """
    n = n_2 / n_1
    return np.conjugate((np.cos(zenith_incoming) - SM.sqrt(n**2 - np.sin(zenith_incoming)**2)) / \
                        (np.cos(zenith_incoming) + SM.sqrt(n**2 - np.sin(zenith_incoming)**2)))


def fresnel_factors_and_signal_zenith(detector, station, channel_id, zenith):
    """
    Returns the zenith angle at the antenna and the fresnel coefficients t for theta (parallel) 
    and phi (perpendicular) polarization. Handles potential refraction into the firn if that 
    applies to the antenna position.

    parallel and perpendicular refers to the signal's polarization with respect
    to the 'plane of incident' which is definded as: "the plane of incidence
    is the plane which contains the surface normal and the propagation vector
    of the incoming radiation.

    Parameters
    ----------
    detector: NuRadioReco.detector.Detector
        Detector object
    station: NuRadioReco.detector.Station
        Station object
    channel_id: int
        Channel ID of the desired channel
    zenith: float
        Zenith angle of the incoming signal
    
    Returns
    -------
    zenith_antenna: float
        Zenith angle at the antenna (potentially including refraction)
    t_theta: float
        Fresnel transmission coefficient for theta polarization
    t_phi: float
        Fresnel transmission coefficient for phi polarization
    """
    n_ice = ice.get_refractive_index(-0.01, detector.get_site(station.get_id()))

    # no reflection/refraction at the ice-air boundary
    zenith_antenna = zenith
    t_theta = 1.
    t_phi = 1.

    # first check case if signal comes from above
    if zenith <= 0.5 * np.pi and station.is_cosmic_ray():
        # is antenna below surface?
        position = detector.get_relative_position(station.get_id(), channel_id)
        if position[2] <= 0:
            zenith_antenna = get_fresnel_angle(zenith, n_ice, 1)
            t_theta = get_fresnel_t_p(zenith, n_ice, 1)
            t_phi = get_fresnel_t_s(zenith, n_ice, 1)
            logger.debug(("Channel {:d}: electric field is refracted into the firn. "
                            "theta {:.0f} -> {:.0f}. Transmission coefficient p (eTheta) "
                            "{:.2f} s (ePhi) {:.2f}".format(
                                channel_id, zenith / units.deg, zenith_antenna / units.deg, t_theta, t_phi)))
    else:
        # now the signal is coming from below, do we have an antenna above the surface?
        position = detector.get_relative_position(station.get_id(), channel_id)
        if position[2] > 0:
            zenith_antenna = get_fresnel_angle(zenith, 1., n_ice)

    if zenith_antenna is None:
        logger.warning("Fresnel reflection at air-firn boundary leads to unphysical results, no reconstruction possible")
        return None, None, None

    return zenith_antenna, t_theta, t_phi
