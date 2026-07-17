"""
This module contains helper functions to calculate the effective refractive
indices and the polarization vectors for a birefringent medium.
The calculations are described here: https://link.springer.com/article/10.1140/epjc/s10052-023-11238-y
"""
import numpy as np
from radiotools import helper as hp


def get_effective_index_birefringence(direction, nx, ny, nz):

    """
    Function to find the analytical solutions for the effective refractive indices.
    The calculations are described here: https://link.springer.com/article/10.1140/epjc/s10052-023-11238-y

    Parameters
    ----------
    direction: numpy.array
        Propagation direction of the wave
    nx: float
        The index of refraction in the x-direction
    ny: float
        The index of refraction in the y-direction
    nz: float
        The index of refraction in the z-direction

    Returns
    -------
    output format: np.array([n1, n2])
    meaning: effective refractive indices
    """

    sx = direction[0]
    sy = direction[1]
    sz = direction[2]

    n1 = np.sqrt((-2 * nx ** 2 * ny ** 2 * nz ** 2) /
                 (ny ** 2 * nz ** 2 * ( - 1 + sx ** 2) + nx ** 2 * (nz ** 2 * ( -1 + sy ** 2) + ny ** 2 * ( - 1 + sz ** 2))
                  - np.sqrt(4 * nx ** 2 * ny ** 2 * nz ** 2 * (nz ** 2 * ( - 1 + sx ** 2 + sy ** 2)
                                                                + ny ** 2 * (-1 + sx ** 2 + sz ** 2)
                                                                + nx ** 2 * ( - 1 + sy ** 2 + sz ** 2))
                                                                + (ny ** 2 * nz ** 2 * ( - 1 + sx ** 2)
                                                                + nx ** 2 * (nz ** 2 * ( - 1 + sy ** 2)
                                                                + ny ** 2 * ( - 1 + sz ** 2))) ** 2)))
    n2 = np.sqrt((-2 * nx ** 2 * ny ** 2 * nz ** 2) /
                 (ny ** 2 * nz ** 2 * ( - 1 + sx ** 2) + nx ** 2 * (nz ** 2 * ( -1 + sy ** 2) + ny ** 2 * ( - 1 + sz ** 2))
                  + np.sqrt(4 * nx ** 2 * ny ** 2 * nz ** 2 * (nz ** 2 * ( - 1 + sx ** 2 + sy ** 2)
                                                                + ny ** 2 * (-1 + sx ** 2 + sz ** 2)
                                                                + nx ** 2 * ( - 1 + sy ** 2 + sz ** 2))
                                                                + (ny ** 2 * nz ** 2 * ( - 1 + sx ** 2)
                                                                + nx ** 2 * (nz ** 2 * ( - 1 + sy ** 2)
                                                                + ny ** 2 * ( - 1 + sz ** 2))) ** 2)))

    return np.array([n1, n2])


def get_polarization_birefringence_simple(n, direction, nx, ny, nz):

    """
    Function for the normalized e-field vector of a wave for the direction of propagation in cartesian coordinates without special cases.
    For the function with special cases see get_polarization_birefringence.
    For a birefringent medium, the e-field vector is calculated from the diagonalized dielectric tensor and the propagation direction.
    The calculations are described here: https://link.springer.com/article/10.1140/epjc/s10052-023-11238-y

    Parameters
    ----------
    n: float
        The effective index of refraction in the propagation direction calculated by get_effective_index_birefringence
    direction: numpy.array
        Propagation direction of the wave
    nx: float
        The index of refraction in the x-direction
    ny: float
        The index of refraction in the y-direction
    nz: float
        The index of refraction in the z-direction

    Returns
    -------
    efield : np.ndarray of shape (3,)
        normalized e-field vector in cartesian coordinates
    """

    polarization = np.array([direction[0] / (n ** 2 - nx ** 2), direction[1] / (n ** 2 - ny ** 2), direction[2] / (n ** 2 - nz ** 2)])
    polarization = polarization / np.linalg.norm(polarization)

    return polarization


def get_polarization_birefringence(N1, N2, direction, nx, ny, nz, logger=None):

    """
    Function for the normalized e-field vector of a wave for the direction of propagation in spherical coordinates with special cases.
    For a birefringent medium, the e-field vector is calculated from the diagonalized dielectric tensor and the propagation direction.
    The calculations are described here: https://link.springer.com/article/10.1140/epjc/s10052-023-11238-y

    Parameters
    ----------
    N1: float
        The first effective index of refraction in the propagation direction calculated by get_effective_index_birefringence
    N2: float
        The second effective index of refraction in the propagation direction calculated by get_effective_index_birefringence
    direction: numpy.array
        Propagation direction of the wave
    nx: float
        The index of refraction in the x-direction
    ny: float
        The index of refraction in the y-direction
    nz: float
        The index of refraction in the z-direction
    logger: logging.Logger, optional
        If given, used to warn if the polarization vectors are not computable

    Returns
    -------
    efield : np.ndarray of shape (2, 3)
        normalized e-field vector in spherical coordinates for both birefringence solutions
    """

    narrow_check = 1e-9
    wide_check = 1e-10

    if (np.isclose(N1, np.array([nx, ny, nz]), rtol=0, atol=narrow_check).any()) or (np.isclose(N2, np.array([nx, ny, nz]), rtol=0, atol=narrow_check).any()):

        if (np.isclose(N1, np.array([nx, ny, nz]), rtol=0, atol=narrow_check).any()) and (np.isclose(N2, np.array([nx, ny, nz]), rtol=0, atol=narrow_check).any()):
            if logger is not None:
                logger.warning("warning: Polarization vectors not computable")
            sky_polarization_1 = np.array([0, 0, 0])
            sky_polarization_2 = np.array([0, 0, 0])

        elif np.isclose(N1, nx, rtol=0, atol=wide_check):

            if direction[0] < 0:
                sky_polarization_1 = np.array([0, 0, 1])
                sky_polarization_2 = np.array([0, 1, 0])

            else:
                sky_polarization_1 = np.array([0, 0, -1])
                sky_polarization_2 = np.array([0, 1, 0])

        elif np.isclose(N1, ny, rtol=0, atol=narrow_check):

            if direction[1] < 0:
                sky_polarization_1 = np.array([0, 0, 1])
                sky_polarization_2 = np.array([0, 1, 0])

            else:
                sky_polarization_1 = np.array([0, 0, -1])
                sky_polarization_2 = np.array([0, 1, 0])

        elif np.isclose(N2, ny, rtol=0, atol=narrow_check):

            if direction[1] < 0:
                sky_polarization_1 = np.array([0, 1, 0])
                sky_polarization_2 = np.array([0, 0, -1])

            else:
                sky_polarization_1 = np.array([0, 1, 0])
                sky_polarization_2 = np.array([0, 0, 1])

        elif np.isclose(N2, nz, rtol=0, atol=wide_check):

            if direction[2] < 0:
                sky_polarization_1 = np.array([0, 0, -1])
                sky_polarization_2 = np.array([0, -1, 0])

            else:
                sky_polarization_1 = np.array([0, 0, -1])
                sky_polarization_2 = np.array([0, 1, 0])

        else:
            polarization_1 = get_polarization_birefringence_simple(N1, direction, nx, ny, nz)
            polarization_2 = get_polarization_birefringence_simple(N2, direction, nx, ny, nz)

            zenith, azimuth = hp.cartesian_to_spherical( * (direction))
            sky_polarization_1 = on_sky_birefringence(zenith, azimuth, polarization_1)
            sky_polarization_2 = on_sky_birefringence(zenith, azimuth, polarization_2)

    else:
        polarization_1 = get_polarization_birefringence_simple(N1, direction, nx, ny, nz)
        polarization_2 = get_polarization_birefringence_simple(N2, direction, nx, ny, nz)

        zenith, azimuth = hp.cartesian_to_spherical( * (direction))
        sky_polarization_1 = on_sky_birefringence(zenith, azimuth, polarization_1)
        sky_polarization_2 = on_sky_birefringence(zenith, azimuth, polarization_2)

    return np.vstack((sky_polarization_1, sky_polarization_2))


def on_sky_birefringence(theta, phi, polarization):

    """
    Function for the normalized e-field vector from cartesian to spherical coordinates.
    The function does the same as the following radiotool functions, only faster:
    from radiotools import coordinatesystems
    cs = coordinatesystems.cstrafo(theta, phi)
    sky = cs.transform_from_ground_to_onsky(p)

    Parameters
    ----------
    theta: float
        Zenith angle of the propagation direction
    phi: float
        Azimuth angle of the propagation direction
    polarization: np.array([px, py, pz])
        Normalized e-field vector in cartesian coordinates

    Returns
    -------
    efield : np.ndarray of shape (3,)
        normalized e-field vector in spherical coordinates
    """

    transform = np.array([  [np.sin(theta) * np.cos(phi) , np.sin(theta) * np.sin(phi)   , np.cos(theta)    ],
                            [np.cos(theta) * np.cos(phi) , np.cos(theta) * np.sin(phi)   , - np.sin(theta)  ],
                            [- np.sin(phi)               , np.cos(phi)                   , 0                ]       ])

    return transform.dot(polarization)
