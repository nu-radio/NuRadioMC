import numpy as np

from scipy import constants

from NuRadioReco.utilities import units


lightspeed = constants.c * units.m / units.s


def mini_beamformer(fft_data, frequencies, positions, direction):
    """
    Beamforms the spectra given the arrival direction and antenna positions.
    This function is a wrapper around the `beamformer()` function, which beamforms the
    spectra given the timedelay. Here we first calculate the geometric delays in the far
    field, based on the arrival direction.

    Parameters
    ----------
    fft_data : np.ndarray
        The Fourier transformed time traces of all antennae, shaped as (nr_of_ant, nr_of_freq_samples)
    frequencies : np.ndarray
        The values of the frequencies samples, shaped as (nr_of_freq_samples,)
    positions : np.ndarray
        The position of antenna, shaped as (nr_of_ant, 3)
    direction : np.ndarray
        The arrival direction in the sky, in cartesian coordinates, shape (3,)

    Returns
    -------
    beamformed : np.ndarray
        The beamformed (ie summed) frequency spectrum

    Notes
    -----
    Adapted from PyCrTools hBeamformBlock
    """
    delays = geometric_delay_far_field(positions, direction)

    return beamformer(fft_data, frequencies, delays)


def beamformer(fft_data, frequencies, delays):
    """
    Beamform the spectra according to the given delays, by phase shifting them according
    to the time delays and summing up the resulting spectra.

    Parameters
    ----------
    fft_data : np.ndarray
        The Fourier transformed time traces of all antennae, shaped as (nr_of_ant, nr_of_freq_samples)
    frequencies : np.ndarray
        The values of the frequencies samples, shaped as (nr_of_freq_samples,)
    delays : np.ndarray
        The delay per antenna, shaped as (nr_of_ant,)

    Returns
    -------
    beamformed : np.ndarray
        The beamformed (ie summed) frequency spectrum
    """
    real = 1.0 * np.cos(2 * np.pi * frequencies[np.newaxis, :] * delays[:, np.newaxis])
    imag = 1.0 * np.sin(2 * np.pi * frequencies[np.newaxis, :] * delays[:, np.newaxis])

    de = real + 1j * imag
    output = fft_data * de

    return np.sum(output, axis=0)  # sum all the antennas together to beamform


def geometric_delays_near_field(ant_pos, sky):
    """
    Calculate the geometric delays of given antenna position, given a source location in the sky.

    Parameters
    ----------
    ant_pos : array_like
        Antenna positions
    sky : array_like
        Source location in the sky

    Returns
    -------
    delays : array_like
        The geometric delays for all antenna positions.
    """
    delays = np.linalg.norm(ant_pos - sky, axis=1) / lightspeed

    return delays


def geometric_delay_far_field(ant_positions, direction):
    """
    Calculate the geometric delays in the far field approximation, by projecting the positions onto
    the arrival direction vector and dividing by the lightspeed.

    If the `direction` vector points towards the incoming direction (as usually calculated using
    zenith and azimuth), then positive delays correspond to later times compared to the [0, 0, 0]
    location.

    Parameters
    ----------
    ant_positions : array_like
        The positions of the antennae, with shape (nr_of_ant, 3)
    direction : array_like
        The arrival direction, in cartesian coordinates
    """
    direction_normalised = np.asarray(direction) / np.linalg.norm(direction)

    delays = np.dot(ant_positions, direction_normalised)
    delays /= -1 * lightspeed

    return delays
