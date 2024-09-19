import numpy as np
import radiotools.helper as hp

from scipy.optimize import fmin_powell
from scipy import constants

from NuRadioReco.utilities import units


lightspeed = constants.c * units.m / units.s


def mini_beamformer(fft_data, frequencies, positions, direction):
    """
    Adapted from PyCrTools hBeamformBlock

    Beamform the spectra given the arrival direction. Based on the arrival direction,
    the geometric delays are calculated.
    """
    n_antennas = len(positions)

    output = np.zeros([len(frequencies)], dtype=complex)

    norm = np.sqrt(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2])

    for a in np.arange(n_antennas):
        delay = GeometricDelayFarField(positions[a], direction, norm)

        real = 1.0 * np.cos(2 * np.pi * frequencies * delay)
        imag = 1.0 * np.sin(2 * np.pi * frequencies * delay)

        de = real + 1j * imag
        output = output + fft_data[a] * de

    return output


def beamformer(fft_data, frequencies, delay):
    """
    Beamform the spectra according to the given delays.
    """
    n_antennas = len(delay)
    output = np.zeros([len(frequencies)], dtype=complex)

    for a in np.arange(n_antennas):
        real = 1.0 * np.cos(2 * np.pi * frequencies * delay[a])
        imag = 1.0 * np.sin(2 * np.pi * frequencies * delay[a])

        de = real + 1j * imag
        output = output + fft_data[a] * de

    return output


def geometric_delays(ant_pos, sky):
    """
    Calculate the geometric delays of given antenna position, given an arrival direction in the sky.
    """
    distance = np.sqrt(sky[0] ** 2 + sky[1] ** 2 + sky[2] ** 2)
    delays = (np.sqrt(
        (sky[0] - ant_pos[0]) ** 2 + (sky[1] - ant_pos[1]) ** 2 + (sky[2] - ant_pos[2]) ** 2) - distance) / lightspeed
    return delays


def GeometricDelayFarField(position, direction, length):
    delay = np.sum(np.asarray(direction) * np.asarray(position)) / length / lightspeed
    return delay
