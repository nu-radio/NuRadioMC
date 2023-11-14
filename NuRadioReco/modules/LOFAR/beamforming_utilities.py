import numpy as np
import radiotools.helper as hp

from scipy.optimize import fmin_powell
from scipy import constants

from NuRadioReco.utilities import units


lightspeed=constants.c * units.m / units.s


def mini_beamformer(fft_data, frequencies, positions, direction):
    """
    Adapted from PyCrTools hBeamformBlock
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
    n_antennas = len(delay)
    output = np.zeros([len(frequencies)], dtype=complex)

    for a in np.arange(n_antennas):
        real = 1.0 * np.cos(2 * np.pi * frequencies * delay[a])
        imag = 1.0 * np.sin(2 * np.pi * frequencies * delay[a])

        de = real + 1j * imag
        output = output + fft_data[a] * de

    return output


def geometric_delays(ant_pos, sky):
    distance = np.sqrt(sky[0] ** 2 + sky[1] ** 2 + sky[2] ** 2)
    delays = (np.sqrt(
        (sky[0] - ant_pos[0]) ** 2 + (sky[1] - ant_pos[1]) ** 2 + (sky[2] - ant_pos[2]) ** 2) - distance) / lightspeed
    return delays


def GeometricDelayFarField(position, direction, length):
    delay = (direction[0] * position[0] + direction[1] * position[1] + direction[2] * position[2]) / length / lightspeed
    return delay


def directionFitBF(fft_data, frequencies, antpos, start_direction, maxiter):
    def negative_beamed_signal(direction):
        print('direction: ', direction)

        theta = direction[0]
        phi = direction[1]
        direction_cartesian = hp.spherical_to_cartesian(theta, phi)
        delays = geometric_delays(antpos, direction_cartesian)
        out = beamformer(fft_data, frequencies, delays)
        timeseries = np.fft.irfft(out)
        return -100 * np.max(timeseries ** 2)

    fit_direction = fmin_powell(negative_beamed_signal, np.asarray(start_direction), maxiter=maxiter, xtol=1.0)

    theta = fit_direction[0]
    phi = fit_direction[1]
    direction_cartesian = hp.spherical_to_cartesian(theta, phi)
    delays = geometric_delays(antpos, direction_cartesian)
    out = beamformer(fft_data, frequencies, delays)
    timeseries = np.fft.irfft(out)

    return fit_direction, timeseries

