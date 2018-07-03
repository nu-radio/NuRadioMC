from __future__ import absolute_import, division, print_function
import numpy as np
from NuRadioMC.utilities import units, fft

"""

Analytic parametrizations of the radio pulse produced by an in-ice particle shower.

Generic functions to provide the frequency spectrum and the pulse in the time domain
are defined. All models/parametrizations should be added to each of these functions,
such that different parametrizations can be exchanged by just modifying the 'model'
argument of the respective function.

The following models are implemented
 * Alvarez2000, 10.1103/PhysRevD.62.063001

"""


def get_parametrizations():
    """ returns a list of all implemented parametrizations """
    return ['Alvarez2000', 'Alvarez2012']


def get_frequency_spectrum(energy, theta, freqs, is_em_shower, n_index, R, model):
    """
    returns the magnitude of the frequency spectrum of the neutrino radio signal

    Parameters
    ----------
    Enu : float
        energy of the shower
    theta: float or array
        viewangle: angle between shower axis (neutrino direction) and the line
        of sight between interaction and detector
    freqs : float or array
        frequency
    is_em_shower: bool
        true if EM shower, false otherwise
    n_index: float
        index of refraction at interaction vertex
    R: float
        distance from vetex to observer

    Returns
    -------
    E: float or array
        the amplitude for the given frequency

    """
    if(model == 'Alvarez2000'):
        cherenkov_angle = np.arccos(1. / n_index)

        Elpm = 2e15 * units.eV
        dThetaEM = np.deg2rad(2.7) * 500 * units.MHz / freqs * (Elpm / (0.14 * energy + Elpm)) ** 0.3

        epsilon = np.log10(energy / units.TeV)
        dThetaHad = 0
        if (epsilon >= 0 and epsilon <= 2):
            dThetaHad = 500 * units.MHz / freqs * (2.07 - 0.33 * epsilon + 7.5e-2 * epsilon ** 2)
        elif (epsilon > 2 and epsilon <= 5):
            dThetaHad = 500 * units.MHz / freqs * (1.74 - 1.21e-2 * epsilon)
        elif(epsilon > 5 and epsilon <= 7):
            dThetaHad = 500 * units.MHz / freqs * (4.23 - 0.785 * epsilon + 5.5e-2 * epsilon ** 2)
        elif(epsilon > 7):
            dThetaHad = 500 * units.MHz / freqs * (4.23 - 0.785 * 7 + 5.5e-2 * 7 ** 2) * (1 + (epsilon - 7) * 0.075)

        f0 = 1.15 * units.GHz
        E = 2.53e-7 * energy / units.TeV * freqs / f0 / (1 + (freqs / f0) ** 1.44)
        E *= units.V / units.m / units.MHz
        E *= np.sin(theta) / np.sin(cherenkov_angle)

        if(is_em_shower):
            return E * np.exp(-np.log(2) * ((theta - cherenkov_angle) / dThetaEM) ** 2) / R
        else:
            return E * np.exp(-np.log(2) * ((theta - cherenkov_angle) / dThetaHad) ** 2) / R

    elif(model == 'Alvarez2012'):
        import pyrex.signals
        n_samples = (len(freqs) - 1) * 2
        dt = 1. / freqs.max()
        tt = np.arange(0, n_samples * dt, dt)
        ask = pyrex.signals.AskaryanSignal(tt / units.s, energy / units.GeV, theta, n_index)
        trace = ask.values * units.V / units.m * (1. * units.m / R)  # rescale to distance R, pyrex output is for 1m
        return fft.time2freq(trace)

