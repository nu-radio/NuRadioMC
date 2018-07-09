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
    return ['ZHS1992', 'Alvarez2000', 'Alvarez2011']


def get_frequency_spectrum(energy, theta, freqs, is_em_shower, n_index, R, model):
    """
    returns the complex amplitudes of the frequency spectrum of the neutrino radio signal

    Parameters
    ----------
    energy : float
        energy of the shower
    theta: float
        viewangle: angle between shower axis (neutrino direction) and the line
        of sight between interaction and detector
    freqs : array
        frequencies, the array must be equally spaced
    is_em_shower: bool
        true if EM shower, false otherwise
    n_index: float
        index of refraction at interaction vertex
    R: float
        distance from vetex to observer
    model: string
        specifies the signal model
        * ZHS1992: the original ZHS parametrization from E. Zas, F. Halzen, and T. Stanev, Phys. Rev. D 45, 362 (1992), doi:10.1103/PhysRevD.45.362, this parametrization does not contain any phase information
        * Alvarez2000: what is in shelfmc
        * Alvarez2011: parametrization based on ZHS from Jaime Alvarez-Muñiz, Andrés Romero-Wolf, and Enrique Zas Phys. Rev. D 84, 103003, doi:10.1103/PhysRevD.84.103003. The model is implemented in pyrex and here only a wrapper around the pyrex code is implemented

    Returns
    -------
    spectrum: array
        the complex amplitudes for the given frequencies

    """
    if(model == 'ZHS1992'):
        """ Parametrization from E. Zas, F. Halzen, and T. Stanev, Phys. Rev. D 45, 362 (1992)."""
        vv0 = freqs / (0.5 * units.GHz)
        cherenkov_angle = np.arccos(1. / n_index)
        domega = (theta - cherenkov_angle) 
        tmp = 1.1e-7 * energy / units.TeV * vv0 * 1. / (1 + 0.4 * (vv0) ** 2) * np.exp(-0.5 * (domega / (2.4 * units.deg / vv0)) ** 2) * units.V / units.m / (R / units.m)
        # normalize the signal correctly
        df = np.mean(freqs[1:] - freqs[:-1])
        tmp *= (df / units.MHz) ** 0.5
        return tmp

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
        E *= units.V / units.m
        E *= np.sin(theta) / np.sin(cherenkov_angle)

        if(is_em_shower):
            tmp = E * np.exp(-np.log(2) * ((theta - cherenkov_angle) / dThetaEM) ** 2) / R
        else:
            tmp = E * np.exp(-np.log(2) * ((theta - cherenkov_angle) / dThetaHad) ** 2) / R

        # normalize the signal correctly
        df = np.mean(freqs[1:] - freqs[:-1])
        tmp *= (df / units.MHz) ** 0.5
        return tmp

    elif(model == 'Alvarez2012'):
        import pyrex.signals
        n_samples = (len(freqs) - 1) * 2
        dt = 1. / freqs.max()
        tt = np.arange(0, n_samples * dt, dt)
        ask = pyrex.signals.AskaryanSignal(tt / units.s, energy / units.GeV, theta, n_index)
        trace = ask.values * units.V / units.m * (1. * units.m / R)  # rescale to distance R, pyrex output is for 1m
        return fft.time2freq(trace)

