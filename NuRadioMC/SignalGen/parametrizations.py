# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import numpy as np
from NuRadioReco.utilities import units
from scipy import constants
import logging
logger = logging.getLogger("SignalGen.parametrizations")


def set_log_level(level):
    logger.setLevel(level)

"""

Analytic parametrizations of the radio pulse produced by an in-ice particle shower.

Generic functions to provide the frequency spectrum and the pulse in the time domain
are defined. All models/parametrizations should be added to each of these functions,
such that different parametrizations can be exchanged by just modifying the 'model'
argument of the respective function.

"""

_random_generators = {}
_Alvarez2009_k_L = None


def get_parametrizations():
    """ returns a list of all implemented parametrizations """
    return ['ZHS1992', 'Alvarez2000', 'Alvarez2009', 'Alvarez2012']


def get_time_trace(energy, theta, N, dt, shower_type, n_index, R, model, seed=None, same_shower=False):
    """
    returns the Askaryan pulse in the time domain of the eTheta component

    We implement only the time-domain solution and obtain the frequency spectrum
    via FFT (with the standard normalization of NuRadioMC). This approach assures
    that the units are interpreted correctly. In the time domain, the amplitudes
    are well defined and not details about fourier transform normalizations needs
    to be known by the user.

    Parameters
    ----------
    energy : float
        energy of the shower
    theta: float
        viewangle: angle between shower axis (neutrino direction) and the line
        of sight between interaction and detector
    N : int
        number of samples in the time domain
    dt: float
        time bin width, i.e. the inverse of the sampling rate
    shower_type: string (default "HAD")
        type of shower, either "HAD" (hadronic), "EM" (electromagnetic)
    n_index: float
        index of refraction at interaction vertex
    R: float
        distance from vertex to observer
    model: string
        specifies the signal model
        * ZHS1992: the original ZHS parametrization from E. Zas, F. Halzen, and T. Stanev, Phys. Rev. D 45, 362 (1992), doi:10.1103/PhysRevD.45.362, this parametrization does not contain any phase information
        * Alvarez2000: parameterization based on ZHS mainly based on J. Alvarez-Muniz, R. A. Vazquez, and E. Zas, Calculation methods for radio pulses from high energyshowers,Physical Review D62 (2000) https://doi.org/10.1103/PhysRevD.84.103003
        * Alvarez2009: parameterization based on ZHS from J. Alvarez-Muniz, W. R. Carvalho, M. Tueros, and E. Zas, Coherent cherenkov radio pulses fromhadronic showers up to eev energies,Astroparticle Physics35(2012), no. 6 287 – 299 and J. Alvarez-Muniz, C. James, R. Protheroe, and E. Zas, Thinned simulations of extremely energeticshowers in dense media for radio applications, Astroparticle Physics 32 (2009), no. 2 100 – 111
    seed: None or int
        the random seed for the Askaryan modules
    same_shower: bool (default False)
        if False, for each request a new random shower realization is choosen.
        if True, the shower from the last request of the same shower type is used. This is needed to get the Askaryan
        signal for both ray tracing solutions from the same shower.

    Returns
    -------
    spectrum: array
        the complex amplitudes for the given frequencies

    """
    if(model not in _random_generators):
        _random_generators[model] = np.random.RandomState(seed)
    if(model == 'ZHS1992'):
        """ Parametrization from E. Zas, F. Halzen, and T. Stanev, Phys. Rev. D 45, 362 (1992)."""
        freqs = np.fft.rfftfreq(N, dt)
        vv0 = freqs / (0.5 * units.GHz)
        cherenkov_angle = np.arccos(1. / n_index)
        domega = (theta - cherenkov_angle)
        tmp = np.exp(+0.5j * np.pi)  # set phases to 90deg
        tmp *= 1.1e-7 * energy / units.TeV * vv0 * 1. / \
            (1 + 0.4 * (vv0) ** 2) * np.exp(-0.5 * (domega / (2.4 * units.deg / vv0)) ** 2) * \
            units.V / units.m / (R / units.m) / units.MHz
        # the factor 0.5 is introduced to compensate the unusual fourier transform normalization used in the ZHS code
        trace = 0.5 * np.fft.irfft(tmp) / dt
        trace = np.roll(trace, int(2 * units.ns / dt))
        return trace

    elif(model == 'Alvarez2009'):
        # This parameterisation is not very accurate for energies above 10 EeV
        # The ARZ model should be used instead
        freqs = np.fft.rfftfreq(N, dt)[1:]  # exclude zero frequency

        E_C = 73.1 * units.MeV
        rho = 0.924 * units.g / units.cm ** 3
        X_0 = 36.08 * units.g / units.cm ** 2
        R_M = 10.57 * units.g / units.cm ** 2
        c = constants.c * units.m / units.s

        def A(E_0, theta, freq):

            if (shower_type == 'HAD'):
                k_E_0 = 4.13e-16 * units.V / units.cm / units.MHz ** 2
                k_E_1 = 2.54
                log10_E_E = 10.60
                k_E_bar = k_E_0 * np.tanh((np.log10(E_0 / units.eV) - log10_E_E) / k_E_1)
            elif (shower_type == 'EM'):
                k_E_bar = 4.65e-16 * units.V / units.cm / units.MHz ** 2
            else:
                raise NotImplementedError("shower type {} is not implemented in Alvarez2009 model.".format(shower_type))

            return k_E_bar * E_0 / E_C * X_0 / rho * np.sin(theta) * freq

        def nu_L(E_0, theta):

            if (shower_type == 'HAD'):
                k_L_0 = 31.25
                gamma = 3.01e-2
                E_L = 1.e15 * units.eV
                k_L = k_L_0 * (E_0 / E_L) ** gamma
            elif (shower_type == 'EM'):
                sigma_0 = 3.39e-2
                log10_E_sigma = 14.99
                delta_0 = 0
                delta_1 = 2.25e-2
                log10_E_0 = np.log10(E_0 / units.eV)
                if (log10_E_0 < log10_E_sigma):
                    sigma_k_L = sigma_0 + delta_0 * (log10_E_0 - log10_E_sigma)
                else:
                    sigma_k_L = sigma_0 + delta_1 * (log10_E_0 - log10_E_sigma)

                log10_k_0 = 1.52
                log10_E_LPM = 16.61
                gamma_0 = 5.59e-2
                gamma_1 = 0.39
                if (log10_E_0 < log10_E_LPM):
                    log10_k_L_bar = log10_k_0 + gamma_0 * (log10_E_0 - log10_E_LPM)
                else:
                    log10_k_L_bar = log10_k_0 + gamma_1 * (log10_E_0 - log10_E_LPM)

                global _Alvarez2009_k_L
                if(same_shower):
                    if _Alvarez2009_k_L is None:
                        logger.error("the same shower was requested but the function hasn't been called before.")
                        raise AttributeError("the same shower was requested but the function hasn't been called before.")
                    else:
                        k_L = _Alvarez2009_k_L
                else:
                    _Alvarez2009_k_L = 10 ** _random_generators[model].normal(log10_k_L_bar, sigma_k_L)
                    k_L = _Alvarez2009_k_L
            else:
                raise NotImplementedError("shower type {} is not implemented in Alvarez2009 model.".format(shower_type))

            nu_L = rho / k_L / X_0

            cher_cut = 1.e-8
            if (np.abs(1 - n_index * np.cos(theta)) < cher_cut):
                nu_L *= c / cher_cut
            else:
                nu_L *= c / np.abs(1 - n_index * np.cos(theta))

            return nu_L

        def d_L(E_0, theta, freq):

            if (shower_type == "HAD"):
                beta = 2.57
            elif (shower_type == "EM"):
                beta = 2.74
            else:
                raise NotImplementedError("shower type {} is not implemented in Alvarez2009 model.".format(shower_type))

            return 1 / (1 + (freq / nu_L(E_0, theta)) ** beta)

        def d_R(E_0, theta, freq):

            if (shower_type == "HAD"):
                k_R_0 = 2.73
                k_R_1 = 1.72
                log10_E_R = 12.92
                k_R_bar = k_R_0 + np.tanh((log10_E_R - np.log10(E_0 / units.eV)) / k_R_1)
            elif (shower_type == "EM"):
                k_R_bar = 1.54
            else:
                raise NotImplementedError("shower type {} is not implemented in Alvarez2009 model.".format(shower_type))
            nu_R = rho / k_R_bar / R_M * c / np.sqrt(n_index ** 2 - 1)

            alpha = 1.27
            return 1 / (1 + (freq / nu_R) ** alpha)

        spectrum = A(energy, theta, freqs) * d_L(energy, theta, freqs) * d_R(energy, theta, freqs)
        spectrum *= 0.5  #  ZHS Fourier transform normalisation
        spectrum /= R
        spectrum = np.insert(spectrum, 0, 0)

        trace = np.fft.irfft(spectrum * np.exp(0.5j * np.pi)) / dt  # set phases to 90deg
        trace = np.roll(trace, len(trace) // 2)
        return trace

    elif(model == 'Alvarez2000'):
        freqs = np.fft.rfftfreq(N, dt)[1:]  # exclude zero frequency
        cherenkov_angle = np.arccos(1. / n_index)

        Elpm = 2e15 * units.eV
        dThetaEM = 2.7 * units.deg * 500 * units.MHz / freqs * (Elpm / (0.14 * energy + Elpm)) ** 0.3
#         logger.debug("dThetaEM = {}".format(dThetaEM))

        epsilon = np.log10(energy / units.TeV)
        dThetaHad = 0
        if (epsilon >= 0 and epsilon <= 2):
            dThetaHad = 500 * units.MHz / freqs * (2.07 - 0.33 * epsilon + 7.5e-2 * epsilon ** 2) * units.deg
        elif (epsilon > 2 and epsilon <= 5):
            dThetaHad = 500 * units.MHz / freqs * (1.74 - 1.21e-2 * epsilon) * units.deg
        elif(epsilon > 5 and epsilon <= 7):
            dThetaHad = 500 * units.MHz / freqs * (4.23 - 0.785 * epsilon + 5.5e-2 * epsilon ** 2) * units.deg
        elif(epsilon > 7):
            dThetaHad = 500 * units.MHz / freqs * (4.23 - 0.785 * 7 + 5.5e-2 * 7 ** 2) * \
                (1 + (epsilon - 7) * 0.075) * units.deg

        f0 = 1.15 * units.GHz
        E = 2.53e-7 * energy / units.TeV * freqs / f0 / (1 + (freqs / f0) ** 1.44)
        E *= units.V / units.m / units.MHz
        E *= np.sin(theta) / np.sin(cherenkov_angle)

        tmp = np.zeros(len(freqs) + 1)
        if(shower_type == "EM"):
            tmp[1:] = E * np.exp(-np.log(2) * ((theta - cherenkov_angle) / dThetaEM) ** 2) / R
        elif(shower_type == "HAD"):
            if(np.any(dThetaHad != 0)):
                tmp[1:] = E * np.exp(-np.log(2) * ((theta - cherenkov_angle) / dThetaHad) ** 2) / R

                def missing_energy_factor(E_0):
                    # Missing energy factor for hadronic cascades
                    # Taken from DOI: 10.1016/S0370-2693(98)00905-8
                    epsilon = np.log10(E_0 / units.TeV)
                    f_epsilon = -1.27e-2 - 4.76e-2 * (epsilon + 3)
                    f_epsilon += -2.07e-3 * (epsilon + 3) ** 2 + 0.52 * np.sqrt(epsilon + 3)
                    return f_epsilon

                tmp[1:] *= missing_energy_factor(energy)
            else:
                pass
                # energy is below a TeV, setting Askaryan pulse to zero
        else:
            raise NotImplementedError("shower type {} not implemented in {} Askaryan module".format(shower_type, model))

        tmp *= 0.5  # the factor 0.5 is introduced to compensate the unusual fourier transform normalization used in the ZHS code

#         df = np.mean(freqs[1:] - freqs[:-1])
        trace = np.fft.irfft(tmp * np.exp(0.5j * np.pi)) / dt  # set phases to 90deg
        trace = np.roll(trace, len(trace) // 2)
        return trace

    else:
        raise NotImplementedError("model {} unknown".format(model))
