# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import numpy as np
from NuRadioReco.utilities import units, fft
from NuRadioMC.SignalGen import parametrizations as par
import logging
logger = logging.getLogger("SignalGen.askaryan")


def set_log_level(level):
    logger.setLevel(level)
    par.set_log_level(level)


def get_time_trace(energy, theta, N, dt, shower_type, n_index, R, model, interp_factor=None, interp_factor2=None,
                   same_shower=False, seed=None, **kwargs):
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
        type of shower, either "HAD" (hadronic), "EM" (electromagnetic) or "TAU" (tau lepton induced)
        note that TAU showers are currently only implemented in the ARZ2019 model
    n_index: float
        index of refraction at interaction vertex
    R: float
        distance from vertex to observer
    model: string
        specifies the signal model
        * ZHS1992: the original ZHS parametrization from E. Zas, F. Halzen, and T. Stanev, Phys. Rev. D 45, 362 (1992), doi:10.1103/PhysRevD.45.362, this parametrization does not contain any phase information
        * Alvarez2000: parameterization based on ZHS mainly based on J. Alvarez-Muniz, R. A. V ́azquez, and E. Zas, Calculation methods for radio pulses from high energyshowers, Physical Review D62 (2000) https://doi.org/10.1103/PhysRevD.84.103003
        * Alvarez2009: parameterization based on ZHS from J. Alvarez-Muniz, W. R. Carvalho, M. Tueros, and E. Zas, Coherent cherenkov radio pulses fromhadronic showers up to EeV energies, Astroparticle Physics 35 (2012), no. 6 287 – 299 and J. Alvarez-Muniz, C. James, R. Protheroe, and E. Zas, Thinned simulations of extremely energeticshowers in dense media for radio applications, Astroparticle Physics 32 (2009), no. 2 100 – 111
        * HCRB2017: analytic model from J. Hanson, A. Connolly Astroparticle Physics 91 (2017) 75-89
        * ARZ2019 semi MC time domain model from Alvarez-Muñiz, J., Romero-Wolf, A., & Zas, E. (2011). Practical and accurate calculations of Askaryan radiation. Physical Review D - Particles, Fields, Gravitation and Cosmology, 84(10). https://doi.org/10.1103/PhysRevD.84.103003

    interp_factor: float or None
        controls the interpolation of the charge-excess profiles in the ARZ model
    interp_Factor2: float or None
        controls the second interpolation of the charge-excess profiles in the ARZ model
    same_shower: bool (default False)
        controls the random behviour of picking a shower from the library in the ARZ model, see description there for
        more details
    seed: None or int
        the random seed for the Askaryan modules

    Returns
    -------
    time trace: array
        the amplitudes for each time bin

    """
    if(energy == 0):
        return np.zeros(N)
    if model in par.get_parametrizations():
        return par.get_time_trace(energy, theta, N, dt, shower_type, n_index, R, model, seed=seed, same_shower=same_shower)
    elif(model == 'HCRB2017'):
        from NuRadioMC.SignalGen import HCRB2017
        is_em_shower = None
        if(shower_type == "HAD"):
            is_em_shower = False
        elif(shower_type == "EM"):
            is_em_shower = True
        else:
            raise NotImplementedError("shower type {} not implemented in {} Askaryan module".format(shower_type, model))
        LPM = True
        a = None
        if('LPM' in kwargs):
            LPM = kwargs['LPM']
        if('a' in kwargs):
            a = kwargs['a']
        return HCRB2017.get_time_trace(energy, theta, N, dt, is_em_shower, n_index, R, LPM, a)[1]
    elif(model == 'ARZ2019' or model == 'ARZ2020'):
        from NuRadioMC.SignalGen.ARZ import ARZ
        gARZ = ARZ.ARZ(arz_version=model, seed=seed)
        if(interp_factor is not None):
            gARZ.set_interpolation_factor(interp_factor)

        if(interp_factor2 is not None):
            gARZ.set_interpolation_factor2(interp_factor2)
        return gARZ.get_time_trace(energy, theta, N, dt, shower_type, n_index, R, same_shower=same_shower, **kwargs)[1]

    elif(model == 'spherical'):
        amplitude = 1. * energy / R
        trace = np.zeros(N)
        trace[N // 2] = amplitude
        return trace
    else:
        raise NotImplementedError("model {} unknown".format(model))


def get_frequency_spectrum(energy, theta, N, dt, shower_type, n_index, R, model, **kwargs):
    """
    returns the complex amplitudes of the frequency spectrum of the neutrino radio signal

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
        type of shower, either "HAD" (hadronic), "EM" (electromagnetic) or "TAU" (tau lepton induced)
        note that TAU showers are currently only implemented in the ARZ2019 model
    n_index: float
        index of refraction at interaction vertex
    R: float
        distance from vertex to observer
    model: string
        specifies the signal model
        * ZHS1992: the original ZHS parametrization from E. Zas, F. Halzen, and T. Stanev, Phys. Rev. D 45, 362 (1992), doi:10.1103/PhysRevD.45.362, this parametrization does not contain any phase information
        * Alvarez2000: parameterization based on ZHS mainly based on J. Alvarez-Muniz, R. A. V ́azquez, and E. Zas, Calculation methods for radio pulses from high energyshowers, Physical Review D62 (2000) https://doi.org/10.1103/PhysRevD.84.103003
        * Alvarez2009: parameterization based on ZHS from J. Alvarez-Muniz, W. R. Carvalho, M. Tueros, and E. Zas, Coherent cherenkov radio pulses fromhadronic showers up to EeV energies, Astroparticle Physics 35 (2012), no. 6 287 – 299 and J. Alvarez-Muniz, C. James, R. Protheroe, and E. Zas, Thinned simulations of extremely energeticshowers in dense media for radio applications, Astroparticle Physics 32 (2009), no. 2 100 – 111
        * HCRB2017: analytic model from J. Hanson, A. Connolly Astroparticle Physics 91 (2017) 75-89
        * ARZ2019 semi MC time domain model from Alvarez-Muñiz, J., Romero-Wolf, A., & Zas, E. (2011). Practical and accurate calculations of Askaryan radiation. Physical Review D - Particles, Fields, Gravitation and Cosmology, 84(10). https://doi.org/10.1103/PhysRevD.84.103003
    Returns
    -------
    spectrum: array
        the complex amplitudes for the given frequencies

    """
    return fft.time2freq(get_time_trace(energy, theta, N, dt, shower_type, n_index, R, model, **kwargs), 1 / dt)
