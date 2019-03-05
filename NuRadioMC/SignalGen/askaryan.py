# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import numpy as np
from NuRadioMC.utilities import units, fft
from NuRadioMC.SignalGen import parametrizations as par
import logging
logger = logging.getLogger("SignalGen.askaryan")

gARZ = None

def set_log_level(level):
    logger.setLevel(level)
    par.set_log_level(level)

def get_time_trace(energy, theta, N, dt, shower_type, n_index, R, model, interp_factor=None,
                   same_shower=False):
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
        * Alvarez2000: what is in shelfmc
        * Alvarez2012: parametrization based on ZHS from Jaime Alvarez-Muñiz, Andrés Romero-Wolf, and Enrique Zas Phys. Rev. D 84, 103003, doi:10.1103/PhysRevD.84.103003. The model is implemented in pyrex and here only a wrapper around the pyrex code is implemented
        * Hanson2017: analytic model from J. Hanson, A. Connolly Astroparticle Physics 91 (2017) 75-89
        * ARZ2019 semi MC time domain model
    interp_factor: float or None
        controls the interpolation of the charge-excess profiles in the ARZ model
    same_shower: bool (default False)
        controls the random behviour of picking a shower from the library in the ARZ model, see description there for
        more details

    Returns
    -------
    spectrum: array
        the complex amplitudes for the given frequencies

    """
    if model in par.get_parametrizations():
        return par.get_time_trace(energy, theta, N, dt, shower_type, n_index, R, model)
    elif(model == 'Hanson2017'):
        from NuRadioMC.SignalGen.RalstonBuniy import askaryan_module
        is_em_shower = None
        if(shower_type == "HAD"):
            is_em_shower = False
        elif(shower_type == "EM"):
            is_em_shower = True
        else:
            raise NotImplementedError("shower type {} not implemented in {} Askaryan module".format(shower_type, model))
        return askaryan_module.get_time_trace(energy, theta, N, dt, is_em_shower, n_index, R)[1]
    elif(model == 'ARZ2019'):
        from NuRadioMC.SignalGen.ARZ import ARZ
        global gARZ
        if(gARZ is None):
            gARZ = ARZ.ARZ()
        if(interp_factor is not None):
            gARZ.set_interpolation_factor(interp_factor)
        return gARZ.get_time_trace(energy, theta, N, dt, shower_type, n_index, R, same_shower=same_shower)[1]

    elif(model == 'spherical'):
        amplitude = 1. * energy / R
        trace = np.zeros(N)
        trace[N//2] = amplitude
        return trace
    else:
        raise NotImplementedError("model {} unknown".format(model))


def get_frequency_spectrum(energy, theta, N, dt, is_em_shower, n_index, R, model, **kwargs):
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
    is_em_shower: bool
        true if EM shower, false otherwise
    n_index: float
        index of refraction at interaction vertex
    R: float
        distance from vertex to observer
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
    return fft.time2freq(get_time_trace(energy, theta, N, dt, is_em_shower, n_index, R, model, **kwargs))
