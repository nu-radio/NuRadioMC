# -*- coding: utf-8 -*-
import numpy as np
from NuRadioReco.utilities import units, fft
from NuRadioMC.SignalGen import parametrizations as par
import logging
logger = logging.getLogger("SignalGen.askaryan")


def set_log_level(level):
    logger.setLevel(level)
    par.set_log_level(level)


def get_time_trace(energy, theta, N, dt, shower_type, n_index, R, model, full_output=False, **kwargs):
    """
    returns the electric field of an emitter

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
        * spherical: a simple signal model of a spherical delta pulse emitter
    full_output: bool (default False)
        if True, askaryan modules can return additional output

    Returns
    -------
    time trace: 2d array, shape (3, N)
        the amplitudes for each time bin
    additional information: dict
        only available if `full_output` enabled

    """
    shower_type = shower_type.upper()  # some convenience to be sloppy with upper/lower case
    trace = None
    additional_output = {}
    if(energy == 0):
        trace = np.zeros(3, N)
    if(model == 'spherical'):
        amplitude = 1. * energy / R
        trace = np.zeros(3, N)
        trace[1, N // 2] = amplitude / 0.5 ** 2  # equal amplitude in eTheta (p-polarization) and ePhi (s-polarization)
        trace[2, N // 2] = amplitude / 0.5 ** 2
    else:
        raise NotImplementedError("model {} unknown".format(model))
    if(full_output):
        return trace, additional_output
    else:
        return trace


def get_frequency_spectrum(energy, theta, N, dt, shower_type, n_index, R, model, full_output=False, **kwargs):
    """
    returns the complex amplitudes of the frequency spectrum of an emitter

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
        * spherical: a simple signal model of a spherical delta pulse emitter
    full_output: bool (default False)
        if True, askaryan modules can return additional output

    Returns
    -------
    time trace: 2d array, shape (3, N)
        the amplitudes for each time bin
    additional information: dict
        only available if `full_output` enabled

    """
    tmp = get_time_trace(energy, theta, N, dt, shower_type, n_index, R, model, full_output=full_output, **kwargs)
    if(full_output):
        return fft.time2freq(tmp[0], 1 / dt), tmp[1]
    else:
        return fft.time2freq(tmp, 1 / dt)
