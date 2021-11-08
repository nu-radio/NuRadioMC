# -*- coding: utf-8 -*-
import numpy as np
import h5py
from scipy.interpolate import interp1d
from NuRadioReco.utilities import units, fft
from NuRadioMC.SignalGen import parametrizations as par
import logging
logger = logging.getLogger("SignalGen.emitter")

def set_log_level(level):
    logger.setLevel(level)
    par.set_log_level(level)

def get_time_trace(amplitude, N, dt, model, full_output=False, **kwargs):
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
    half_width = kwargs.get("half_width")
    emitter_frequency = kwargs.get("emitter_frequency")
    trace = None
    additional_output = {}
    if(amplitude == 0):
        trace = np.zeros(3, N)
    if(model == 'spherical'):         # this takes spherical signal as input voltage
        trace = np.zeros(N)
        trace[N // 2] = amplitude
    elif(model == 'cw'):              # generates a sine wave of given frequency 
        time = np.linspace(-(N/2) * dt, ((N-1)/2) * dt , N) 
        trace = amplitude * np.sin(2 * np.pi * emitter_frequency * time)
    elif(model == 'square' or model == 'tone_burst' ):     # generates a rectangular or tone_burst signal of given width and frequency 
        if(half_width > int(N/2)):
            raise NotImplementedError(" half_width {} should be < half of the number of samples N " . format( half_width ) )
        time = np.linspace(- N * dt/(2), (N-1) * dt/(2), N)
        voltage = np.zeros(N)
        for i in range(0,N):
            if time[i] >= - half_width and time[i] <= half_width:
                voltage[i] = amplitude
        if(model == 'square'):
            trace = voltage
        else:
            trace = voltage * np.sin(2 * np.pi * emitter_frequency * time) 
    elif(model == 'idl' or model == 'hvsp2'):            # the idl & hvsp2 lab data from KU stored in hdf5 file
        if(model == 'idl'):
            read_file = h5py.File('idl_data.hdf5', 'r')
        else:
            read_file = h5py.File('hvsp2_data.hdf5', 'r')
        time_original = read_file.get('time') 
        time_new = np.linspace( time_original[0], time_original[len(time_original)-1], (int((time_original[len(time_original)-1]-time_original[0])/dt)+1))
        voltage1 = read_file.get('voltage')
        interpolation = interp1d(time_original,voltage1,kind='cubic')
        voltage2 = interpolation(time_new)
        add_zeros = int(( N-len(voltage2)) /2)
        trace = np.pad(voltage2, (add_zeros, add_zeros), 'constant', constant_values=(0, 0))
        trace = amplitude * trace /np.max(np.abs( trace ))                    # trace now has dimension of amplitude given from event generation file
        peak_amplitude_index = np.where( np.abs( trace ) == np.max( np.abs( trace ) ) )[0][0]
        trace = np.roll( trace, int(N/2) - peak_amplitude_index )             # this rolls the array(trace) to keep peak amplitude at center
    else:
        raise NotImplementedError("model {} unknown".format(model))
    if(full_output):
        return trace, additional_output
    else:
        return trace

def get_frequency_spectrum(amplitude, N, dt, model, full_output=False, **kwargs):
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
    tmp = get_time_trace(amplitude, N, dt, model, full_output=full_output, **kwargs)
    if(full_output):
        return fft.time2freq(tmp[0], 1 / dt), tmp[1]
    else:
        return fft.time2freq(tmp, 1 / dt)




