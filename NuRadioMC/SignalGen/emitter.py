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
    freq = kwargs.get("freq")


    if(half_width> int(N/2)):
        raise NotImplementedError("half_width {} not applicable".format(half_width))
    trace = None
    additional_output = {}
    if(amplitude == 0):
        trace = np.zeros(3, N)
    if(model == 'spherical'):
        trace = np.zeros(N)
        trace[N // 2] = amplitude
    elif(model=='cw'):   # generates a continuous signal of given frequency
        f=freq*units.GHz
        t=np.linspace(0,(N-1/2)*dt,N)*units.ns
        trace=amplitude*np.sin(2*np.pi*f*t)
    elif(model=='square'):  #generates a rectangular pulse of given width 
        t=np.linspace(-N*dt/(2),(N-1)*dt/(2),N)*units.ns
        shift=half_width*units.ns
        Voltage=np.zeros(N)
        for i in range(0,N):
            if t[i]>=-shift and t[i]<=shift:
                Voltage[i]=amplitude
        trace=Voltage*units.volt

    elif(model=='tone_burst'):  # a continuos signal that uses desired frequency and width
        width=2*half_width*units.ns
        f=freq*units.GHz
        t=np.linspace(-(width*dt),(width*dt),int(2*width))*units.ns
        Voltage=amplitude*np.sin(2*np.pi*f*t)
        add_zeros=int((N-2*width)/2)
        trace=np.pad(Voltage, (add_zeros, add_zeros), 'constant', constant_values=(0, 0))             
    elif(model=='hvsp2'):
        hf = h5py.File('hvsp2_data.hdf5', 'r')
        time=hf.get('dataset_1')
        t=np.linspace(time[0],time[len(time)-1],int(len(time)*dt))
        Voltage1=hf.get('dataset_2')
        interpolation=interp1d(time,Voltage1,kind='cubic')
        Voltage2=amplitude*interpolation(t)
        add_zeros=int((N-len(Voltage2))/2)
        trace=amplitude*np.pad(Voltage2, (add_zeros, add_zeros), 'constant', constant_values=(0, 0))
        trace=np.roll(trace,58)
    elif(model=='idl'):
        hf = h5py.File('idl_data.hdf5', 'r')
        time=hf.get('dataset_1')
        t=np.linspace(time[0],time[len(time)-1],int(len(time)*dt))
        Voltage1=hf.get('dataset_2')
        interpolation=interp1d(time,Voltage1,kind='cubic')
        Voltage2=amplitude*interpolation(t)
        add_zeros=int((N-len(Voltage2))/2)
        trace=amplitude*np.pad(Voltage2, (add_zeros, add_zeros), 'constant', constant_values=(0, 0))
        trace=np.roll(trace,77)
               
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


