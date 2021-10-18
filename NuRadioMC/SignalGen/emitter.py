# -*- coding: utf-8 -*-
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from NuRadioReco.utilities import units, fft
from NuRadioMC.SignalGen import parametrizations as par
import logging
logger = logging.getLogger("SignalGen.emitter")
plotDir = "./Signals/" 
if (not os.path.exists(plotDir)):
    os.makedirs(plotDir)

def set_log_level(level):
    logger.setLevel(level)
    par.set_log_level(level)


def get_time_trace(amplitude, N, dt, model,half_width,freq, full_output=False, **kwargs):
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
    trace = None
    additional_output = {}
    if(amplitude == 0):
        trace = np.zeros(3, N)
    if(model == 'spherical'):
        trace = np.zeros(N)
        trace[N // 2] = amplitude
    elif(model=='cw'):
        f=freq*units.GHz
        t=np.linspace(0,(N-1/2)*dt,N)*units.ns
        trace=amplitude*np.sin(2*np.pi*f*t)
    elif(model=='square'):
        t=np.linspace(-N*dt/(2),(N-1)*dt/(2),N)*units.ns
        shift=half_width*units.ns
        Amplitude=1*units.volt
        Voltage=np.zeros(N)
        for i in range(0,N):
            if t[i]>=-shift and t[i]<=shift:
                Voltage[i]=Amplitude
        trace=amplitude*Voltage
        plt.plot(t,trace)
        plt.title('Time Domain Signal')
        plt.ylabel('Voltage(Volts)')
        plt.xlabel('time(ns)')
        plt.savefig(str(plotDir) + "/Vt" + ".png", bbox_inches = "tight")
        plt.close()        

    elif(model=='tone_burst'):
        width=2*half_width*units.ns
        f=freq*units.GHz
        t=np.linspace(-(width*dt),(width*dt),int(2*width))*units.ns
        Voltage=amplitude*np.sin(2*np.pi*f*t)
        number=int((N-2*width)/2)
        trace=np.pad(Voltage, (number, number), 'constant', constant_values=(0, 0))
        time=np.linspace(-N*dt/2, (N-1)*dt/2, N)
        plt.plot(time, trace)
        plt.savefig(str(plotDir) + "Vt" +".png", bbox_inches = "tight")
        plt.clf()
        plt.close()
        
    elif(model=='hvsp2'):
        hf = h5py.File('hvsp2_final_data.hdf5', 'r')
        Voltage = hf.get('dataset_1')
        time= hf.get('dataset_2')
        trace= amplitude*Voltage
        plt.plot(time,trace)
        plt.title('Time Domain Signal')
        plt.ylabel('Voltage(Volts)')
        plt.xlabel('time(ns)')
        plt.savefig(str(plotDir) + "/Vt" + ".png", bbox_inches = "tight")
        plt.close()

    elif(model=='idl'):
        hf = h5py.File('idl_final_data.hdf5', 'r')
        Voltage = hf.get('dataset_1')
        trace = amplitude*Voltage
        time= hf.get('dataset_2')
        plt.plot(time,trace)
        plt.title('Time Domain Signal')
        plt.ylabel('Voltage(Volts)')
        plt.xlabel('time(ns)')
        plt.savefig(str(plotDir) + "/Vt" + ".png", bbox_inches = "tight")
        plt.close()
    else:
        raise NotImplementedError("model {} unknown".format(model))
    if(full_output):
        return trace, additional_output
   
    
    else:
        return trace


def get_frequency_spectrum(amplitude, N, dt, model,half_width,freq, full_output=False, **kwargs):
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
    tmp = get_time_trace(amplitude, N, dt, model, half_width,freq, full_output=full_output, **kwargs)
    if(full_output):
        return fft.time2freq(tmp[0], 1 / dt), tmp[1]
    else:
        Vf=fft.time2freq(tmp, 1 / dt)
        freqs=np.fft.rfftfreq(N,dt)
        plt.plot(freqs,np.abs(Vf))
        plt.title('Voltage Domain Signal')
        plt.ylabel('Voltage(Volts)')
        plt.xlabel('freqs(GHz)')
        plt.savefig(str(plotDir) + "/Vf" + ".png", bbox_inches = "tight")
        plt.close()
        return fft.time2freq(tmp, 1 / dt)

plt.clf()
