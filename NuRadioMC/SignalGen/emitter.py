import numpy as np
import h5py
from scipy.interpolate import interp1d
from NuRadioReco.utilities import units, fft
import logging
logger = logging.getLogger("SignalGen.emitter")
import os


def get_time_trace(amplitude, N, dt, model, full_output=False, **kwargs):
    """
    returns the voltage trace of an emitter

    We implement only the time-domain solution and obtain the frequency spectrum
    via FFT (with the standard normalization of NuRadioMC). This approach assures
    that the units are interpreted correctly. In the time domain, the amplitudes
    are well defined and not details about fourier transform normalizations needs
    to be known by the user.

    Parameters
    ----------
    amplitude : float 
        strength of a pulse
    N : int
        number of samples in the time domain
    dt: float
        time bin width, i.e. the inverse of the sampling rate
    model: string
        specifies the signal model

        * delta_pulse: a simple signal model of a delta pulse emitter
        * cw : a sinusoidal wave of given frequency
        * square : a rectangular pulse of given amplituede and width
        * tone_burst : a short sine wave pulse of given frequency and desired width
        * idl & hvsp2 : these are the waveforms generated in KU lab and stored in hdf5 files
        * gaussian : represents a gaussian pulse where sigma is defined through the half width at half maximum

    full_output: bool (default False)
        if True, can return additional output

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
    if(model == 'delta_pulse'):  # this takes delta signal as input voltage
        trace = np.zeros(N)
        trace[N // 2] = amplitude
    elif(model == 'cw'):  # generates a sine wave of given frequency
        time = np.linspace(-(N / 2) * dt, ((N - 1) - N / 2) * dt, N)
        trace = amplitude * np.sin(2 * np.pi * emitter_frequency * time)
    elif(model == 'square' or model == 'tone_burst'):  # generates a rectangular or tone_burst signal of given width and frequency
        if(half_width > int(N / 2)):
            raise NotImplementedError(" half_width {} should be < half of the number of samples N " . format(half_width))
        time = np.linspace(-(N / 2) * dt, ((N - 1) - N / 2) * dt, N)
        voltage = np.zeros(N)
        for i in range(0, N):
            if time[i] >= -half_width and time[i] <= half_width:
                voltage[i] = amplitude
        if(model == 'square'):
            trace = voltage
        else:
            trace = voltage * np.sin(2 * np.pi * emitter_frequency * time)
    elif(model == 'gaussian'):  # generates gaussian pulse where half_width represents the half width at half maximum
        time = np.linspace(-(N / 2) * dt, ((N - 1) - N / 2) * dt, N)
        sigma = half_width / (np.sqrt(2 * np.log(2)))
        trace = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * ((time - 500) / sigma) ** 2)
        trace = amplitude * 1 / np.max(np.abs(trace)) * trace
    elif(model == 'idl' or model == 'hvsp2'):  # the idl & hvsp2 waveforms gemerated in KU Lab stored in hdf5 file
        path = os.path.dirname(os.path.dirname(__file__))
        if(model == 'idl'):
            input_file = os.path.join(path, 'data/idl_data.hdf5')
        else:
            input_file = os.path.join(path, 'data/hvsp2_data.hdf5')
        read_file = h5py.File(input_file, 'r')
        time_original = read_file.get('time')
        voltage_original = read_file.get('voltage')
        time_new = np.linspace(time_original[0], time_original[len(time_original) - 1], (int((time_original[len(time_original) - 1] - time_original[0]) / dt) + 1))
        interpolation = interp1d(time_original, voltage_original, kind='cubic')
        voltage_new = interpolation(time_new)
        # if the interpolated waveform has larger sample size than N , it will truncate the data keeping peak amplitude at center
        if len(voltage_new) > N:
            peak_amplitude_index = np.where(np.abs(voltage_new) == np.max(np.abs(voltage_new)))[0][0]
            voltage_new = np.roll(voltage_new, int(len(voltage_new) / 2) - peak_amplitude_index)
            lower_index = int(len(voltage_new) / 2 - N / 2)
            trace = voltage_new[lower_index: lower_index + N]  # this truncate data making trace lenght of N
        # for the case with larger N, trace size will be adjusted depending on whether the number (N + len(voltage_new)) is even or odd
        else:
            add_zeros = int((N - len(voltage_new)) / 2)
            adjustment = 0
            if ((N + len(voltage_new)) % 2 != 0):
                adjustment = 1
            trace = np.pad(voltage_new, (add_zeros + adjustment, add_zeros), 'constant', constant_values=(0, 0))
        trace = amplitude * trace / np.max(np.abs(trace))  # trace now has dimension of amplitude given from event generation file
        peak_amplitude_index_new = np.where(np.abs(trace) == np.max(np.abs(trace)))[0][0]
        trace = np.roll(trace, int(N / 2) - peak_amplitude_index_new)  # this rolls the array(trace) to keep peak amplitude at center

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
    amplitude : float
        strength of a pulse
    N : int
        number of samples in the time domain
    dt: float
        time bin width, i.e. the inverse of the sampling rate
    model: string
        specifies the signal model

        * delta_pulse: a simple signal model of a delta pulse emitter
        * cw : a sinusoidal wave of given frequency
        * square : a rectangular pulse of given amplituede and width
        * tone_burst : a short sine wave pulse of given frequency and desired width
        * idl & hvsp2 : these are the waveforms generated in KU lab and stored in hdf5 files
        * gaussian : represents a gaussian pulse where sigma is defined through the half width at half maximum

    full_output: bool (default False)
        if True, can return additional output

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
