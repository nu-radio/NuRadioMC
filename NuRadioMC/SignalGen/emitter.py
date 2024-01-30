import numpy as np
import h5py
from scipy.interpolate import interp1d
from radiotools import helper as hp
from NuRadioReco.utilities import units, fft
import NuRadioReco.framework.base_trace
import logging
logger = logging.getLogger("SignalGen.emitter")
import os

buffer_emitter_model = None


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
        If the model string starts with "efield_", the function provides the three dimensional electric field emitted
        by the pulser/antena combination normalized to a distance of 1m.
        If not, then the voltage of the pulser is returned (which needs to be folded with an antenna response pattern to obtain
        the emitted electric field. This is automatically done in a NuRadioMC simulation).

        * delta_pulse: a simple signal model of a delta pulse emitter
        * cw : a sinusoidal wave of given frequency
        * square : a rectangular pulse of given amplituede and width
        * tone_burst : a short sine wave pulse of given frequency and desired width
        * idl1 & hvsp1 : these are the waveforms generated in KU lab and stored in hdf5 files
        * gaussian : represents a gaussian pulse where sigma is defined through the half width at half maximum
        * ARA02-calPulser : a new normalized voltage signal which depicts the original CalPulser shape used in ARA-02
        * efield_idl1_spice: direct measurement of the efield from the idl1 pulser and its antenna as used in the SPICE
          calibration campaigns from 2018 and 2019. 
          The `launch_vector` needs to be specified in the kwargs. See Journal of Instrumentation 15 (2020) P09039,
          doi:10.1088/1748-0221/15/09/P09039 arXiv:2006.03027 for details.
        * efield_delta_pulse: a simple signal model of a delta pulse emitter. The kwarg `polarization` needs
          to be specified to select the polarization of the efield, defined as float between 0 and 1 with
          0 = eTheta polarized and 1 = ePhi polarized. The default is 0.5, i.e. unpolarized. The amplitudes are
          set to preserve the total power of the delta pulse, i.e. A_theta = sqrt(1-polarization)
          and A_phi = sqrt(polarization).
          Use kwarg `iN` to select a specific pulse from the 10 available pulses. The default is a random selection.
    full_output: bool (default False)
        if True, can return additional output

    Returns
    -------
    time trace: 1d or 2d array, shape (N) or (3, N) for efield
        the amplitudes for each time bin. In case of an efield, the the amplitude for the three componente eR, eTheta, ePhi are returned.
    additional information: dict
        only available if `full_output` enabled

    """
    trace = None
    additional_output = {}
    if(amplitude == 0):
        if(model.startswith("efield_")):
            trace = np.zeros((3, N))
        else:
            trace = np.zeros(N)
    if(model == 'delta_pulse'):  # this takes delta signal as input voltage
        trace = np.zeros(N)
        trace[N // 2] = amplitude
    elif(model == 'cw'):  # generates a sine wave of given frequency
        time = np.linspace(-(N / 2) * dt, ((N - 1) - N / 2) * dt, N)
        trace = amplitude * np.sin(2 * np.pi * kwargs["emitter_frequency"] * time)
    elif(model == 'square' or model == 'tone_burst'):  # generates a rectangular or tone_burst signal of given width and frequency
        half_width = kwargs.get("half_width")
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
            trace = voltage * np.sin(2 * np.pi * kwargs["emitter_frequency"] * time)
    elif(model == 'gaussian'):  # generates gaussian pulse where half_width represents the half width at half maximum
        time = np.linspace(-(N / 2) * dt, ((N - 1) - N / 2) * dt, N)
        sigma = kwargs["half_width"] / (np.sqrt(2 * np.log(2)))
        trace = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * ((time - 500) / sigma) ** 2)
        trace = amplitude * 1 / np.max(np.abs(trace)) * trace
    elif(model == 'idl1' or model == 'hvsp1' or model == 'ARA02_calPulser'):  # the idl1 & hvsp1 waveforms gemerated in KU Lab stored in hdf5 file
        path = os.path.dirname(os.path.dirname(__file__))
        if(model == 'idl1'):
            input_file = os.path.join(path, 'data/idl1_data.hdf5')
        elif(model == 'hvsp1'):
            input_file = os.path.join(path, 'data/hvsp1_data.hdf5')
        else:
            input_file = os.path.join(path, 'data/ARA02_Cal_data.hdf5')
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
    elif(model == 'efield_delta_pulse'):  # this takes delta signal as input voltage
        trace = np.zeros((3, N))
        trace[1, N // 2] = (1.0 - kwargs.get("polarization", 0.5)) ** 0.5 * amplitude
        trace[2, N // 2] = kwargs.get("polarization", 0.5) ** 0.5 * amplitude
    elif(model == "efield_idl1_spice"):
        launch_zenith, _ = hp.cartesian_to_spherical(*kwargs["launch_vector"])
        iN = None

        if model not in buffer_emitter_model:
            path = os.path.dirname(os.path.dirname(__file__))
            launch_angles = np.array([0, 15, 30, 45, 60, 75, 90]) * units.deg
            buffer_emitter_model[model] = {}
            for launch_angle in launch_angles:
                buffer_emitter_model[model][launch_angle] = []
                for i in range(0, 10):
                    input_file = os.path.join(path,
                        f'SignalProp/examples/birefringence_examples/SPice_pulses/eField_launchAngle_{(90*units.deg - launch_angle) / units.deg:.0f}_set_{i}.npy')
                    buffer_emitter_model[model][launch_angle].append(np.load(input_file))

        launch_angles = np.array(list(buffer_emitter_model[model].keys()))
        launch_angle = launch_angles[np.argmin(np.abs(launch_angles - launch_zenith))]
        n_pulses = len(buffer_emitter_model[model][launch_angle])

        if "iN" in kwargs:
            iN = kwargs["iN"]
            if iN >= n_pulses:
                raise ValueError(f"the selected pulse iN {iN} is out of range. Only {n_pulses} different pulses are available")
        else:
            iN = np.randint(0, n_pulses)

        spice_pulse = buffer_emitter_model[model][launch_angle][iN]

        time_original = spice_pulse[0]
        voltage_original_theta = spice_pulse[1]
        voltage_original_phi = spice_pulse[2]
        n_samples_tmp = len(time_original)
        sampling_rate_tmp = 1/(time_original[1] - time_original[0])
        trace = NuRadioReco.framework.base_trace.BaseTrace(n_samples_tmp)
        trace.set_trace(np.array(np.zeros_like(voltage_original_theta), voltage_original_theta, voltage_original_phi), sampling_rate_tmp)
        trace.resample(N, dt)  # this resamples the trace to have N samples with dt sampling rate
        voltage_theta_new = trace.get_trace[1]
        voltage_phi_new = trace.get_trace[2]

        if len(voltage_theta_new) > N:
            peak_amplitude_index_theta = np.where(np.abs(voltage_theta_new) == np.max(np.abs(voltage_theta_new)))[0][0]
            voltage_theta_new = np.roll(voltage_theta_new, int(len(voltage_theta_new) / 2) - peak_amplitude_index_theta)
            lower_index = int(len(voltage_theta_new) / 2 - N / 2)
            trace_theta = voltage_theta_new[lower_index: lower_index + N]  # this truncate data making trace lenght of N
        # for the case with larger N, trace size will be adjusted depending on whether the number (N + len(voltage_new)) is even or odd
        else:
            add_zeros = int((N - len(voltage_theta_new)) / 2)
            adjustment = 0
            if ((N + len(voltage_theta_new)) % 2 != 0):
                adjustment = 1
            trace_theta = np.pad(voltage_theta_new, (add_zeros + adjustment, add_zeros), 'constant', constant_values=(0, 0))

        if len(voltage_phi_new) > N:
            peak_amplitude_index_phi = np.where(np.abs(voltage_phi_new) == np.max(np.abs(voltage_phi_new)))[0][0]
            voltage_phi_new = np.roll(voltage_phi_new, int(len(voltage_phi_new) / 2) - peak_amplitude_index_phi)
            lower_index = int(len(voltage_phi_new) / 2 - N / 2)
            trace_phi = voltage_phi_new[lower_index: lower_index + N]  # this truncate data making trace lenght of N
        # for the case with larger N, trace size will be adjusted depending on whether the number (N + len(voltage_new)) is even or odd
        else:
            add_zeros = int((N - len(voltage_phi_new)) / 2)
            adjustment = 0
            if ((N + len(voltage_phi_new)) % 2 != 0):
                adjustment = 1
            trace_phi = np.pad(voltage_phi_new, (add_zeros + adjustment, add_zeros), 'constant', constant_values=(0, 0))

        #trace_theta = amplitude * trace_theta / np.max(np.abs(trace_theta))  # trace now has dimension of amplitude given from event generation file
        peak_amplitude_index_theta_new = np.where(np.abs(trace_theta) == np.max(np.abs(trace_theta)))[0][0]
        trace_theta = np.roll(trace_theta, int(N / 2) - peak_amplitude_index_theta_new)

        trace_phi = amplitude * trace_phi / np.max(np.abs(trace_phi))  # trace now has dimension of amplitude given from event generation file
        peak_amplitude_index_phi_new = np.where(np.abs(trace_phi) == np.max(np.abs(trace_phi)))[0][0]
        trace_phi = np.roll(trace_phi, int(N / 2) - peak_amplitude_index_phi_new)

        trace = np.zeros((3, N))
        trace[1,:] = trace_theta
        trace[2,:] = trace_phi

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
        If the model string starts with "efield_", the function provides the three dimensional electric field emitted
        by the pulser/antena combination normalized to a distance of 1m. 
        If not, then the voltage of the pulser is returned (which needs to be folded with an antenna response pattern to obtain
        the emitted electric field. This is automatically done in a NuRadioMC simulation).  

        * delta_pulse: a simple signal model of a delta pulse emitter
        * cw : a sinusoidal wave of given frequency
        * square : a rectangular pulse of given amplituede and width
        * tone_burst : a short sine wave pulse of given frequency and desired width
        * idl1 & hvsp1 : these are the waveforms generated in KU lab and stored in hdf5 files
        * gaussian : represents a gaussian pulse where sigma is defined through the half width at half maximum
        * ARA02-calPulser : a new normalized voltage signal which depicts the original CalPulser shape used in ARA-02
    full_output: bool (default False)
        if True, can return additional output

    Returns
    -------
    time trace: 1d or 2d array, shape (N) or (3, N) for efield
        the amplitudes for each time bin. In case of an efield, the the amplitude for the three componente eR, eTheta, ePhi are returned.
    additional information: dict
        only available if `full_output` enabled
    """
    tmp = get_time_trace(amplitude, N, dt, model, full_output=full_output, **kwargs)
    if(full_output):
        return fft.time2freq(tmp[0], 1 / dt), tmp[1]
    else:
        return fft.time2freq(tmp, 1 / dt)
