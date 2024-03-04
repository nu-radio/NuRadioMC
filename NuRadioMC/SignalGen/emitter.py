import numpy as np
import h5py
from scipy.interpolate import interp1d
from radiotools import helper as hp
from NuRadioReco.utilities import units, fft
import NuRadioReco.framework.base_trace
import logging
logger = logging.getLogger("SignalGen.emitter")
import os
import pickle, lzma

buffer_emitter_model = {}


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
          the `amplitude` is used to rescale the efield relatively, i.e., amplitude = 1 will return the measured efield amplitude, an 
          amplitude of 10 will return 10 times the measured efield amplitude, etc.
          Use kwarg `iN` to select a specific pulse from the 10 available pulses. The default is a random selection.
        * efield_delta_pulse: a simple signal model of a delta pulse emitter. The kwarg `polarization` needs
          to be specified to select the polarization of the efield, defined as float between 0 and 1 with
          0 = eTheta polarized and 1 = ePhi polarized. The default is 0.5, i.e. unpolarized. The amplitudes are
          set to preserve the total power of the delta pulse, i.e. A_theta = sqrt(1-polarization)
          and A_phi = sqrt(polarization).

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

        if model not in buffer_emitter_model:
            path = os.path.dirname(os.path.dirname(__file__))
            SPice_pulses = os.path.join(path,"SignalProp/examples/birefringence_examples/extra_files/SPice_pulses.xz")

            with lzma.open(SPice_pulses, "r") as f:
                buffer_emitter_model[model] = pickle.load(f)

        launch_keys = np.array(list(buffer_emitter_model[model]['efields'].keys()))
        launch_angles = launch_keys * units.deg

        launch_angle = launch_keys[np.argmin(np.abs(launch_angles - launch_zenith))]
        n_pulses = len(buffer_emitter_model[model]['efields'][launch_angle])

        if "iN" in kwargs:
            iN = kwargs["iN"]
            if iN >= n_pulses:
                raise ValueError(f"the selected pulse iN {iN} is out of range. Only {n_pulses} different pulses are available")
        else:
            if "rnd" in kwargs:
                iN = kwargs["rnd"].integers(0, n_pulses)
            else:
                iN = np.random.randint(0, n_pulses)
                logger.warning(f"no random number generator provided, using np.random.randint to select pulse {iN} from {n_pulses} available pulses. This might not be reproducible.")

        additional_output['iN'] = iN

        spice_pulse = buffer_emitter_model[model]['efields'][launch_angle][iN]

        voltage_original_theta = spice_pulse[0]
        voltage_original_phi = spice_pulse[1]
        sampling_rate_tmp = buffer_emitter_model[model]['sampling_rate']

        btrace = NuRadioReco.framework.electric_field.ElectricField([1], position=None,
                shower_id=None, ray_tracing_id=None)

        btrace.set_trace(np.array([np.zeros_like(voltage_original_theta), voltage_original_theta, voltage_original_phi]), sampling_rate_tmp)
        btrace.resample(1/dt)  # this resamples the trace to have 1/dt sampling rate

        trace = btrace.get_trace()

        if len(trace[1]) > N:
            peak_amplitude_index_theta = np.where(np.abs(trace[1]) == np.max(np.abs(trace[1])))[0][0]
            trace[1] = np.roll(trace[1], int(len(trace[1]) / 2) - peak_amplitude_index_theta)
            lower_index = int(len(trace[1]) / 2 - N / 2)
            final_theta = trace[1, lower_index: lower_index + N]  # this truncate data making trace lenght of N
        # for the case with larger N, trace size will be adjusted depending on whether the number (N + len(voltage_new)) is even or odd
        else:
            add_zeros = int((N - len(trace[1])) / 2)
            adjustment = 0
            if ((N + len(trace[1])) % 2 != 0):
                adjustment = 1
            final_theta = np.pad(trace[1], (add_zeros + adjustment, add_zeros), 'constant', constant_values=(0, 0))

        if len(trace[2]) > N:
            peak_amplitude_index_phi = np.where(np.abs(trace[2]) == np.max(np.abs(trace[2])))[0][0]
            trace[2] = np.roll(trace[2], int(len(trace[2]) / 2) - peak_amplitude_index_phi)
            lower_index = int(len(trace[2]) / 2 - N / 2)
            final_phi = trace[2][lower_index: lower_index + N]  # this truncate data making trace lenght of N
        # for the case with larger N, trace size will be adjusted depending on whether the number (N + len(voltage_new)) is even or odd
        else:
            add_zeros = int((N - len(trace[2])) / 2)
            adjustment = 0
            if ((N + len(trace[2])) % 2 != 0):
                adjustment = 1
            final_phi = np.pad(trace[2], (add_zeros + adjustment, add_zeros), 'constant', constant_values=(0, 0))

        #trace_theta = amplitude * trace_theta / np.max(np.abs(trace_theta))  # trace now has dimension of amplitude given from event generation file
        peak_amplitude_index_theta_new = np.where(np.abs(final_theta) == np.max(np.abs(final_theta)))[0][0]
        final_theta = np.roll(final_theta, int(N / 2) - peak_amplitude_index_theta_new)

        #trace_phi = amplitude * trace_phi / np.max(np.abs(trace_phi))  # trace now has dimension of amplitude given from event generation file
        peak_amplitude_index_phi_new = np.where(np.abs(final_phi) == np.max(np.abs(final_phi)))[0][0]
        final_phi = np.roll(final_phi, int(N / 2) - peak_amplitude_index_phi_new)

        trace = np.zeros((3,N))
        trace[1,:] = final_theta * amplitude
        trace[2,:] = final_phi * amplitude

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
