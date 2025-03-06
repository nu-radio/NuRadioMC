from NuRadioReco.utilities import units, ice, geometryUtilities as geo_utl, fft
import NuRadioReco.framework.base_trace

import scipy.ndimage
from scipy.ndimage import uniform_filter1d
import numpy as np
import scipy.constants
import scipy.signal
from scipy.signal import hilbert, argrelextrema
from scipy.stats import kurtosis, entropy

import logging
logger = logging.getLogger('NuRadioReco.trace_utilities')

conversion_factor_integrated_signal = scipy.constants.c * scipy.constants.epsilon_0 * units.joule / units.s / units.volt ** 2

# see Phys. Rev. D DOI: 10.1103/PhysRevD.93.122005
# to convert V**2/m**2 * s -> J/m**2 -> eV/m**2


def get_efield_antenna_factor(station, frequencies, channels, detector, zenith, azimuth, antenna_pattern_provider):
    """
    Returns the antenna response to a radio signal coming from a specific direction

    Parameters
    ----------

    station: Station
    frequencies: array of complex
        frequencies of the radio signal for which the antenna response is needed
    channels: array of int
        IDs of the channels
    detector: Detector
    zenith, azimuth: float, float
        incoming direction of the signal. Note that refraction and reflection at the ice/air boundary are taken into account
    antenna_pattern_provider: AntennaPatternProvider
    """
    n_ice = ice.get_refractive_index(-0.01, detector.get_site(station.get_id()))
    efield_antenna_factor = np.zeros((len(channels), 2, len(frequencies)), dtype=complex)  # from antenna model in e_theta, e_phi
    for iCh, channel_id in enumerate(channels):
        zenith_antenna = zenith
        t_theta = 1.
        t_phi = 1.
        # first check case if signal comes from above
        if zenith <= 0.5 * np.pi and station.is_cosmic_ray():
            # is antenna below surface?
            position = detector.get_relative_position(station.get_id(), channel_id)
            if position[2] <= 0:
                zenith_antenna = geo_utl.get_fresnel_angle(zenith, n_ice, 1)
                t_theta = geo_utl.get_fresnel_t_p(zenith, n_ice, 1)
                t_phi = geo_utl.get_fresnel_t_s(zenith, n_ice, 1)
                logger.info("channel {:d}: electric field is refracted into the firn. theta {:.0f} -> {:.0f}. Transmission coefficient p (eTheta) {:.2f} s (ePhi) {:.2f}".format(iCh, zenith / units.deg, zenith_antenna / units.deg, t_theta, t_phi))
        else:
            # now the signal is coming from below, do we have an antenna above the surface?
            position = detector.get_relative_position(station.get_id(), channel_id)
            if(position[2] > 0):
                zenith_antenna = geo_utl.get_fresnel_angle(zenith, 1., n_ice)
        if(zenith_antenna is None):
            logger.warning("fresnel reflection at air-firn boundary leads to unphysical results, no reconstruction possible")
            return None

        logger.debug("angles: zenith {0:.0f}, zenith antenna {1:.0f}, azimuth {2:.0f}".format(np.rad2deg(zenith), np.rad2deg(zenith_antenna), np.rad2deg(azimuth)))
        antenna_model = detector.get_antenna_model(station.get_id(), channel_id, zenith_antenna)
        antenna_pattern = antenna_pattern_provider.load_antenna_pattern(antenna_model)
        ori = detector.get_antenna_orientation(station.get_id(), channel_id)
        VEL = antenna_pattern.get_antenna_response_vectorized(frequencies, zenith_antenna, azimuth, *ori)
        efield_antenna_factor[iCh] = np.array([VEL['theta'] * t_theta, VEL['phi'] * t_phi])
    return efield_antenna_factor


def get_channel_voltage_from_efield(station, electric_field, channels, detector, zenith, azimuth, antenna_pattern_provider, return_spectrum=True):
    """
    Returns the voltage traces that would result in the channels from the station's E-field.

    Parameters
    ----------

    station: Station
    electric_field: ElectricField
    channels: array of int
        IDs of the channels for which the expected voltages should be calculated
    detector: Detector
    zenith, azimuth: float
        incoming direction of the signal. Note that reflection and refraction
        at the air/ice boundary are already being taken into account.
    antenna_pattern_provider: AntennaPatternProvider
    return_spectrum: boolean
        if True, returns the spectrum, if False return the time trace
    """

    frequencies = electric_field.get_frequencies()
    spectrum = electric_field.get_frequency_spectrum()
    efield_antenna_factor = get_efield_antenna_factor(station, frequencies, channels, detector, zenith, azimuth, antenna_pattern_provider)
    if return_spectrum:
        voltage_spectrum = np.zeros((len(channels), len(frequencies)), dtype=complex)
        for i_ch, ch in enumerate(channels):
            voltage_spectrum[i_ch] = np.sum(efield_antenna_factor[i_ch] * np.array([spectrum[1], spectrum[2]]), axis=0)
        return voltage_spectrum
    else:
        voltage_trace = np.zeros((len(channels), 2 * (len(frequencies) - 1)), dtype=complex)
        for i_ch, ch in enumerate(channels):
            voltage_trace[i_ch] = fft.freq2time(np.sum(efield_antenna_factor[i_ch] * np.array([spectrum[1], spectrum[2]]), axis=0), electric_field.get_sampling_rate())
        return np.real(voltage_trace)


def get_electric_field_energy_fluence(electric_field_trace, times, signal_window_mask=None, noise_window_mask=None):

    if signal_window_mask is None:
        f_signal = np.sum(electric_field_trace ** 2, axis=1)
    else:
        f_signal = np.sum(electric_field_trace[:, signal_window_mask] ** 2, axis=1)
    dt = times[1] - times[0]
    if noise_window_mask is not None:
        f_noise = np.sum(electric_field_trace[:, noise_window_mask] ** 2, axis=1)
        f_signal -= f_noise * np.sum(signal_window_mask) / np.sum(noise_window_mask)

    return f_signal * dt * conversion_factor_integrated_signal

def get_stokes(trace_u, trace_v, window_samples=128, squeeze=True):
    """
    Compute the stokes parameters for electric field traces

    Parameters
    ----------
    trace_u : 1d array (float)
        The u component of the electric field trace
    trace_v : 1d array (float)
        The v component of the electric field trace.
        The two components should have equal lengths,
        and the (u, v) coordinates should be perpendicular.
        Common choices are (theta, phi) or (vxB, vxvxB)
    window_samples : int | None, default: 128
        If not None, return a running average
        of the stokes parameters over ``window_samples``.
        If None, compute the stokes parameters over the
        whole trace (equivalent to ``window_samples=len(trace_u)``).
    squeeze : bool, default: True
        Only relevant if ``window_samples=None``. Squeezes
        out the second axis (which has a length of one)
        and returns an array of shape (4,)

    Returns
    -------
    stokes : 2d array of floats
        The stokes parameters I, Q, U, V. The shape of
        the returned array is ``(4, len(trace_u) - window_samples +1)``,
        i.e. stokes[0] returns the I parameter,
        stokes[1] corresponds to Q, and so on.

    Examples
    --------
    For an electric field defined in (eR, eTheta, ePhi) components,
    the stokes parameters can be given simply by:

    .. code-block::

        get_stokes(electric_field.get_trace()[1], electric_field.get_trace()[2])

    To instead get the stokes parameters in vxB and vxvxB, we need to first obtain
    the appropriate electric field components

    .. code-block::

        cs = radiotools.coordinatesystems.cstrafo(zenith, azimuth, magnetic_field_vector)

        efield_trace_vxB_vxvxB = cs.transform_to_vxB_vxvxB(
            cs.transform_from_onsky_to_ground(efield.get_trace())
        )

    """

    assert len(trace_u) == len(trace_v)
    h1 = scipy.signal.hilbert(trace_u)
    h2 = scipy.signal.hilbert(trace_v)
    stokes_i = np.abs(h1)**2 + np.abs(h2)**2
    stokes_q = np.abs(h1)**2 - np.abs(h2)**2
    uv = 2 * h1 * np.conjugate(h2)
    stokes_u = np.real(uv)
    stokes_v = np.imag(uv)
    stokes = np.array([stokes_i, stokes_q, stokes_u, stokes_v])
    if window_samples == 1: # no need to average
        return stokes
    elif window_samples is None:
        window_samples = len(h1)

    stokes = np.asarray([
        scipy.signal.convolve(i, np.ones(window_samples), mode='valid') for i in stokes
    ])
    stokes /= window_samples

    if squeeze:
        return np.squeeze(stokes)
    return stokes

def upsampling_fir(trace, original_sampling_frequency, int_factor=2, ntaps=2 ** 7):
    """
    This function performs an upsampling by inserting a number of zeroes
    between samples and then applying a finite impulse response (FIR) filter.

    Parameters
    ----------

    trace: array of floats
        Trace to be upsampled
    original_sampling_frequency: float
        Sampling frequency of the input trace
    int_factor: integer
        Upsampling factor. The resulting trace will have a sampling frequency
        int_factor times higher than the original one
    ntaps: integer
        Number of taps (order) of the FIR filter

    Returns
    -------
    upsampled_trace: array of floats
        The upsampled trace
    """

    if (np.abs(int(int_factor) - int_factor) > 1e-3):
        warning_msg = "The input upsampling factor does not seem to be close to an integer."
        warning_msg += "It has been rounded to {}".format(int(int_factor))
        logger.warning(warning_msg)

    int_factor = int(int_factor)

    if (int_factor <= 1):
        error_msg = "Upsampling factor is less or equal to 1. Upsampling will not be performed."
        raise ValueError(error_msg)

    zeroed_trace = np.zeros(len(trace) * int_factor)
    for i_point, point in enumerate(trace[:-1]):
        zeroed_trace[i_point * int_factor] = point

    upsampled_delta_time = 1 / (int_factor * original_sampling_frequency)
    upsampled_times = np.arange(0, len(zeroed_trace) * upsampled_delta_time, upsampled_delta_time)

    cutoff = 1. / int_factor
    fir_coeffs = scipy.signal.firwin(ntaps, cutoff, window='boxcar')
    upsampled_trace = np.convolve(zeroed_trace, fir_coeffs)[:len(upsampled_times)] * int_factor

    return upsampled_trace


def butterworth_filter_trace(trace, sampling_frequency, passband, order=8):
    """
    Filters a trace using a Butterworth filter.

    Parameters
    ----------

    trace: array of floats
        Trace to be filtered
    sampling_frequency: float
        Sampling frequency
    passband: (float, float) tuple
        Tuple indicating the cutoff frequencies
    order: integer
        Filter order

    Returns
    -------

    filtered_trace: array of floats
        The filtered trace
    """

    n_samples = len(trace)

    spectrum = fft.time2freq(trace, sampling_frequency)
    frequencies = np.fft.rfftfreq(n_samples, 1 / sampling_frequency)

    filtered_spectrum = apply_butterworth(spectrum, frequencies, passband, order)
    filtered_trace = fft.freq2time(filtered_spectrum, sampling_frequency)

    return filtered_trace


def apply_butterworth(spectrum, frequencies, passband, order=8):
    """
    Calculates the response from a Butterworth filter and applies it to the
    input spectrum

    Parameters
    ----------
    spectrum: array of complex
        Fourier spectrum to be filtere
    frequencies: array of floats
        Frequencies of the input spectrum
    passband: (float, float) tuple
        Tuple indicating the cutoff frequencies
    order: integer
        Filter order

    Returns
    -------
    filtered_spectrum: array of complex
        The filtered spectrum
    """

    f = np.zeros_like(frequencies, dtype=complex)
    mask = frequencies > 0
    b, a = scipy.signal.butter(order, passband, 'bandpass', analog=True)
    w, h = scipy.signal.freqs(b, a, frequencies[mask])
    f[mask] = h

    filtered_spectrum = f * spectrum

    return filtered_spectrum


def delay_trace(trace, sampling_frequency, time_delay, crop_trace=True):
    """
    Delays a trace by transforming it to frequency and multiplying by phases.

    A positive delay means that the trace is shifted to the right, i.e., its delayed.
    A negative delay would mean that the trace is shifted to the left. Since this
    method is cyclic, the delayed trace will have unphysical samples at either the
    beginning (delayed, positive `time_delay`) or at the end (negative `time_delay`).
    Those samples can be cropped (optional, default=True).

    Parameters
    ----------
    trace: array of floats or `NuRadioReco.framework.base_trace.BaseTrace`
        Array containing the trace
    sampling_frequency: float
        Sampling rate for the trace
    time_delay: float
        Time delay used for transforming the trace. Must be positive or 0
    crop_trace: bool (default: True)
        If True, the trace is cropped to remove samples what are unphysical
        after delaying (rolling) the trace.

    Returns
    -------
    delayed_trace: array of floats
        The delayed, cropped trace
    dt_start: float (optional)
        The delta t of the trace start time. Only returned if crop_trace is True.
    """
    # Do nothing if time_delay is 0
    if not time_delay:
        if isinstance(trace, NuRadioReco.framework.base_trace.BaseTrace):
            if crop_trace:
                return trace.get_trace(), 0
            else:
                return trace.get_trace()
        else:
            if crop_trace:
                return trace, 0
            else:
                return trace

    if isinstance(trace, NuRadioReco.framework.base_trace.BaseTrace):
        spectrum = trace.get_frequency_spectrum()
        frequencies = trace.get_frequencies()
        if trace.get_sampling_rate() != sampling_frequency:
            raise ValueError("The sampling frequency of the trace does not match the given sampling frequency.")
    else:
        n_samples = len(trace)
        spectrum = fft.time2freq(trace, sampling_frequency)
        frequencies = np.fft.rfftfreq(n_samples, 1 / sampling_frequency)

    spectrum *= np.exp(-1j * 2 * np.pi * frequencies * time_delay)

    delayed_trace = fft.freq2time(spectrum, sampling_frequency)
    cycled_samples = int(round(time_delay * sampling_frequency))

    if crop_trace:
        # according to a NuRadio convention, traces should have an even number of samples.
        # Make sure that after cropping the trace has an even number of samples (assuming that it was even before).
        if cycled_samples % 2 != 0:
            cycled_samples += 1

        if time_delay >= 0:
            delayed_trace = delayed_trace[cycled_samples:]
            dt_start = cycled_samples * sampling_frequency
        else:
            delayed_trace = delayed_trace[:-cycled_samples]
            dt_start = 0

        return delayed_trace, dt_start

    else:
        # Check if unphysical samples contain any signal and if so, throw a warning
        if time_delay > 0:
            if np.any(np.abs(delayed_trace[:cycled_samples]) > 0.01 * units.microvolt):
                logger.warning("The delayed trace has unphysical samples that contain signal. "
                    "Consider cropping the trace to remove these samples.")
        else:
            if np.any(np.abs(delayed_trace[-cycled_samples:]) > 0.01 * units.microvolt):
                logger.warning("The delayed trace has unphysical samples that contain signal. "
                    "Consider cropping the trace to remove these samples.")

        return delayed_trace


def maximum_peak_to_peak_amplitude(trace, coincidence_window_size):
    """
    Calculates the maximal peak to peak amplitude of a given trace.

    Parameters
    ----------
    trace: array of floats
        Array containing the trace
    coincidence_window_size: int
        Length along which to calculate minimum

    Returns
    -------
    maximal peak to peak amplitude of the trace
    """
    return scipy.ndimage.maximum_filter1d(trace, coincidence_window_size) - scipy.ndimage.minimum_filter1d(trace, coincidence_window_size)

def split_trace_noise_rms(trace, segments=4, lowest=2):
    """
    Calculates the noise rms of a given trace by splitting the trace into segments, calculating the rms of each trace and subsequently taking the mean of the lowest few segemt rms values.

    Parameters
    ----------
    trace: array of floats
        Array containing the trace
    segments: int
        Amount of segments to cut the trace int
    lowest: int
        Amount of lowest segment rms values to use when calculating the mean rms end result

    Returns
    -------
    rms: float
        The mean rms of the lowest few segment rms values
    """
    split_array = np.array_split(trace, segments)
    split_array = np.array(split_array, dtype="object") #Objectify dtype to allow timetraces indivisible by amount of segments
    rms_of_splits = [np.std(split) for split in split_array]
    ordered_rmss = np.sort(rms_of_splits)
    lowest_rmss = ordered_rmss[:lowest]
    rms = np.mean(lowest_rmss)
    return rms

def get_signal_to_noise_ratio(trace, noise_rms, window_size=None):
    """
    Computes the Signal to Noise Ratio (SNR) of a given trace.

    Parameters:
    ----------
    trace : waveform of given channel
        the 1d array array containing trace of a channel
    noise_rms: float
        noise root mean square.
    window_size: int
        coincidence window size.
    Returns:
    --------
    root_power_ratio: float
        Root Power Ratio value.
    """
    if window_size:
        p2p = np.amax(maximum_peak_to_peak_amplitude(trace, window_size))
    else:
        upper_peak_idx = argrelextrema(trace, np.greater_equal, order = 1)[0]
        lower_peak_idx = argrelextrema(trace, np.less_equal, order = 1)[0]
        peak_idx = np.unique(np.concatenate((upper_peak_idx, lower_peak_idx)))
        peak = trace[peak_idx]
        p2p = np.abs(np.diff(peak))
        p2p = np.nanmax(p2p)

    snr = p2p / (2 * noise_rms)

    return snr

def get_root_power_ratio(trace, times, noise_rms):
    """
    Computes the Root Power Ratio (RPR) of a given trace.

    Parameters:
    ----------
    trace : waveform of given channel
        the 1d array array containing trace of a channel
    times:
        time array of a channel
    noise_rms: float
        noise root mean square.
    Returns:
    --------
    root_power_ratio: float
        Root Power Ratio value.
    """

    # Calculating RPR (Root Power Ratio)
    if noise_rms == 0:
        root_power_ratio = np.inf
    else:
        wf_len = len(trace)
        channel_wf = trace ** 2

        # Calculate the smoothing window size based on sampling rate
        dt = times[1] - times[0]
        sum_win = 25  # Smoothing window in ns
        sum_win_idx = int(np.round(sum_win / dt))  # Convert window size to sample points

        channel_wf = np.sqrt(uniform_filter1d(channel_wf, size=sum_win_idx, mode='constant'))

        # Find the maximum value of the smoothed waveform
        max_bin = np.argmax(channel_wf)
        max_val = channel_wf[max_bin]

        root_power_ratio = max_val / noise_rms

    return root_power_ratio

def get_hilbert_envelope(trace):
    """
    Calculates the Hilbert envelope of given waveform.


    Parameters
    ----------
    trace : waveform trace of given channel
        the 1d array array containing trace of a channel
    times:
        time array of a channel
    Returns
    -------
    float
        Hilbert envelope of the waveform.
    """

    # Get the Hilbert envelope of the waveform trace
    envelope = np.abs(hilbert(trace))

    return envelope

def get_impulsivity(trace):
    """
    Calculate the impulsivity of a signal.

    This function computes the impulsivity of a signal by performing a Hilbert transform
    to obtain the analytic signal, calculating the envelope, and then determining the
    cumulative distribution function (CDF) of the square of sorted envelope values (power) based on their
    closeness to the maximum value. The average of the CDF is then scaled and returned
    as the impulsivity measure.

    Parameters
    ----------
    trace : waveform of given channel
        the 1d array array containing trace of a channel

    Returns
    -------
    float
        The impulsivity measure of the signal, scaled between 0 and 1.
    """

    envelope = get_hilbert_envelope(trace)
    maxv = np.argmax(envelope)
    envelope_indexes = np.arange(len(envelope)) ## just a list of indices the same length as the array
    closeness = list(
        np.abs(envelope_indexes - maxv)
    )  ## create an array containing index distance to max voltage (lower the value, the closer it is)

    sorted_envelope = [x for _, x in sorted(zip(closeness, envelope))]
    cdf = np.cumsum(sorted_envelope**2) ## taken reference from ARA : https://github.com/ara-software/AraProc/blob/f7e28c03a6a0603d2ac580ab353bf70940ee97f9/araproc/analysis/impulsivity.py#L64

    cdf = cdf / cdf[-1]

    cdf_avg = (np.mean(np.asarray([cdf])) * 2.0) - 1.0

    if cdf_avg < 0:
        cdf_avg = 0.0
    return cdf_avg

def get_coherent_sum(trace_set, ref_trace, use_envelope = False):
    sum_wf = ref_trace
    for idx, trace in enumerate(trace_set):
        if use_envelope:
            sig_ref = trace_utilities.get_hilbert_envelope(ref_trace)
            sig_i = trace_utilities.get_hilbert_envelope(trace)
        else:
            sig_ref = ref_trace
            sig_i = trace
        cor = signal.correlate(sig_ref, sig_i, mode = "full")
        lag = int(np.argmax((cor)) - (np.size(cor)/2.))

        aligned_wf = np.roll(trace, lag)
        sum_wf += aligned_wf
    return sum_wf

def get_entropy(trace, n_hist_bins = 50):
    """
    Calculate the shannon entropy of a signal.


    Parameters
    ----------
    trace : waveform of given channel
        the 1d array array containing trace of a channel
    Returns
    -------
    float
        The entropy (randomness) measure of the signal.
    """

    # Step 1: Discretize the signal into bins
    # If density = True, the result is the value of the probability density function at the bin,
    # normalized such that the integral over the range is 1.
    hist, bin_edges = np.histogram(trace, bins = n_hist_bins, density = True)

    # Step 2: Calculate the probability distribution (normalized)
    probabilities = hist / np.sum(hist)

    # Step 3: Calculate Shannon Entropy
    # Using base = 2 for entropy in bits
    signal_entropy = entropy(probabilities, base = 2)

    return signal_entropy

def get_kurtosis(trace):
    """
    Calculate the kurtosis (tailedness) of a signal.


    Parameters
    ----------
    trace : waveform of given channel
        the 1d array array containing trace of a channel
    Returns
    -------
    float
        The kurtosis (tailedness) measure of the signal.
    """

    signal_kurtosis = kurtosis(trace)

    return signal_kurtosis

def is_trace_corrupt(trace):
    """
    Makes sure the trace has no NAN nor INF"


    Parameters
    ----------
    trace : waveform trace of given channel
        the 1d array array containing trace of a channel
    Returns
    -------
    bool
        True or False based on NAN or INF check
    """

    is_bad_waveform = False

    trace = np.array(trace)

    npoints_NAN = len( np.argwhere(np.isnan(trace)) )
    npoints_INF = len( np.argwhere(np.isinf(trace)) )

    if npoints_NAN or npoints_INF:
       is_bad_waveform = True

    return is_bad_waveform
