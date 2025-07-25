"""
This module contains utility functions to compute various observables from waveforms.

The functions in this module can be used to compute observables from waveforms, such as the energy fluence,
the stokes parameters, the signal-to-noise ratio, the root power ratio, the Hilbert envelope, the impulsivity,
the coherent sum, the entropy, the kurtosis, and the correlation between two traces.

All functions in this module do not depend on the NuRadioReco framework
and can be used independently. The functions do not alter the input traces,
but only compute observables from them.

See Also
--------
`NuRadioReco.utilities.signal_processing`
    Module for functions that modify traces, e.g., by filtering, delaying, etc.
"""

from NuRadioReco.utilities import units, signal_processing, fft

import numpy as np
import scipy.stats
import scipy.signal
import scipy.ndimage
import scipy.constants
import warnings

import logging
logger = logging.getLogger('NuRadioReco.trace_utilities')

conversion_factor_integrated_signal = scipy.constants.c * scipy.constants.epsilon_0 * units.joule / units.s / units.volt ** 2

# see Phys. Rev. D DOI: 10.1103/PhysRevD.93.122005
# to convert V**2/m**2 * s -> J/m**2 -> eV/m**2


def get_efield_antenna_factor(*args, **kwargs):
    warnings.warn("get_efield_antenna_factor is moved to NuRadioReco.utilities.signal_processing.get_efield_antenna_factor", DeprecationWarning)
    return signal_processing.get_efield_antenna_factor(*args, **kwargs)


def get_channel_voltage_from_efield(*args, **kwargs):
    warnings.warn("get_channel_voltage_from_efield is moved to NuRadioReco.utilities.signal_processing.get_channel_voltage_from_efield", DeprecationWarning)
    return signal_processing.get_channel_voltage_from_efield(*args, **kwargs)

def upsampling_fir(*args, **kwargs):
    warnings.warn("upsampling_fir is moved to NuRadioReco.utilities.signal_processing.upsampling_fir", DeprecationWarning)
    return signal_processing.upsampling_fir(*args, **kwargs)


def butterworth_filter_trace(*args, **kwargs):
    warnings.warn("butterworth_filter_trace is moved to NuRadioReco.utilities.signal_processing.butterworth_filter_trace", DeprecationWarning)
    return signal_processing.butterworth_filter_trace(*args, **kwargs)


def apply_butterworth(*args, **kwargs):
    warnings.warn("apply_butterworth is moved to NuRadioReco.utilities.signal_processing.apply_butterworth", DeprecationWarning)
    return signal_processing.apply_butterworth(*args, **kwargs)


def delay_trace(*args, **kwargs):
    warnings.warn("delay_trace is moved to NuRadioReco.utilities.signal_processing.delay_trace", DeprecationWarning)
    return signal_processing.delay_trace(*args, **kwargs)


def get_electric_field_energy_fluence(electric_field_trace, times, signal_window_mask=None, noise_window_mask=None, return_uncertainty=False, method="noise_subtraction", estimator_kwargs={}):
    """
    Returns the energy fluence of each component of a 3-dimensional electric field trace.

    Parameters
    ----------
    electric_field_trace : numpy.ndarray
        The electric field trace to calculate the energy fluence for
    times : numpy.ndarray
        The time grid for the electric field trace
    signal_window_mask : numpy.ndarray (optional)
        A boolean mask that selects the signal window in which the energy fluence is calculated
    noise_window_mask : numpy.ndarray (optional)
        A boolean mask that selects the noise window. Only used if method is "noise_subtraction"
    return_uncertainty : bool (optional)
        If True, the uncertainty of the energy fluence is returned
    method : str (optional)
        The method to use for the energy fluence calculation. Can be either "noise_subtraction" or "rice_distribution".
        The Rice distribution method implementation is based on the code published alongside S. Martinelli et al.: https://arxiv.org/pdf/2407.18654
    estimator_kwargs : dict (optional)
        Additional keyword arguments for the _get_noise_fluence_estimators and _get_signal_fluence_estimators functions.
        Only used if method is "rice_distribution".

    Returns
    -------
    signal_energy_fluence : numpy.ndarray
        The energy fluence of each component of the electric field trace
    signal_energy_fluence_error : numpy.ndarray
        The uncertainty of the energy fluences. Only returned if return_uncertainty is True
    """

    dt = times[1] - times[0]

    if method == "noise_subtraction":
        if signal_window_mask is None:
            f_signal = np.sum(electric_field_trace ** 2, axis=1)
        else:
            f_signal = np.sum(electric_field_trace[:, signal_window_mask] ** 2, axis=1)

        if noise_window_mask is not None and np.sum(noise_window_mask) > 0:
            f_noise = np.sum(electric_field_trace[:, noise_window_mask] ** 2, axis=1)
            f_signal -= f_noise * np.sum(signal_window_mask) / np.sum(noise_window_mask)
            f_signal[f_signal < 0] = 0

            # calculate RMS noise for error estimation
            RMSNoise = np.sqrt(np.mean(electric_field_trace[:, noise_window_mask] ** 2, axis=1))
        else:
            RMSNoise = None

        signal_energy_fluence = f_signal * dt * conversion_factor_integrated_signal

        # calculate error if RMSNoise is known:
        if RMSNoise is not None and return_uncertainty:
            signal_window_duration = sum(signal_window_mask) * dt if signal_window_mask is not None else len(times) * dt
            signal_energy_fluence_error = (4 * np.abs(signal_energy_fluence / conversion_factor_integrated_signal) * RMSNoise ** 2 * dt + 2 * signal_window_duration * RMSNoise ** 4 * dt) ** 0.5  * conversion_factor_integrated_signal
        else:
            signal_energy_fluence_error = np.zeros(3)
    
    elif method == "rice_disttribution":
        signal_energy_fluence = np.zeros(len(electric_field_trace))
        signal_energy_fluence_error = np.zeros(len(electric_field_trace))
        for i_pol in range(len(electric_field_trace)):
            noise_estimators, frequencies_window = _get_noise_fluence_estimators(
                trace = electric_field_trace[i_pol, :],
                times = times,
                signal_window_mask = signal_window_mask,
                **estimator_kwargs
                )
            estimators, variances = _get_signal_fluence_estimators(
                trace = electric_field_trace[i_pol, :],
                times = times,
                signal_window_mask = signal_window_mask,
                noise_estimators = noise_estimators,
                **estimator_kwargs
                )

            #sample frequency (after the windowing) in MHz
            delta_f = frequencies_window[1] - frequencies_window[0]

            #get the fluence of the trace summing up the frequency estimators and converting in eV/m^2
            fluence_freq = np.sum(estimators) * delta_f * conversion_factor_integrated_signal

            if estimator_kwargs.get("truncate_negative_estimators") == "after_sum":
                fluence_freq = np.maximum(fluence_freq, 0)

            #get the variance of the trace fluence summing up the frequency variances and converting in (eV/m^2)^2
            fluence_freq_variance = np.sum(variances) * (delta_f * conversion_factor_integrated_signal)**2

            #get the fluence uncertainty as the root square of the variance
            fluence_freq_error = np.sqrt(fluence_freq_variance)

            signal_energy_fluence[i_pol] = fluence_freq
            signal_energy_fluence_error[i_pol] = fluence_freq_error

    if return_uncertainty:
        return signal_energy_fluence, signal_energy_fluence_error
    else:
        return signal_energy_fluence

      
def _get_noise_fluence_estimators(trace, times, signal_window_mask, spacing_noise_signal=20*units.ns, relative_taper_width=0.142857143, use_median_value=False, truncate_negative_estimators="before_sum"):
    """
    Estimate the noise fluence from the trace.

    Parameters
    ----------
    trace : np.ndarray
        Trace to estimate the noise fluence from.
    times : np.ndarray
        Time grid for the trace.
    signal_window_mask : np.ndarray
        Boolean mask for the signal window.
    spacing_noise_signal : float (optional)
        Spacing between noise windows and signal window. Makes sure no signal leaks into the noise windows. (default: 20 ns,
        which should be enough for most applications, otherwise the signal_window_mask is too small)
    relative_taper_width : float (optional)
        Width of the taper region for the Tukey window relative to the full window length. (default: 0.142857143,
        which corresponds to 1/7 of the window length at each end)
    use_median_value : bool (optional)
        If True, the median of the squared spectra of the noise windows is used as estimator. Otherwise, the mean is used. (default: False)
    truncate_negative_estimators : str (optional)
        Not used in this function. Introduced for compatibility with _get_signal_fluence_estimators.

    Returns
    -------
    estimators : np.ndarray
        Estimators for the noise fluence.
    frequencies_window : np.ndarray
        Frequencies corresponding to the estimators.
    """

    dt = times[1] - times[0]
    n_samples_window = sum(signal_window_mask)
    signal_start = times[signal_window_mask][0] - spacing_noise_signal
    signal_stop = times[signal_window_mask][-1] + spacing_noise_signal
    list_ffts_squared = []

    if signal_start < times[0]:
        signal_start = times[0]
        logger.warning("The signal window overlaps with the start of the trace. The efield pulse may be partialy outside the trace.")
    elif signal_stop > times[-1]:
        logger.warning("The signal window overlaps with the end of the trace. The efield pulse may be partialy outside the trace.")

    #generate Tukey window
    window = scipy.signal.windows.tukey(n_samples_window, relative_taper_width * 2)

    #loop over the trace defining noise windows (excluding the signal window)
    noise_start = times[0]
    while noise_start < times[-1]:

        noise_stop = noise_start + n_samples_window * dt
        if noise_stop > times[-1]:
            break

        elif (noise_stop <= signal_start and noise_start < signal_start) or (noise_stop > signal_stop and noise_start >= signal_stop):

            #clipping the noise window (rounding is needed because noise_stop = noise_start + n_samples_window * dt has numerical uncertainties)
            mask_time = np.all([np.round(times, 5) >= np.round(noise_start, 5), np.round(times, 5) < np.round(noise_stop, 5)], axis=0)
            time_trace_clipped = trace[mask_time]

            #applying the Tukey window
            windowed_trace = time_trace_clipped * window

            #calculating the spectrum and frequencies
            frequencies_window = fft.freqs(n_samples_window, 1/dt)
            spectrum_window = np.abs(fft.time2freq(windowed_trace, 1/dt))

            list_ffts_squared.append(spectrum_window**2)
            noise_start = noise_stop

        elif noise_stop > signal_start and noise_start <= signal_start:
            noise_start = signal_stop

        else:
            logger.error("The noise window does not fulfill any of the conditions. This should not happen.")
            raise RuntimeError("The noise window does not fulfill any of the conditions. This should not happen.")

    list_ffts_squared = np.array(list_ffts_squared, dtype=float)

    if use_median_value:
        #robust estimator in presence of outliers from the noise windows
        estimators = np.median(list_ffts_squared, axis=0) / 1.405 #from chi2 distribution
    else:
        #it works well in presence of small number of outliers
        estimators=np.mean(list_ffts_squared, axis=0)

    return estimators, frequencies_window


def _get_signal_fluence_estimators(trace, times, signal_window_mask, noise_estimators, spacing_noise_signal=20*units.ns, relative_taper_width=0.142857143, use_median_value=False, truncate_negative_estimators="before_sum"):
    """
    Estimate the signal fluence from the trace.

    Parameters
    ----------
    trace : np.ndarray
        Trace to estimate the signal fluence from.
    times : np.ndarray
        Time grid for the trace.
    signal_window_mask : np.ndarray
        Boolean mask for the signal window.
    noise_estimators : np.ndarray
        Estimators for the noise fluence.
    spacing_noise_signal : float (optional)
        Not used in this function. Indroduced for compatibility with _get_noise_fluence_estimators.
    relative_taper_width : float (optional)
        Width of the taper region for the Tukey window relative to the full window length. (default: 0.142857143,
        which corresponds to 1/7 of the window length at each end)
    use_median_value : bool (optional)
        Not used in this function. Indroduced for compatibility with _get_noise_fluence_estimators.
    truncate_negative_estimators : str (optional)
        If "before_sum", negative estimators are set to zero before summing over frequencies,
        which is consistent with S. Martinelli et al.: https://arxiv.org/pdf/2407.18654.
        If "after_sum", negative estimators are set to zero after summing over frequencies. Note
        that this is done in get_electric_field_energy_fluence. (default: "before_sum")

    Returns
    -------
    signal_estimators : np.ndarray
        Estimators for the signal fluence.
    variances : np.ndarray
        Variance of the signal fluence estimators.
    """

    dt = times[1] - times[0]
    n_samples_window = sum(signal_window_mask)
    signal_start = times[signal_window_mask][0]
    signal_stop = times[signal_window_mask][-1] + dt

    #generate Tukey window
    window = scipy.signal.windows.tukey(n_samples_window, relative_taper_width * 2)

    #clipping the signal window around the pulse position
    mask_time = np.all([times >= signal_start, times < signal_stop], axis=0)
    trace_clipped = trace[mask_time]

    #applying the Tukey window
    windowed_trace = trace_clipped * window

    #calculating the spectrum and frequencies
    spectrum_window = np.abs(fft.time2freq(windowed_trace, 1/dt))

    #signal estimator and variance for each frequency bin
    signal_estimators = spectrum_window**2 - noise_estimators
    if truncate_negative_estimators == "before_sum":
        signal_estimators[signal_estimators < 0] = 0
    variances = noise_estimators * (noise_estimators + 2*signal_estimators)

    return signal_estimators, variances


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


def peak_to_peak_amplitudes(trace, coincidence_window_size):
    """
    Calculates all local peak to peak amplitudes of a given trace.

    Parameters
    ----------
    trace: array of floats
        Array containing the trace
    coincidence_window_size: int
        Length along which to calculate minimum

    Returns
    -------
    amplitudes: array of floats (same length as the input trace)
        Local peak to peak amplitudes
    """
    amplitudes = scipy.ndimage.maximum_filter1d(trace, coincidence_window_size) - scipy.ndimage.minimum_filter1d(trace, coincidence_window_size)
    return amplitudes


def get_split_trace_noise_RMS(trace, segments=4, lowest=2):
    """
    Calculates the noise root mean square (RMS) of a given trace.

    This method splits the trace into segments,
    then calculates the RMS of each segment,
    and then takes the mean of the lowest few segemts' RMS values.

    Parameters
    ----------
    trace: array of floats
        Array containing the trace
    segments: int
        Amount of segments to cut the trace int
    lowest: int
        Amount of lowest segment rms values to use when calculating the mean RMS end result

    Returns
    -------
    noise_root_mean_square: float
        The mean of the lowest few segments' RMS values
    """
    split_array = np.array_split(trace, segments)
    split_array = np.array(split_array, dtype="object") #Objectify dtype to allow timetraces indivisible by amount of segments
    rms_of_splits = [np.std(split) for split in split_array]
    ordered_rmss = np.sort(rms_of_splits)
    lowest_rmss = ordered_rmss[:lowest]
    noise_root_mean_square = np.mean(lowest_rmss)

    return noise_root_mean_square


def get_signal_to_noise_ratio(trace, noise_rms, window_size=3):
    """
    Computes the Signal to Noise Ratio (SNR) of a given trace.

    The signal to noise ratio is calculated as the peak to peak amplitude
    within a given window size divided by twice the noise root mean square (RMS).

    Parameters
    ----------
    trace : array of floats
        Trace of a waveform
    noise_rms: float
        Noise root mean square (RMS)
    window_size: int
        Coincidence window size (default: 3)

    Returns
    -------
    signal_to_noise_ratio: float
        Signal to Noise Ratio (SNR) value
    """
    if not window_size >= 2:
        logger.error(f"Window size must be greater-equal 2 (but is {window_size})")
        raise ValueError(f"Window size must be greater-equal 2 (but is {window_size})")

    p2p = np.amax(peak_to_peak_amplitudes(trace, window_size))
    signal_to_noise_ratio = p2p / (2 * noise_rms)

    return signal_to_noise_ratio


def get_root_power_ratio(trace, times, noise_rms):
    """
    Computes the Root Power Ratio (RPR) of a given trace.

    It compares the peak signal strength to the baseline noise.
    The waveform’s power is smoothed using a 25 ns sliding window,
    and the square root of this smoothed power gives the rolling root power.
    The RPR is the maximum root power divided by the noise RMS

    Parameters
    ----------
    trace: array of floats
        Trace of a waveform
    times: array of floats
        Times of a waveform
    noise_rms: float
        noise root mean square (RMS)

    Returns
    -------
    root_power_ratio: float
        Root Power Ratio (RPR) value
    """

    # Calculating RPR (Root Power Ratio)
    if noise_rms == 0:
        root_power_ratio = np.inf
    else:
        channel_wf = trace ** 2

        # Calculate the smoothing window size based on sampling rate
        dt = times[1] - times[0]
        sum_win = 25  # Smoothing window in ns
        sum_win_idx = int(np.round(sum_win / dt))  # Convert window size to sample points

        channel_wf = np.sqrt(scipy.ndimage.uniform_filter1d(channel_wf, size=sum_win_idx, mode='constant'))

        # Find the maximum value of the smoothed waveform
        max_bin = np.argmax(channel_wf)
        max_val = channel_wf[max_bin]

        root_power_ratio = max_val / noise_rms

    return root_power_ratio


def get_hilbert_envelope(trace):
    """
    Applies the Hilbert Tranform to a given waveform trace,
    then it will give us an envelope trace.

    Parameters
    ----------
    trace: array of floats
        Trace of a waveform

    Returns
    -------
    envelope: array of floats
        Hilbert envelope of the waveform trace
    """
    # Get the Hilbert envelope of the waveform trace
    envelope = np.abs(scipy.signal.hilbert(trace))
    return envelope


def get_impulsivity(trace):
    """
    Calculates the impulsivity of a signal (trace).

    This function computes the impulsivity of a trace by first performing the Hilbert Transform
    to obtain an analytic signal (Hilbert envelope) and then determining the
    cumulative distribution function (CDF) of the square of the sorted envelope values i.e. power values based on their
    closeness to the maximum value. The average of the CDF is then scaled and returned
    as the impulsivity value.

    Parameters
    ----------
    trace: array of floats
        Trace of a waveform

    Returns
    -------
    impulsivity: float
        Impulsivity of the signal (scaled between 0 and 1)
    """

    envelope = get_hilbert_envelope(trace)
    maxv = np.argmax(envelope)
    envelope_indexes = np.arange(len(envelope)) ## just a list of indices the same length as the array
    closeness = list(
        np.abs(envelope_indexes - maxv)
    )  ## create an array containing index distance to max voltage (lower the value, the closer it is)

    sorted_envelope = np.array([x for _, x in sorted(zip(closeness, envelope))])
    cdf = np.cumsum(sorted_envelope**2)
    cdf = cdf / cdf[-1]

    impulsivity = (np.mean(np.asarray([cdf])) * 2.0) - 1.0
    if impulsivity < 0:
        impulsivity = 0.0

    return impulsivity


def get_coherent_sum(trace_set, ref_trace, use_envelope = False):
    """
    Generates the coherently-summed waveform (CSW) of a sets of traces.

    This function finds the correlation between each trace from a set and the reference trace,
    then rolls all traces to align with the reference trace,
    and then add them up to get the CSW.

    Parameters
    ----------
    trace_set: 2-D array of floats
        Traces of multiple channel waveforms without the reference trace
    ref_trace: 1-D array of floats
        Trace of the reference channel
    use_envelope: bool
        See if users would like to find the correlation between envelopes or just normal traces (default: False)

    Returns
    -------
    sum_trace: 1-D array of floats
        CSW of the set of traces
    """
    # Normalize: subtract mean, divide by std (z-score)
    def process(trace):
        if use_envelope:
            trace = get_hilbert_envelope(trace)
        return (trace - np.mean(trace, axis=-1, keepdims=True)) / np.std(trace, axis=-1, keepdims=True)

    n_samples = len(ref_trace)
    ref_processed = process(ref_trace)

    # Process all traces
    trace_set = np.stack(trace_set)  # Make sure it's 2D
    traces_processed = process(trace_set)

    sum_trace = np.copy(ref_trace)

    lag_array = scipy.signal.correlation_lags(n_samples, n_samples, mode='full')

    for i, trace in enumerate(trace_set):
        corr = scipy.signal.correlate(ref_processed, traces_processed[i], mode='full') / n_samples
        best_lag = lag_array[np.argmax(corr)]
        aligned_trace = np.roll(trace, best_lag)
        sum_trace += aligned_trace

    return sum_trace


def get_entropy(trace, n_hist_bins = 50):
    """
    Calculates the shannon entropy (randomness measurement) of a trace.

    Parameters
    ----------
    trace: array of floats
        Trace of a waveform
    n_hist_bins: int
        Number of bins for the histogram (default: 50)

    Returns
    -------
    entropy: float
        Shannon entropy of the signal (trace)
    """

    # Step 1: Discretize the signal into bins
    # If density = True, the result is the value of the probability density function at the bin,
    # normalized such that the integral over the range is 1.
    hist, bin_edges = np.histogram(trace, bins = n_hist_bins, density = True)

    # Step 2: Calculate the probability distribution (normalized)
    probabilities = hist / np.sum(hist)

    # Step 3: Calculate Shannon Entropy
    # Using base = 2 for entropy in bits
    entropy = scipy.stats.entropy(probabilities, base = 2)

    return entropy


def get_kurtosis(trace):
    """
    Calculates the kurtosis (tailedness) of a trace.

    Parameters
    ----------
    trace: array of floats
        Trace of a waveform

    Returns
    -------
    kurtosis: float
        Kurtosis of the signal (trace)
    """
    kurtosis = scipy.stats.kurtosis(trace)
    return kurtosis


def get_teager_kaiser_energy(trace):
    """
    Uses the Teager-Kaiser Energy Operator (TKEO) on a trace.

    Parameters
    ----------
    trace: array of floats
        Trace of a waveform

    Returns
    -------
    np.abs(tkeo): array of floats
        TKEO of the input trace
    """
    tkeo = np.zeros_like(trace)
    tkeo[1:-1] = trace[1:-1]**2 - trace[0:-2] * trace[2:]

    return np.abs(tkeo)


def is_NAN_or_INF(trace):
    """
    To see if a trace has any NAN or INF.

    If there's any NAN or any INF,
    this function will tell us how many points of NAN and INF there are.

    Parameters
    ----------
    trace: array of floats
        Trace of a waveform

    Returns
    -------
    is_bad_trace: bool
        True if there's any or False if the trace doesn't have any unreadable point
    npoints_NAN: int
        Number of NAN points
    npoints_INF: int
        Number of INF points
    """
    is_bad_trace = False

    trace = np.array(trace)

    npoints_NAN = len(np.argwhere(np.isnan(trace)))
    npoints_INF = len(np.argwhere(np.isinf(trace)))

    if npoints_NAN or npoints_INF:
        is_bad_trace = True

    return is_bad_trace, npoints_NAN, npoints_INF


def get_variable_window_size_correlation(data_trace, template_trace, window_size, sampling_rate=3.2*units.GHz, return_time_difference=False, debug=False):
    """
    Calculate the correlation between two traces using a variable window size and matrix multiplication

    Parameters
    ----------
    data_trace: array
        full trace of the data event
    template_trace: array
        full trace of the template
    window_size: float
        Size of the template window in nanoseconds
    sampling_rate: float
        sampling rate of the data and template trace
    return_time_difference: boolean
        if true, the time difference (for the maximal correlation value) between the starting of the data
        trace and the starting of the (cut) template trace is returned (returned time is in units.ns)
    debug: boolean
        if true, debug plots are created

    Returns
    -------
    correlation : array of floats
    time_diff : float, optional
        The time difference of the maximal correlation value. Returned only if ``return_time_difference==True``
    """
    # preparing the traces
    data_trace = np.asarray(data_trace, dtype=float)
    template_trace = np.asarray(template_trace, dtype=float)

    # create the template window
    window_steps = int(window_size * sampling_rate)

    max_amp = np.max(abs(template_trace))
    max_amp_i = np.where(abs(template_trace) == max_amp)[0][0]
    lower_bound = int(max_amp_i - window_steps / 3)
    upper_bound = int(max_amp_i + 2 * window_steps / 3)
    template_trace = template_trace[lower_bound:upper_bound]

    # zero padding on the data trace
    data_trace = np.append(np.zeros(len(template_trace) - 1), data_trace)
    data_trace = np.append(data_trace, np.zeros(len(template_trace) - 1))

    if debug:
        plot_data_trace = data_trace.copy()

    # only calculate the correlation of the part of the trace where at least 10% of the maximum is visible (fastens the calculation)
    max_amp_data = np.max(abs(data_trace))
    help_val = np.where(abs(data_trace) >= 0.1 * max_amp_data)[0]
    lower_bound_data = help_val[0] - (len(template_trace) - 1)
    upper_bound_data = help_val[len(help_val) - 1] + (len(template_trace) - 1)
    data_trace = data_trace[lower_bound_data:upper_bound_data]

    # run the correlation using matrix multiplication
    dataMatrix = np.lib.stride_tricks.sliding_window_view(data_trace, len(template_trace))
    corr_numerator = dataMatrix.dot(template_trace)
    norm_dataMatrix = np.linalg.norm(dataMatrix, axis=1)
    norm_template_trace = np.linalg.norm(template_trace)
    corr_denominator = norm_dataMatrix * norm_template_trace
    correlation = corr_numerator / corr_denominator

    max_correlation = np.max(abs(correlation))
    max_corr_i = np.where(abs(np.asarray(correlation)) == max_correlation)[0][0]

    if return_time_difference:
        # calculate the time difference between the beginning of the template and data trace for the largest correlation value
        # time difference is given in ns
        time_diff = (max_corr_i + (lower_bound_data - len(template_trace))) / sampling_rate

    logger.debug('max correlation: {}'.format(max_correlation))
    if return_time_difference:
        logger.debug('time difference: {:.2f} ns'.format(time_diff))

    if debug:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2)
        axs[0].plot(correlation)
        axs[0].plot(np.array([np.where(abs(correlation) == max(abs(correlation)))[0][0]]), np.array([max_correlation]), marker="x", markersize=12, color='tab:red')
        axs[0].set_ylim(-1.1, 1.1)
        axs[0].set_ylabel(r"$\chi$")
        axs[0].set_xlabel('N')
        axs[1].plot(plot_data_trace, label='complete data trace')
        x_data = np.arange(0, len(data_trace), 1)
        x_data = x_data + lower_bound_data
        axs[1].plot(x_data, data_trace, label='scanned data trace')
        x_template = np.arange(0, len(template_trace), 1)
        x_template = x_template + max_corr_i + lower_bound_data
        axs[1].plot(x_template, template_trace, label='template')
        axs[1].set_xlabel('time')
        axs[1].set_ylabel('amplitude')
        axs[1].legend()
        fig.savefig('debug_plots_get_variable_window_size_correlation.png')

    if return_time_difference:
        return correlation, time_diff
    else:
        return correlation
