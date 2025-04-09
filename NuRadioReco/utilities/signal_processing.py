"""
This module contains various functions for signal processing.

Functions for filtering, delaying, and upsampling traces:

- `half_hann_window`
- `resample`
- `upsampling_fir`
- `get_filter_response`
- `butterworth_filter_trace`
- `apply_butterworth`
- `delay_trace`

It also contains functions to calculate the electric field or voltage amplitude
from a given noise temperature and bandwidth:

- `get_electric_field_from_temperature`
- `calculate_vrms_from_temperature`

See Also
--------
`NuRadioReco.utilities.trace_utilities`
    Contains functions to calculate observables from traces.
"""

from NuRadioReco.utilities import units, geometryUtilities as geo_utl, fft

from NuRadioReco.detector import filterresponse
import NuRadioReco.framework.base_trace

from scipy.signal.windows import hann
from scipy import signal, constants
import numpy as np
import fractions
import decimal
import copy

import logging

logger = logging.getLogger("NuRadioReco.utilities.signal_processing")


def half_hann_window(length, half_percent=None, hann_window_length=None):
    """
    Produce a half-Hann window. This is the Hann window from SciPY with ones inserted in the
    middle to make the window `length` long. Note that this is different from a Hamming window.

    Parameters
    ----------
    length : int
        The desired total length of the window
    half_percent : float, default=None
        The percentage of `length` at the beginning **and** end that should correspond to half of the Hann window
    hann_window_length : int, default=None
        The length of the half the Hann window. If `half_percent` is set, this value will be overwritten by it.
    """
    if half_percent is not None:
        hann_window_length = int(length * half_percent)
    elif hann_window_length is None:
        raise ValueError("Either half_percent or half_window_length should be set!")
    hann_window = hann(2 * hann_window_length)

    half_hann_widow = np.ones(length, dtype=np.double)
    half_hann_widow[:hann_window_length] = hann_window[:hann_window_length]
    half_hann_widow[-hann_window_length:] = hann_window[hann_window_length:]

    return half_hann_widow


def resample(trace, sampling_factor):
    """Resample a trace by a given resampling factor.

    Parameters
    ----------
    trace : ndarray
        The trace to resample. Can have multiple dimensions, but the last dimension should be the one to resample.
    sampling_factor : float
        The resampling factor. If the factor is a fraction, the denominator should be less than 5000.

    Returns
    -------
    resampled_trace : ndarray
        The resampled trace.
    """
    resampling_factor = fractions.Fraction(
        decimal.Decimal(sampling_factor)
    ).limit_denominator(5000)
    n_samples = trace.shape[-1]
    resampled_trace = copy.copy(trace)

    if resampling_factor.numerator != 1:
        # resample and use axis -1 since trace might be either shape (N) for analytic trace or shape (3,N) for E-field
        resampled_trace = signal.resample(
            resampled_trace, resampling_factor.numerator * n_samples, axis=-1
        )

    if resampling_factor.denominator != 1:
        # resample and use axis -1 since trace might be either shape (N) for analytic trace or shape (3,N) for E-field
        resampled_trace = signal.resample(
            resampled_trace,
            np.shape(resampled_trace)[-1] // resampling_factor.denominator,
            axis=-1,
        )

    if resampled_trace.shape[-1] % 2 != 0:
        resampled_trace = resampled_trace.T[:-1].T

    return resampled_trace


def upsampling_fir(trace, original_sampling_frequency, int_factor=2, ntaps=2**7):
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

    if np.abs(int(int_factor) - int_factor) > 1e-3:
        warning_msg = (
            "The input upsampling factor does not seem to be close to an integer."
        )
        warning_msg += "It has been rounded to {}".format(int(int_factor))
        logger.warning(warning_msg)

    int_factor = int(int_factor)

    if int_factor <= 1:
        error_msg = (
            "Upsampling factor is less or equal to 1. Upsampling will not be performed."
        )
        raise ValueError(error_msg)

    zeroed_trace = np.zeros(len(trace) * int_factor)
    for i_point, point in enumerate(trace[:-1]):
        zeroed_trace[i_point * int_factor] = point

    upsampled_delta_time = 1 / (int_factor * original_sampling_frequency)
    upsampled_times = np.arange(
        0, len(zeroed_trace) * upsampled_delta_time, upsampled_delta_time
    )

    cutoff = 1.0 / int_factor
    fir_coeffs = signal.firwin(ntaps, cutoff, window="boxcar")
    upsampled_trace = (
        np.convolve(zeroed_trace, fir_coeffs)[: len(upsampled_times)] * int_factor
    )

    return upsampled_trace


def get_filter_response(
    frequencies, passband, filter_type, order, rp=None, roll_width=None
):
    """
    Convenience function to obtain a bandpass filter response

    Parameters
    ----------
    frequencies: array of floats
        the frequencies the filter is requested for
    passband: list
        passband[0]: lower boundary of filter, passband[1]: upper boundary of filter
    filter_type: string or dict

        * 'rectangular': perfect straight line filter
        * 'butter': butterworth filter from scipy
        * 'butterabs': absolute of butterworth filter from scipy
        * 'gaussian_tapered' : a rectangular bandpass filter convolved with a Gaussian

        or any filter that is implemented in :mod:`NuRadioReco.detector.filterresponse`.
        In this case the passband parameter is ignored
    order: int
        for a butterworth filter: specifies the order of the filter
    rp: float
        The maximum ripple allowed below unity gain in the passband.
        Specified in decibels, as a positive number.
        (Relevant for chebyshev filter)
    roll_width : float, default=None
        Determines the sigma of the Gaussian to be used in the convolution of the rectangular filter.
        (Relevant for the Gaussian tapered filter)

    Returns
    -------
    f: array of floats
        The bandpass filter response. Has the same shape as ``frequencies``.

    """

    if filter_type == "rectangular":
        mask = np.all([passband[0] <= frequencies, frequencies <= passband[1]], axis=0)
        return np.where(mask, 1, 0)

    # we need to specify if we have a lowpass filter
    # otherwise scipy>=1.8.0 raises an error
    if passband[0] == 0:
        scipy_args = [passband[1], "lowpass"]
    else:
        scipy_args = [passband, "bandpass"]

    if filter_type == "butter":
        f = np.zeros_like(frequencies, dtype=complex)
        mask = frequencies > 0
        b, a = signal.butter(order, *scipy_args, analog=True)
        w, h = signal.freqs(b, a, frequencies[mask])
        f[mask] = h
        return f

    elif filter_type == "butterabs":
        f = np.zeros_like(frequencies, dtype=complex)
        mask = frequencies > 0
        b, a = signal.butter(order, *scipy_args, analog=True)
        w, h = signal.freqs(b, a, frequencies[mask])
        f[mask] = h
        return np.abs(f)

    elif filter_type == "cheby1":
        f = np.zeros_like(frequencies, dtype=complex)
        mask = frequencies > 0
        b, a = signal.cheby1(order, rp, *scipy_args, analog=True)
        w, h = signal.freqs(b, a, frequencies[mask])
        f[mask] = h
        return f

    elif filter_type == "gaussian_tapered":
        f = np.ones_like(frequencies, dtype=complex)
        f[np.where(frequencies < passband[0])] = 0.0
        f[np.where(frequencies > passband[1])] = 0.0

        gaussian_weights = signal.windows.gaussian(
            len(frequencies), int(round(roll_width / (frequencies[1] - frequencies[0])))
        )

        f = signal.convolve(f, gaussian_weights, mode="same")
        f /= np.max(f)  # convolution changes peak value
        return f

    elif filter_type.find("FIR") >= 0:
        raise NotImplementedError("FIR filter not yet implemented")

    elif filter_type == "hann_tapered":
        raise NotImplementedError(
            "'hann_tapered' is a time-domain filter, cannot return frequency response"
        )

    else:
        return filterresponse.get_filter_response(frequencies, filter_type)


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
    frequencies = fft.freqs(n_samples, sampling_frequency)

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
    b, a = signal.butter(order, passband, "bandpass", analog=True)
    w, h = signal.freqs(b, a, frequencies[mask])
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
            raise ValueError(
                "The sampling frequency of the trace does not match the given sampling frequency."
            )
    else:
        n_samples = len(trace)
        spectrum = fft.time2freq(trace, sampling_frequency)
        frequencies = fft.freqs(n_samples, sampling_frequency)

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
                logger.warning(
                    "The delayed trace has unphysical samples that contain signal. "
                    "Consider cropping the trace to remove these samples."
                )
        else:
            if np.any(np.abs(delayed_trace[-cycled_samples:]) > 0.01 * units.microvolt):
                logger.warning(
                    "The delayed trace has unphysical samples that contain signal. "
                    "Consider cropping the trace to remove these samples."
                )

        return delayed_trace


def get_electric_field_from_temperature(frequencies, noise_temperature, solid_angle):
    """
    Calculate the electric field amplitude from the radiance of a radio signal.

    The radiance is calculated using the Rayleigh-Jeans law per frequency bin, by adjusting
    the value with the frequency spacing. After this, the electric field amplitude per bin
    is calculated using the radiance and the vacuum permittivity.

    Parameters
    ----------
    frequencies: array of floats
        The frequencies at which to calculate the electric field amplitude
    noise_temperature: float
        The noise temperature to use in the Rayleigh-Jeans law
    solid_angle: float
        The solid angle over which the radiance is integrated

    Returns
    -------
    efield_amplitude: array of floats
        The electric field amplitude at each frequency
    """
    # Get constants in correct units
    boltzmann = constants.Boltzmann * units.joule / units.kelvin
    epsilon_0 = constants.epsilon_0 * (units.coulomb / units.V / units.m)
    c_vac = constants.c * units.m / units.s

    # Calculate frequency spacing
    d_f = frequencies[2] - frequencies[1]

    # Calculate spectral radiance of radio signal using Rayleigh-Jeans law
    spectral_radiance = (
        2.0 * boltzmann * frequencies**2 * noise_temperature * solid_angle / c_vac**2
    )
    spectral_radiance[np.isnan(spectral_radiance)] = 0

    # calculate radiance per energy bin
    spectral_radiance_per_bin = spectral_radiance * d_f

    # calculate electric field per energy bin from the radiance per bin
    efield_amplitude = np.sqrt(spectral_radiance_per_bin / (c_vac * epsilon_0)) / d_f

    return efield_amplitude


def calculate_vrms_from_temperature(temperature, bandwidth=None, response=None, impedance=50 * units.ohm, freqs=None):
    """ Helper function to calculate the noise vrms from a given noise temperature and bandwidth.

    For details see https://en.wikipedia.org/wiki/Johnson%E2%80%93Nyquist_noise
    (sec. "Maximum transfer of noise power") or our wiki
    https://nu-radio.github.io/NuRadioMC/NuRadioMC/pages/HDF5_structure.html

    Parameters
    ----------
    temperature: float
        The noise temperature of the channel in Kelvin
    bandwidth: float or tuple of 2 floats (list of 2 floats) (default: None)
        If single float, this argument is interpreted as the effective bandwidth. If tuple, the argument is
        interpreted as the lower and upper frequency of the bandwidth. Can be `None` if `response` is specified.
    response: `NuRadioReco.detector.response.Response` (default: None)
        If not None, the response of the channel is taken into account to calculate the noise vrms.
    impedance: float (default: 50)
        Electrical impedance of the channel in Ohm.
    freqs: array_like (default: None -> np.arange(0, 2500, 0.1) * units.MHz)
        Frequencies at which the response is evaluated. Only used if `response` is not None.

    Returns
    -------
    vrms_per_channel: float
        The vrms of the channel
    """
    if bandwidth is None and response is None:
        raise ValueError("Please specify bandwidth or response")

    if impedance > 1000 * units.ohm:
        logger.warning(
            f"Impedance is {impedance / units.ohm:.2f} Ohm, did you forget to specify the unit?"
        )

    # (effective) bandwidth, i.e., \Delta f in equation
    if response is None:
        if not isinstance(bandwidth, (float, int)):
            bandwidth = bandwidth[1] - bandwidth[0]
    else:
        freqs = freqs or np.arange(0, 2500, 0.1) * units.MHz
        bandwidth = np.trapz(np.abs(response(freqs)) ** 2, freqs)

    return (temperature * impedance * bandwidth * constants.k * units.joule / units.kelvin) ** 0.5


def get_efield_antenna_factor(station, frequencies, channels, detector, zenith, azimuth, antenna_pattern_provider, efield_is_at_antenna=False):
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
    efield_is_at_antenna: bool (defaul: False)
        If True, the electric field is assumed to be at the antenna.
        If False, the effects of an air-ice boundary are taken into account if necessary.
        The default is set to False to keep backwards compatibility.

    Returns
    -------
    efield_antenna_factor: list of array of complex values
        The antenna response for each channel at each frequency

    See Also
    --------
    NuRadioReco.utilities.geometryUtilities.fresnel_factors_and_signal_zenith : function that calculates the Fresnel factors
    """

    efield_antenna_factor = np.zeros((len(channels), 2, len(frequencies)), dtype=complex)  # from antenna model in e_theta, e_phi
    for iCh, channel_id in enumerate(channels):
        if not efield_is_at_antenna:
            zenith_antenna, t_theta, t_phi = geo_utl.fresnel_factors_and_signal_zenith(
                detector, station, channel_id, zenith)
        else:
            zenith_antenna = zenith
            t_theta = 1
            t_phi = 1

        if zenith_antenna is None:
            logger.warning(
                "Fresnel reflection at air-firn boundary leads to unphysical results, "
                "no reconstruction possible")
            return None

        logger.debug("angles: zenith {0:.0f}, zenith antenna {1:.0f}, azimuth {2:.0f}".format(
            np.rad2deg(zenith), np.rad2deg(zenith_antenna), np.rad2deg(azimuth)))

        antenna_model = detector.get_antenna_model(station.get_id(), channel_id, zenith_antenna)
        antenna_pattern = antenna_pattern_provider.load_antenna_pattern(antenna_model)
        ori = detector.get_antenna_orientation(station.get_id(), channel_id)
        VEL = antenna_pattern.get_antenna_response_vectorized(frequencies, zenith_antenna, azimuth, *ori)
        efield_antenna_factor[iCh] = np.array([VEL['theta'] * t_theta, VEL['phi'] * t_phi])

    return efield_antenna_factor


def get_channel_voltage_from_efield(
        station, electric_field, channels, detector,
        zenith, azimuth, antenna_pattern_provider, return_spectrum=True):
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
    efield_antenna_factor = get_efield_antenna_factor(
        station, frequencies, channels, detector, zenith, azimuth, antenna_pattern_provider)

    voltage_spectrum = np.array([
        np.sum(efield_antenna_factor[i_ch] * np.array([spectrum[1], spectrum[2]]), axis=0)
        for i_ch, _ in enumerate(channels)])

    if return_spectrum:
        return voltage_spectrum
    else:
        voltage_trace = fft.freq2time(voltage_spectrum, electric_field.get_sampling_rate())
        return np.real(voltage_trace)
