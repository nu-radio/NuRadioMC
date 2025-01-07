import numpy as np
import scipy.constants
import scipy.signal
from NuRadioReco.utilities import units
from NuRadioReco.utilities import ice
from NuRadioReco.utilities import geometryUtilities as geo_utl
from NuRadioReco.utilities import fft
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

def digital_upsampling(trace, adc_sampling_frequency, upsampling_method='fft', upsampling_factor=2, coeff_gain=1, filter_taps=45, adc_output='voltage', rnog_like=False):
    """
    Digital upsampling with various methods and settings.

    Parameters
    ----------
    trace : 1d array (float or int)
        Input trace to upsample
    adc_sampling_frequency : float
        Original sampling frequency for trace
    upsampling_method: str (default 'fft')
        Choose between FFT, FIR, or Linear Interpolaion based upsampling methods
    upsampling_factor : float (default 2)
        The factor which the sampling frequency increases
    coeff_gain: int (default 1)
        If using the FIR upsampling, this will convert the floating point output of the 
        scipy filter to a fixed point value by multiplying by this factor and rounding to an int.
        If set to 1, this will preserve the float value of the filter coefficients.
    filter_taps : int (default 45)
        Number of taps in the FIR filter in FIR-based upsampling.
    adc_output: string (default "voltage")
        - "voltage" : keep upsampled trace as floats
        - "counts" : round upsampling trace to ints
    rnog_like: bool (default False)
        If true, this will apply the RNO-G FLOWER upsampling settings and the fixed point math/rounding done in firmware.
    Returns
    -------
    upsampled_trace : 1d array (float or int)
        Upsampled trace at the new sampling frequency
    new_sampling_frequency : float
        New sampling frequency
    """

    if (np.abs(int(upsampling_factor) - upsampling_factor) > 1e-3):
        warning_msg = "The input upsampling factor does not seem to be close to an integer."
        warning_msg += "It has been rounded to {}".format(int(upsampling_factor))
        logger.warning(warning_msg)

    upsampling_factor = int(upsampling_factor)

    if (upsampling_factor <= 1):
        error_msg = "Upsampling factor is less or equal to 1. Upsampling will not be performed."
        upsampled_trace = trace
        new_sampling_freq = adc_sampling_frequency
    
    else:
        new_sampling_freq = adc_sampling_frequency * upsampling_factor
        new_len = len(trace) * upsampling_factor 

        if upsampling_method=='fft':
            upsampled_trace = scipy.signal.resample(trace, new_len)

        elif upsampling_method=='lin':
            cur_t = np.arange(0, 1/adc_sampling_frequency*len(trace), 1/adc_sampling_frequency)
            new_t = np.arange(0, 1/adc_sampling_frequency*len(trace), 1/new_sampling_freq)
            upsampled_trace = np.interp(new_t, cur_t, trace)

        elif upsampling_method=='fir':        

            if rnog_like:
                filter_taps = 45
                coeff_gain = 128

            upsampled_trace=upsampling_fir(trace, adc_sampling_frequency, int_factor=upsampling_factor, ntaps=filter_taps, coeff_gain=coeff_gain)

        else:
            error_msg = 'Interpolation method must be lin, fft, or fir'
            raise NotImplementedError(error_msg)

        if adc_output=='counts':
            upsampled_trace = np.round(upsampled_trace)

    return upsampled_trace, new_sampling_freq

def upsampling_fir(trace, original_sampling_frequency, int_factor=2, ntaps=2 ** 7, coeff_gain=128):
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

    cutoff = 0.5
    up_filt = scipy.signal.firwin(ntaps, original_sampling_frequency*cutoff, pass_zero='lowpass',
                                fs=original_sampling_frequency*int_factor)
    if coeff_gain!=1:
        up_filt = np.round(up_filt*coeff_gain)/coeff_gain

    zero_padded_sig = np.zeros(len(trace)*int_factor)
    zero_padded_sig[::int_factor] = trace[:]
    upsampled_trace = np.convolve(zero_padded_sig, up_filt, mode='full')[len(up_filt)//2 : len(zero_padded_sig) + len(up_filt)//2] * int_factor      

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


def delay_trace(trace, sampling_frequency, time_delay, delayed_samples=None):
    """
    Delays a trace by transforming it to frequency and multiplying by phases.
    Since this method is cyclic, the trace has to be cropped. It only accepts
    positive delays, so some samples from the beginning are thrown away and then
    some samples from the end so that the total number of samples is equal to
    the argument delayed samples.

    Parameters
    ----------
    trace: array of floats
        Array containing the trace
    sampling_frequency: float
        Sampling rate for the trace
    time_delay: float
        Time delay used for transforming the trace. Must be positive or 0
    delayed_samples: integer or None
        Number of samples that the delayed trace must contain
        if None: the trace is not cut

    Returns
    -------
    delayed_trace: array of floats
        The delayed, cropped trace
    """

    if time_delay < 0:
        msg = 'Time delay must be positive'
        raise ValueError(msg)

    n_samples = len(trace)

    spectrum = fft.time2freq(trace, sampling_frequency)
    frequencies = np.fft.rfftfreq(n_samples, 1 / sampling_frequency)

    spectrum *= np.exp(-1j * 2 * np.pi * frequencies * time_delay)

    delayed_trace = fft.freq2time(spectrum, sampling_frequency)

    init_sample = int(time_delay * sampling_frequency) + 1

    if delayed_samples is not None:
        delayed_trace = delayed_trace[init_sample:None]
        delayed_trace = delayed_trace[:delayed_samples]

    return delayed_trace
