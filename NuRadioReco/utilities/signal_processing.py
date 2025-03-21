from scipy.signal.windows import hann
from scipy import signal
import numpy as np
import fractions
import decimal
import copy


def half_hann_window(length, half_percent=None, hann_window_length=None):
    """
    Produce a half-Hann window. This is the Hann window from SciPY with ones inserted in the middle to make the window
    `length` long. Note that this is different from a Hamming window.

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
    """ Resample a trace by a given resampling factor.

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
    resampling_factor = fractions.Fraction(decimal.Decimal(sampling_factor)).limit_denominator(5000)
    n_samples = trace.shape[-1]
    resampled_trace = copy.copy(trace)

    if resampling_factor.numerator != 1:
        # resample and use axis -1 since trace might be either shape (N) for analytic trace or shape (3,N) for E-field
        resampled_trace = signal.resample(resampled_trace, resampling_factor.numerator * n_samples, axis=-1)

    if resampling_factor.denominator != 1:
        # resample and use axis -1 since trace might be either shape (N) for analytic trace or shape (3,N) for E-field
        resampled_trace = signal.resample(resampled_trace, np.shape(resampled_trace)[-1] // resampling_factor.denominator, axis=-1)

    if resampled_trace.shape[-1] % 2 != 0:
        resampled_trace = resampled_trace.T[:-1].T

    return resampled_trace
