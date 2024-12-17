import numpy as np
import scipy.signal
from NuRadioReco.detector import filterresponse


def get_filter_response(frequencies, passband, filter_type, order, rp=None, roll_width=None):
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

    if filter_type == 'rectangular':
        mask = np.all([passband[0] <= frequencies, frequencies <= passband[1]], axis=0)
        return np.where(mask, 1, 0)

    # we need to specify if we have a lowpass filter
    # otherwise scipy>=1.8.0 raises an error
    if passband[0] == 0:
        scipy_args = [passband[1], 'lowpass']
    else:
        scipy_args = [passband, 'bandpass']

    if filter_type == 'butter':
        f = np.zeros_like(frequencies, dtype=complex)
        mask = frequencies > 0
        b, a = scipy.signal.butter(order, *scipy_args, analog=True)
        w, h = scipy.signal.freqs(b, a, frequencies[mask])
        f[mask] = h
        return f

    elif filter_type == 'butterabs':
        f = np.zeros_like(frequencies, dtype=complex)
        mask = frequencies > 0
        b, a = scipy.signal.butter(order, *scipy_args, analog=True)
        w, h = scipy.signal.freqs(b, a, frequencies[mask])
        f[mask] = h
        return np.abs(f)

    elif filter_type == 'cheby1':
        f = np.zeros_like(frequencies, dtype=complex)
        mask = frequencies > 0
        b, a = scipy.signal.cheby1(order, rp, *scipy_args, analog=True)
        w, h = scipy.signal.freqs(b, a, frequencies[mask])
        f[mask] = h
        return f

    elif filter_type == 'gaussian_tapered':
        f = np.ones_like(frequencies, dtype=complex)
        f[np.where(frequencies < passband[0])] = 0.
        f[np.where(frequencies > passband[1])] = 0.

        gaussian_weights = scipy.signal.windows.gaussian(
            len(frequencies), int(round(roll_width / (frequencies[1] - frequencies[0])))
        )

        f = scipy.signal.convolve(f, gaussian_weights, mode="same")
        f /= np.max(f)  # convolution changes peak value
        return f

    elif filter_type.find('FIR') >= 0:
        raise NotImplementedError("FIR filter not yet implemented")

    elif filter_type == 'hann_tapered':
        raise NotImplementedError(
            "'hann_tapered' is a time-domain filter, cannot return frequency response"
        )

    else:
        return filterresponse.get_filter_response(frequencies, filter_type)
