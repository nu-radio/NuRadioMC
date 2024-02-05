import numpy as np
import scipy.signal
from NuRadioReco.detector import filterresponse


def get_filter_response(frequencies, passband, filter_type, order, rp=None):
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

        or any filter that is implemented in :mod:`NuRadioReco.detector.filterresponse`.
        In this case the passband parameter is ignored
    order: int
        for a butterworth filter: specifies the order of the filter
    rp: float
        The maximum ripple allowed below unity gain in the passband.
        Specified in decibels, as a positive number.
        (Relevant for chebyshev filter)

    Returns
    -------
    f: array of floats
        The bandpass filter response. Has the same shape as ``frequencies``.

    """

    # we need to specify if we have a lowpass filter
    # otherwise scipy>=1.8.0 raises an error
    if passband[0] == 0:
        scipy_args = [passband[1], 'lowpass']
    else:
        scipy_args = [passband, 'bandpass']

    if filter_type == 'rectangular':
        return np.where(passband[0] <= frequencies <= passband[1], 1, 0)

    elif filter_type == 'butter':
        mask = frequencies > 0
        b, a = scipy.signal.butter(order, *scipy_args, analog=True)
        w, h = scipy.signal.freqs(b, a, frequencies[mask])
        return np.where(mask, h, 0)

    elif filter_type == 'butterabs':
        mask = frequencies > 0
        b, a = scipy.signal.butter(order, *scipy_args, analog=True)
        w, h = scipy.signal.freqs(b, a, frequencies[mask])
        return np.abs(np.where(mask, h, 0))

    elif filter_type == 'cheby1':
        mask = frequencies > 0
        b, a = scipy.signal.cheby1(order, rp, *scipy_args, analog=True)
        w, h = scipy.signal.freqs(b, a, frequencies[mask])
        return np.where(mask, h, 0)

    elif filter_type.find('FIR') >= 0:
        raise NotImplementedError("FIR filter not yet implemented")
    else:
        return filterresponse.get_filter_response(frequencies, filter_type)
