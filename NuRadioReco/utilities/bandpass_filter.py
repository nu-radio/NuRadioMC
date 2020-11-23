import numpy as np
import scipy.signal
from NuRadioReco.detector import filterresponse


def get_filter_response(frequencies, passband, filter_type, order, rp=None):
    """
    frequencies: array of floats
        the frequencies the filter is requested for
    passband: list or dict of lists, (default: [55 * units.MHz, 1000 * units.MHz])
        passband[0]: lower boundary of filter, passband[1]: upper boundary of filter
        a dict can be used to specify a different bandwidth per channel, the key is the channel_id
    filter_type: string or dict
        'rectangular': perfect straight line filter
        'butter': butterworth filter from scipy
        'butterabs': absolute of butterworth filter from scipy
        or any filter that is implemented in NuRadioReco.detector.filterresponse. In this case the
        passband parameter is ignored
        or 'FIR <type> <parameter>' - see below for FIR filter options

        a dict can be used to specify a different bandwidth per channel, the key is the channel_id
    order: int (optional, default 2) or dict
        for a butterworth filter: specifies the order of the filter

    rp: float
        The maximum ripple allowed below unity gain in the passband. Specified in decibels, as a positive number.
        (for chebyshev filter)

    """
    if (filter_type == 'rectangular'):
        f = np.ones_like(frequencies)
        f[np.where(frequencies < passband[0])] = 0.
        f[np.where(frequencies > passband[1])] = 0.
        return f
    elif (filter_type == 'butter'):
        f = np.zeros_like(frequencies, dtype=np.complex)
        mask = frequencies > 0
        b, a = scipy.signal.butter(order, passband, 'bandpass', analog=True)
        w, h = scipy.signal.freqs(b, a, frequencies[mask])
        f[mask] = h
        return f
    elif (filter_type == 'butterabs'):
        f = np.zeros_like(frequencies, dtype=np.complex)
        mask = frequencies > 0
        b, a = scipy.signal.butter(order, passband, 'bandpass', analog=True)
        w, h = scipy.signal.freqs(b, a, frequencies[mask])
        f[mask] = h
        return np.abs(f)
    elif(filter_type == 'cheby1'):
        f = np.zeros_like(frequencies, dtype=np.complex)
        mask = frequencies > 0
        b, a = scipy.signal.cheby1(order, rp, passband, 'bandpass', analog=True)
        w, h = scipy.signal.freqs(b, a, frequencies[mask])
        f[mask] = h
        return f
    elif (filter_type.find('FIR') >= 0):
        raise NotImplementedError("FIR filter not yet implemented")
    else:
        return filterresponse.get_filter_response(frequencies, filter_type)
