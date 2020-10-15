import numpy as np
import scipy.signal
from NuRadioReco.detector import filterresponse


def get_filter_response(frequencies, passband, filter_type, order):
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
    elif (filter_type.find('FIR') >= 0):
        raise NotImplementedError("FIR filter not yet implemented")
    else:
        return filterresponse.get_filter_response(frequencies, filter_type)
