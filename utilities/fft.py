import numpy as np


def time2freq(trace):
    """
    performs forward FFT with correct normalization that conserves the power
    """
    return np.fft.rfft(trace, axis=-1, norm="ortho") * 2 ** 0.5  # an additional sqrt(2) is added because negative frequencies are omitted.


def freq2time(spectrum):
    """
    performs backward FFT with correct normalization that conserves the power
    """
    return np.fft.irfft(spectrum, axis=-1, norm="ortho") / 2 ** 0.5
