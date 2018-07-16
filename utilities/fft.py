import numpy as np

"""
the fft utility module implements a real fft with a normalization so that
Plancherel theorem is valid. This means that the fourier transform must be unitary,
which is achieved by normalizing both transforms with 1/sqrt(n) and an additional
factor of sqrt(2) because negative frequencies are omitted in the real FFT.
This means that the power calculated in the time domain is the same as the
power calculated in the frequency domain.
"""


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
