import numpy as np

"""
A wrapper around the numpy fft routines to achive a coherent normalization of the fft
As we have real valued data in the time domain, we use the 'real ffts' that omit the negative frequencies in Fourier 
space. To account for the missing power in the frequency domain, we multiply the frequency spectrum by sqrt(2), 
and divide the iFFT with 1/sqrt(2) accordingly. Then, a calculation of the power leads the same result in 
the time and frequency domain, i.e.
np.sum(trace**2) * dt = np.sum(spectrum**2) * df
"""


def time2freq(trace):
    """
    performs forward FFT with correct normalization that conserves the power
    """
    return np.fft.rfft(trace, axis=-1) * 2 ** 0.5  # an additional sqrt(2) is added because negative frequencies are omitted.


def freq2time(spectrum, n=None):
    """
    performs backward FFT with correct normalization that conserves the power
    
    Parameters
    ----------
    spectrum: complex np array
        the frequency spectrum
    n: int
        the number of sample in the time domain (relevant if time trace has an odd number of samples)
    """
    return np.fft.irfft(spectrum, axis=-1, n=n) / 2 ** 0.5
