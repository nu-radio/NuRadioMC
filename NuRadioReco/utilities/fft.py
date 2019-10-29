import numpy as np

"""
A wrapper around the numpy fft routines to achive a coherent normalization of the fft
As we have real valued data in the time domain, we use the 'real ffts' that omit the negative frequencies in Fourier
space. To account for the missing power in the frequency domain, we multiply the frequency spectrum by sqrt(2),
and divide the iFFT with 1/sqrt(2) accordingly. The frequency spectrum is divided by the sampling rate so that the
units if the spectrum are volts/GHz instead of volts/bin and the voltage in the frequency domain is independent of the
sampling rate.
Then, a calculation of the power leads the same result in
the time and frequency domain, i.e.
np.sum(trace**2) * dt = np.sum(spectrum**2/dt**2) * df
"""


def time2freq(trace, sampling_rate):
    """
    performs forward FFT with correct normalization that conserves the power

    Parameters
    ----------
    sampling_rate: float
        sampling rate of the trace
    """
    return np.fft.rfft(trace, axis=-1) / sampling_rate  * 2 ** 0.5  # an additional sqrt(2) is added because negative frequencies are omitted.


def freq2time(spectrum, sampling_rate, n=None):
    """
    performs backward FFT with correct normalization that conserves the power

    Parameters
    ----------
    spectrum: complex np array
        the frequency spectrum
    sampling_rate: float
        sampling rate of the spectrum
    n: int
        the number of sample in the time domain (relevant if time trace has an odd number of samples)
    """
    return np.fft.irfft(spectrum, axis=-1, n=n) * sampling_rate  / 2 ** 0.5
