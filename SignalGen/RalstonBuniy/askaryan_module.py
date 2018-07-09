from NuRadioMC.SignalGen.RalstonBuniy import create_askaryan
from NuRadioMC.utilities import units, fft
import numpy as np


def get_frequency_spectrum(energy, theta, freqs, is_em_shower, n, R, LPM=True):
    eR, eTheta, ePhi = create_askaryan.get_frequency_spectrum(energy, theta, freqs, is_em_shower, n, R, LPM)
    # normalize amplitudes correctly to the bin width of the freqs array
    # the output of the Askaryan module is normlized to 1MHz
    df = np.mean(freqs[1:] - freqs[:-1])
    eR *= (df / units.MHz) ** 0.5
    eTheta *= (df / units.MHz) ** 0.5
    ePhi *= (df / units.MHz) ** 0.5
    return eR, eTheta, ePhi


def get_time_trace(energy, theta, freqs, is_em_shower, n, R, LPM=True):
    eR, eTheta, ePhi = get_frequency_spectrum(energy, theta, freqs, is_em_shower, n, R, LPM)
    length = (len(freqs) - 1) * 2
    sampling_rate = 2 * freqs[-1]
#     tt = np.arange(0, length / sampling_rate, 1. / sampling_rate)
    traceR = fft.freq2time(eR)
    traceTheta = fft.freq2time(eTheta)
    tracePhi = fft.freq2time(ePhi)
    return traceR, traceTheta, tracePhi
