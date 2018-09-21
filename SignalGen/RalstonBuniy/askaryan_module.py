from NuRadioMC.SignalGen.RalstonBuniy import create_askaryan
from NuRadioMC.utilities import units, fft
import numpy as np

def get_time_trace(energy, theta, N, dt, is_em_shower, n, R, LPM=True, a=None):
    if(a is None):
        a = 0
    freqs = np.fft.rfftfreq(N, dt)
    eR, eTheta, ePhi = create_askaryan.get_frequency_spectrum(energy, theta, freqs, is_em_shower, n, R, LPM, a)
    traceR = np.fft.irfft(eR) / dt
    traceTheta = np.fft.irfft(eTheta) / dt
    tracePhi = np.fft.irfft(ePhi) / dt
    return np.array([traceR, traceTheta, tracePhi])


def get_frequency_spectrum(energy, theta, N, dt, is_em_shower, n, R, LPM=True, a=None):
    eR, eTheta, ePhi = get_time_trace(energy, theta, N, dt, is_em_shower, n, R, LPM, a)
    return np.array([fft.time2freq(eR), fft.time2freq(eTheta), fft.time2freq(ePhi)])
