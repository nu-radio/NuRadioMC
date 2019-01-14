from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import scipy.signal
from NuRadioReco.utilities import units, fft

conversion_factor_integrated_signal = 2.65441729 * 1e-3 * 1.e-9 * 6.24150934 * 1e18


def amp_from_energy(energy):
    """
    energy is defined as the integral of squared voltage normalized to a time window of 128 ns
    """
    return 0.5 * np.log10(energy) + 0.12876705


def get_analytic_pulse_freq(amp_p0, amp_p1, phase_p0, n_samples_time, sampling_rate,
                            phase_p1=0, bandpass=None, filter_type='butter10', quadratic_term=0, quadratic_term_offset = 0):
    amp_p0 /= conversion_factor_integrated_signal  # input variable is energy in eV/m^2
    dt = 1. / sampling_rate
    frequencies = np.fft.rfftfreq(n_samples_time, dt)
    df = frequencies[1] - frequencies[0]
    A = np.sign(amp_p0) * (np.abs(amp_p0)) ** 0.5
    amps = A * 10 ** (frequencies * amp_p1 + quadratic_term * (frequencies - quadratic_term_offset)**2)
    norm = 1.
    if(bandpass is None):
        norm = -1. / (2 * amp_p1 * np.log(10))
    else:
        if(amp_p1 == 0):
            norm = bandpass[1] - bandpass[0]
        else:
            norm = (100 ** (amp_p1 * bandpass[1]) - 100 ** (amp_p1 * bandpass[0])) / (2 * amp_p1 * np.log(10))

    phases = phase_p0 + frequencies * phase_p1
    xx = amps * np.exp(phases * 1j) / norm ** 0.5 / dt ** 0.5 * df ** 0.5

    if(bandpass is not None):
        if filter_type == 'butter10' or filter_type == 'butter10abs':
            b, a = scipy.signal.butter(10, bandpass, 'bandpass', analog=True)
            w, h = scipy.signal.freqs(b, a, frequencies)
            if filter_type == 'butter10abs':   
                xx *= np.abs(h)
            else:
                xx *= h
        elif filter_type == 'rectangular':
            xx[frequencies < bandpass[0]] = 0
            xx[frequencies > bandpass[1]] = 0
        else:
            raise ValueError('Unknown filter_type')
    return xx


def get_analytic_pulse(amp_p0, amp_p1, phase_p0, n_samples_time, sampling_rate,
                       phase_p1=0, bandpass=None, filter_type='butter10', quadratic_term=0, quadratic_term_offset=0):
    xx = get_analytic_pulse_freq(amp_p0, amp_p1, phase_p0, n_samples_time, sampling_rate,
                                 phase_p1=phase_p1, bandpass=bandpass, filter_type=filter_type, quadratic_term=quadratic_term, quadratic_term_offset=quadratic_term_offset)
    return fft.freq2time(xx)
