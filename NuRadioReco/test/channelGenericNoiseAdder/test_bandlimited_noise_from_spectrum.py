"""
Test ``channelGenericNoiseAdder.bandlimited_noise_from_spectrum`` with a callable spectrum.

A callable spectrum is evaluated on the frequency grid of the function. It has to give the same noise
(same seed) as the equivalent array spectrum, and the noise power has to lie within the requested band.
"""
import numpy as np

from NuRadioReco.modules.channelGenericNoiseAdder import channelGenericNoiseAdder
from NuRadioReco.utilities import fft, units

N_SAMPLES = 2048
SAMPLING_RATE = 2 * units.GHz
F_MIN, F_MAX = 100 * units.MHz, 200 * units.MHz


def in_band(frequencies):
    return ((frequencies > F_MIN) & (frequencies < F_MAX)).astype(float)


def get_noise(spectrum, seed=42):
    noise_adder = channelGenericNoiseAdder()
    noise_adder.begin(seed=seed)
    return noise_adder.bandlimited_noise_from_spectrum(
        N_SAMPLES, SAMPLING_RATE, spectrum, amplitude=10 * units.mV, type="rayleigh")


def test_callable_spectrum():
    frequencies = fft.freqs(N_SAMPLES, SAMPLING_RATE)

    trace_callable = get_noise(in_band)
    trace_array = get_noise(in_band(frequencies))
    np.testing.assert_allclose(trace_callable, trace_array, rtol=1e-12, atol=0)

    spectrum = np.abs(fft.time2freq(trace_callable, SAMPLING_RATE))
    occupied = frequencies[spectrum > 1e-6 * spectrum.max()]  # ignore FFT round-off
    assert occupied.min() > F_MIN and occupied.max() < F_MAX, \
        f"noise power between {occupied.min() / units.MHz:.0f} and {occupied.max() / units.MHz:.0f} MHz, " \
        f"expected within {F_MIN / units.MHz:.0f}-{F_MAX / units.MHz:.0f} MHz"


if __name__ == "__main__":
    test_callable_spectrum()
    print("OK")
