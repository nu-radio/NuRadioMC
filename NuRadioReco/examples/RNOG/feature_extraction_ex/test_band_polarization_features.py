"""Synthetic unit + property tests for the band / polarization features.

No real data: builds traces with known content and checks the gain-invariance
property that motivates the design (band_snr and the normalized
cross-correlation must not change when a channel is rescaled by an arbitrary
electronics gain).
"""
import numpy as np
import NuRadioReco.utilities.trace_utilities as tu

RNG = np.random.default_rng(0)
SR = 3.2  # GHz, RNO-G radiant
N = 2048
T = np.arange(N) / SR  # ns


def _tone(freq_ghz, amp, n=N):
    return amp * np.sin(2 * np.pi * freq_ghz * (np.arange(n) / SR))


def _pulse(freq_ghz, amp, n=N, center=None, width=40.0):
    """Gaussian-windowed in-band sinusoid, localized so the rest of the trace
    is signal-free (the split-trace noise assumption real impulsive RF meets)."""
    if center is None:
        center = n // 5
    t = np.arange(n)
    env = np.exp(-0.5 * ((t - center) / width) ** 2)
    return amp * env * np.sin(2 * np.pi * freq_ghz * (t / SR))


def test_band_power_localizes_in_band():
    in_band = _tone(0.2, 1.0)          # 200 MHz, inside [0.1, 0.3]
    out_band = _tone(0.45, 1.0)        # 450 MHz, outside
    fi = tu.get_band_features(in_band, SR, 0.1, 0.3, fmin=0.08, fmax=0.6)
    fo = tu.get_band_features(out_band, SR, 0.1, 0.3, fmin=0.08, fmax=0.6)
    assert fi["band_power_ratio"] > 0.9
    assert fo["band_power_ratio"] < 0.1
    assert abs(fi["peak_frequency"] - 0.2) < 0.02


def test_band_snr_is_gain_invariant():
    sig = _pulse(0.2, 8.0)
    noise = RNG.standard_normal(N)
    trace = sig + noise
    for gain in (0.01, 1.0, 137.0):
        f = tu.get_band_features(gain * trace, SR, 0.1, 0.3, fmin=0.08, fmax=0.6)
        base = tu.get_band_features(trace, SR, 0.1, 0.3, fmin=0.08, fmax=0.6)
        # band_snr is a ratio of in-band signal power to in-band noise power;
        # multiplying the whole trace by `gain` scales both -> ratio invariant.
        assert np.isclose(f["band_snr"], base["band_snr"], rtol=1e-9)
        assert np.isclose(f["band_power_ratio"], base["band_power_ratio"], rtol=1e-9)


def test_band_snr_increases_with_signal():
    noise = RNG.standard_normal(N)
    weak = tu.get_band_features(_pulse(0.2, 2.0) + noise, SR, 0.1, 0.3)
    strong = tu.get_band_features(_pulse(0.2, 12.0) + noise, SR, 0.1, 0.3)
    assert strong["band_snr"] > weak["band_snr"]


def test_xcorr_identical_and_gain_invariant():
    a = _tone(0.2, 1.0) + 0.1 * RNG.standard_normal(N)
    xc, lag = tu.get_normalized_cross_correlation(a, a)
    assert np.isclose(xc, 1.0, atol=1e-9) and lag == 0
    xc2, lag2 = tu.get_normalized_cross_correlation(a, 50.0 * a)
    assert np.isclose(xc2, 1.0, atol=1e-9) and lag2 == 0  # gain-independent


def test_xcorr_recovers_shift():
    a = _pulse(0.2, 1.0, center=N // 2)  # localized -> unambiguous lag
    shift = 7
    b = np.roll(a, shift)
    xc, lag = tu.get_normalized_cross_correlation(a, b, max_lag=64)
    assert xc > 0.9 and abs(abs(lag) - shift) <= 1


def test_xcorr_low_for_independent_noise():
    a = RNG.standard_normal(N)
    b = RNG.standard_normal(N)
    xc, _ = tu.get_normalized_cross_correlation(a, b, max_lag=64)
    assert xc < 0.3


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all band/polarization unit tests passed")
