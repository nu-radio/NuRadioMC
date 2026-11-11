"""Synthetic tests for the SNR window given as a duration (``snr_window_ns``).

The reconstruction resamples traces to 10 GHz before it computes channel SNRs,
while feature extraction stays at the native 3.2 GHz. A window of 3 samples
covers 0.3 ns at 10 GHz and 0.94 ns at 3.2 GHz, so the same waveform reads
about a third lower in the reconstruction. A window given as a duration must
give the same SNR at both rates, and leaving it unset must keep the 3-sample
result exactly.
"""
import numpy as np
import pytest
import scipy.signal

import NuRadioReco.utilities.trace_utilities as tu
from NuRadioReco.modules.interferometricDirectionReconstruction3D import InterferometricReco3D
from NuRadioReco.utilities import units

SR_NATIVE = 3.2 * units.GHz
SR_RECO = 10 * units.GHz
N = 2048


def _event(seed, amplitude=10.0):
    """Band-limited noise plus a short in-band pulse at 3.2 GHz, and its FFT resampling to 10 GHz."""
    rng = np.random.default_rng(seed)
    t = np.arange(N) / SR_NATIVE
    spectrum = np.fft.rfft(rng.normal(size=N))
    freqs = np.fft.rfftfreq(N, d=1 / SR_NATIVE)
    spectrum[(freqs < 0.1 * units.GHz) | (freqs > 0.6 * units.GHz)] = 0
    noise = np.fft.irfft(spectrum, n=N)
    noise /= np.std(noise)
    t0 = 200 * units.ns
    pulse = amplitude * np.exp(-0.5 * ((t - t0) / (2 * units.ns)) ** 2) \
        * np.sin(2 * np.pi * 0.3 * units.GHz * (t - t0))
    native = noise + pulse
    reco = scipy.signal.resample(native, int(N * SR_RECO / SR_NATIVE))
    return native, reco


def _snr(trace, window_size):
    """SNR of a trace over a window of ``window_size`` samples, with the split-trace noise RMS."""
    return tu.get_signal_to_noise_ratio(trace, tu.get_split_trace_noise_RMS(trace), window_size)


def test_unset_window_keeps_three_samples():
    """Without snr_window_ns the window is 3 samples and the SNR equals the historical call."""
    native, reco = _event(0)
    assert tu.get_snr_window_size(None, SR_NATIVE) == 3
    assert tu.get_snr_window_size(None, None) == 3
    for trace in (native, reco):
        noise_rms = tu.get_split_trace_noise_RMS(trace)
        assert _snr(trace, tu.get_snr_window_size(None, SR_NATIVE)) == \
            tu.get_signal_to_noise_ratio(trace, noise_rms)


def test_window_conversion():
    """Durations convert to rounded sample counts, at least 2."""
    assert tu.get_snr_window_size(20, SR_NATIVE) == 64
    assert tu.get_snr_window_size(20, SR_RECO) == 200
    assert tu.get_snr_window_size(0.9375, SR_NATIVE) == 3
    assert tu.get_snr_window_size(0.1, SR_NATIVE) == 2


@pytest.mark.parametrize("window", [0, -1, np.nan, np.inf])
def test_invalid_window_raises(window):
    """A window that is not a positive finite duration is refused."""
    with pytest.raises(ValueError):
        tu.get_snr_window_size(window, SR_NATIVE)


@pytest.mark.parametrize("rate", [0, -1, np.nan, np.inf])
def test_invalid_sampling_rate_raises(rate):
    """A sampling rate that is not positive and finite is refused."""
    with pytest.raises(ValueError):
        tu.get_snr_window_size(20, rate)


@pytest.mark.parametrize("window_ns", [5.0, 20.0])
def test_duration_window_is_rate_independent(window_ns):
    """The same event reads the same SNR at 3.2 and 10 GHz when the window is a duration.

    Windows of about 1 ns hold only 3 samples at 3.2 GHz, which miss the true extrema of a
    300 MHz pulse, so the native SNR reads about a quarter lower there; windows of a few ns
    and longer agree within a few percent.
    """
    for seed in range(5):
        native, reco = _event(seed)
        s_native = _snr(native, tu.get_snr_window_size(window_ns, SR_NATIVE))
        s_reco = _snr(reco, tu.get_snr_window_size(window_ns, SR_RECO))
        assert s_reco == pytest.approx(s_native, rel=0.05)


def test_three_sample_window_depends_on_rate():
    """The 3-sample window reads far lower after resampling to 10 GHz."""
    native, reco = _event(0)
    assert _snr(reco, 3) < 0.6 * _snr(native, 3)


def test_reco_resolves_window_from_trace_times():
    """The reconstruction converts snr_window_ns at the sampling rate of its traces."""
    times = [np.arange(6400) / SR_RECO]
    assert InterferometricReco3D._resolve_snr_window_size({}, times) == 3
    assert InterferometricReco3D._resolve_snr_window_size({'snr_window_ns': None}, times) == 3
    assert InterferometricReco3D._resolve_snr_window_size({'snr_window_ns': 20.0}, times) == 200
    with pytest.raises(ValueError):
        InterferometricReco3D._resolve_snr_window_size({'snr_window_ns': 20.0}, [])
