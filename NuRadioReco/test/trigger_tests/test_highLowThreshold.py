"""
Tests for ``get_high_low_triggers`` and ``get_majority_logic`` in
``NuRadioReco.modules.trigger.highLowThreshold``.

The vectorised (``as_strided``) implementations are compared against a plain python
loop over all complete windows. In particular, the last complete window of a trace
must be evaluated, i.e. the output has one bin per input sample (default padding).
"""
import numpy as np
import pytest

from NuRadioReco.utilities import units
from NuRadioReco.modules.trigger.highLowThreshold import get_high_low_triggers, get_majority_logic


def _windows(trace, window, step, align_strides_to_start):
    """ All complete windows of ``trace`` as the trigger logic sees them (python loop). """
    if not align_strides_to_start:
        trace = np.pad(trace, (window - 1, 0), "constant")
    starts = range(0, len(trace) - window + 1, step)
    return [trace[s:s + window] for s in starts]


def reference_high_low(trace, high, low, window, step=1, align_strides_to_start=False):
    windows = _windows(trace, window, step, align_strides_to_start)
    out = np.array([np.any(w >= high) and np.any(w <= low) for w in windows], dtype=bool)
    return out[:len(trace) // step]


def reference_majority(tts, n_coinc, window, step=1, align_strides_to_start=False):
    window = min(window, len(tts[0]))
    per_channel = np.array([[np.any(w) for w in _windows(np.asarray(tt), window, step, align_strides_to_start)]
                            for tt in tts])
    ttt = np.sum(per_channel, axis=0) >= n_coinc
    bins = np.atleast_1d(np.squeeze(np.argwhere(ttt))) * step
    return np.any(ttt), bins


class TestHighLowTriggers:

    @pytest.mark.parametrize("n_samples", [20, 999, 1000, 2048])
    @pytest.mark.parametrize("window", [1, 2, 5, 13])
    def test_output_length_default_padding(self, n_samples, window):
        trace = np.zeros(n_samples)
        hl = get_high_low_triggers(trace, 1.0, -1.0, window * units.ns, 1 * units.ns)
        assert len(hl) == n_samples

    def test_trigger_in_last_samples(self):
        trace = np.zeros(1000)
        trace[-3] = 3.0
        trace[-1] = -3.0
        hl = get_high_low_triggers(trace, 2.5, -2.5, 5 * units.ns, 1 * units.ns)
        assert len(hl) == 1000
        assert hl[-1]
        # the trigger bin is the sample at which the coincidence completes
        assert np.array_equal(np.flatnonzero(hl), [999])

    def test_trigger_in_first_samples(self):
        trace = np.zeros(100)
        trace[0] = 3.0
        trace[1] = -3.0
        hl = get_high_low_triggers(trace, 2.5, -2.5, 5 * units.ns, 1 * units.ns)
        assert np.array_equal(np.flatnonzero(hl), np.arange(1, 5))

    def test_aligned_strides_last_window(self):
        # 20 samples, window 4, step 4 -> 5 complete windows; crossing only in the last one
        trace = np.zeros(20)
        trace[17] = 3.0
        trace[19] = -3.0
        hl = get_high_low_triggers(trace, 2.5, -2.5, 4 * units.ns, 1 * units.ns,
                                   step=4, align_strides_to_start=True)
        assert len(hl) == 5
        assert np.array_equal(hl, [False, False, False, False, True])

    @pytest.mark.parametrize("step", [1, 2, 3, 4])
    @pytest.mark.parametrize("align", [False, True])
    @pytest.mark.parametrize("window", [1, 4, 5, 8])
    def test_matches_reference_random(self, step, align, window):
        rng = np.random.Generator(np.random.Philox(1234))
        for n_samples in [64, 101, 256]:
            trace = rng.normal(0, 1, n_samples)
            hl = get_high_low_triggers(trace, 2.0, -2.0, window * units.ns, 1 * units.ns,
                                       step=step, align_strides_to_start=align)
            ref = reference_high_low(trace, 2.0, -2.0, window, step, align)
            assert np.array_equal(hl, ref)

    def test_sampling_rate(self):
        # coincidence window is given in time, converted with dt
        trace = np.zeros(200)
        trace[-5] = 3.0
        trace[-1] = -3.0
        dt = 1 / (3.2 * units.GHz)
        hl = get_high_low_triggers(trace, 2.5, -2.5, 5 * dt, dt)
        assert len(hl) == 200
        assert np.array_equal(np.flatnonzero(hl), [199])

    def test_type_mismatch_raises(self):
        with pytest.raises(TypeError):
            get_high_low_triggers(np.zeros(10, dtype=np.float32), 1.0, -1.0)


class TestMajorityLogic:

    def test_last_bin_coincidence_list(self):
        tts = [np.zeros(1000, dtype=bool) for _ in range(2)]
        tts[0][-1] = tts[1][-1] = True
        triggered, bins, times = get_majority_logic(tts, 2, 32 * units.ns, 1 * units.ns)
        assert triggered
        assert np.array_equal(bins, [999])
        assert np.allclose(times, 999 * units.ns)

    def test_2d_array_input(self):
        tts = np.zeros((4, 1000), dtype=bool)
        tts[0, -1] = tts[1, -1] = True
        triggered, bins, _ = get_majority_logic(tts, 2, 32 * units.ns, 1 * units.ns)
        assert triggered
        assert np.array_equal(bins, [999])

    def test_coincidence_window(self):
        def make_tts():
            tts = [np.zeros(200, dtype=bool) for _ in range(3)]
            tts[0][100] = True
            tts[1][131] = True  # within 32 bins of channel 0
            tts[2][150] = True
            return tts

        triggered, bins, _ = get_majority_logic(make_tts(), 2, 32 * units.ns, 1 * units.ns)
        assert triggered
        # channel 0 is extended to bins 100..131, channel 1 to 131..162, channel 2 to 150..181:
        # channels 0 + 1 overlap in bin 131, channels 1 + 2 in bins 150..162
        assert np.array_equal(bins, np.concatenate([[131], np.arange(150, 163)]))

        triggered, _, _ = get_majority_logic(make_tts(), 3, 32 * units.ns, 1 * units.ns)
        assert not triggered

    def test_no_trigger(self):
        tts = np.zeros((4, 300), dtype=bool)
        tts[0, 10] = True
        triggered, bins, _ = get_majority_logic(tts, 2, 32 * units.ns, 1 * units.ns)
        assert not triggered
        assert len(bins) == 0

    def test_window_longer_than_trace(self):
        tts = [np.zeros(10, dtype=bool) for _ in range(2)]
        tts[0][0] = True
        tts[1][9] = True
        triggered, bins, _ = get_majority_logic(tts, 2, 32 * units.ns, 1 * units.ns)
        assert triggered
        assert np.array_equal(bins, [9])

    def test_output_length(self):
        tts = np.zeros((2, 500), dtype=bool)
        get_majority_logic(tts, 2, 32 * units.ns, 1 * units.ns)
        # get_majority_logic writes the per-channel coincidence back into its input
        assert tts.shape == (2, 500)

    @pytest.mark.parametrize("step", [1, 2, 4])
    @pytest.mark.parametrize("align", [False, True])
    @pytest.mark.parametrize("window", [1, 8, 32])
    def test_matches_reference_random(self, step, align, window):
        rng = np.random.Generator(np.random.Philox(42))
        for n_samples in [64, 129, 512]:
            tts = rng.random((3, n_samples)) < 0.01
            ref_triggered, ref_bins = reference_majority(tts, 2, window, step, align)
            triggered, bins, times = get_majority_logic([t.copy() for t in tts], 2, window * units.ns, 1 * units.ns,
                                                        step=step, align_strides_to_start=align)
            assert triggered == ref_triggered
            assert np.array_equal(bins, ref_bins)
            assert np.allclose(times, bins * units.ns)

    def test_chain_high_low_majority(self):
        # single-channel triggers from get_high_low_triggers feed into the majority logic
        n = 1000
        traces = np.zeros((4, n))
        for i in (1, 3):
            traces[i, -4] = 3.0
            traces[i, -2] = -3.0
        tts = [get_high_low_triggers(tr, 2.5, -2.5, 5 * units.ns, 1 * units.ns) for tr in traces]
        assert all(len(t) == n for t in tts)
        triggered, bins, _ = get_majority_logic(tts, 2, 32 * units.ns, 1 * units.ns)
        assert triggered
        assert np.array_equal(bins, [n - 2, n - 1])


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
