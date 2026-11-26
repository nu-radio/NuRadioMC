"""
Test of ``stationCoherentlySummedWaveforms`` on a synthetic station with known signal delays.

Channels 0-3 are summed, channels 20-23 are lag-only channels (weaker signals, larger delays).
Channel 2 has the strongest signal (the reference channel chosen with ``ref_selection="max_snr"``).
"""
import warnings

import numpy as np
import pytest

from NuRadioReco.framework.channel import Channel
from NuRadioReco.framework.event import Event
from NuRadioReco.framework.parameters import stationParametersRNOG as stpRNOG
from NuRadioReco.framework.station import Station
from NuRadioReco.modules.RNO_G.stationCoherentlySummedWaveforms import stationCoherentlySummedWaveforms
from NuRadioReco.utilities import units

SAMPLING_RATE = 3.2 * units.GHz
N_SAMPLES = 2048
# signal time (ns) and amplitude per channel
SIGNALS = {0: (300, 1.), 1: (310, 1.), 2: (295, 2.), 3: (320, 1.),
           20: (450, 0.7), 21: (200, 0.7), 22: (380, 0.7), 23: (250, 0.7)}
SUM_IDS = [0, 1, 2, 3]
LAG_IDS = [20, 21, 22, 23]
TOL = 1.5 / SAMPLING_RATE  # one sample plus rounding of the signal time


def make_event(seed=1, constant={}):
    """ constant: {channel id: value} of channels with a constant trace (no signal read out) """
    rng = np.random.default_rng(seed)
    times = np.arange(N_SAMPLES) / SAMPLING_RATE
    evt = Event(0, 0)
    station = Station(11)
    evt.set_station(station)
    for ch_id, (t0, amp) in SIGNALS.items():
        pulse = amp * np.exp(-0.5 * ((times - t0) / 2) ** 2) * np.sin(2 * np.pi * 0.25 * (times - t0))
        channel = Channel(ch_id)
        trace = pulse + rng.normal(0, 0.05, N_SAMPLES)
        if ch_id in constant:
            trace = np.full(N_SAMPLES, constant[ch_id], dtype=float)
        channel.set_trace(trace, SAMPLING_RATE)
        station.add_channel(channel)
    return evt, station


def run(begin_kwargs={}, run_kwargs={}, constant={}):
    evt, station = make_event(constant=constant)
    module = stationCoherentlySummedWaveforms()
    module.begin(channel_ids=SUM_IDS, **{"time_lag_ref_ch_id": 0, **begin_kwargs})
    module.run(evt, station, None, use_envelope=False, **run_kwargs)
    keys = ["coherent_snr", "coherent_impulsivity", "coherent_entropy", "coherent_kurtosis",
            "coherent_time_lags", "coherent_correlations", "coherent_ref_channel_id"]
    return {key: station.get_parameter(getattr(stpRNOG, key)) for key in keys}


def check_lags(result, ch_ids, fix=0):
    """ Stored lags are T_fix - T_i """
    lags = result["coherent_time_lags"]
    assert lags[fix] == 0
    for ch_id in ch_ids:
        expected = SIGNALS[fix][0] - SIGNALS[ch_id][0]
        assert abs(lags[ch_id] - expected) <= TOL, f"channel {ch_id}: lag {lags[ch_id]} != {expected}"


def test_fixed_reference():
    result = run()
    assert result["coherent_ref_channel_id"] == 0
    assert set(result["coherent_time_lags"]) == set(SUM_IDS)
    check_lags(result, SUM_IDS)


def test_lag_channels_do_not_change_sum():
    reference = run()
    for template in ["sum", "ref_channel"]:
        result = run(dict(lag_channel_ids=LAG_IDS, lag_template=template))
        for key in ["coherent_snr", "coherent_impulsivity", "coherent_entropy", "coherent_kurtosis"]:
            assert result[key] == reference[key], key
        assert set(result["coherent_time_lags"]) == set(SUM_IDS + LAG_IDS)
        check_lags(result, SUM_IDS + LAG_IDS)


def test_crop_to_overlap():
    result = run(dict(lag_channel_ids=LAG_IDS, crop_to_overlap=True))
    check_lags(result, SUM_IDS + LAG_IDS)


def test_max_snr_reference():
    fixed = run(dict(lag_channel_ids=LAG_IDS))
    result = run(dict(lag_channel_ids=LAG_IDS), dict(ref_selection="max_snr"))
    assert result["coherent_ref_channel_id"] == 2
    assert 2 not in result["coherent_correlations"] and 0 in result["coherent_correlations"]
    check_lags(result, SUM_IDS + LAG_IDS)
    for ch_id in SUM_IDS + LAG_IDS:
        assert abs(result["coherent_time_lags"][ch_id] - fixed["coherent_time_lags"][ch_id]) <= 1 / SAMPLING_RATE


def test_time_lag_range_relative_to_fixed_channel():
    # Exclude the true lag of channel 1 (-10 ns) and of the reference channel 2 (+5 ns): the lags must
    # stay in the given ranges (relative to channel 0) also if channel 2 is the reference of the sum.
    ranges = {1: (0 * units.ns, 20 * units.ns), 2: (10 * units.ns, 30 * units.ns), 21: (60 * units.ns, 90 * units.ns)}
    for run_kwargs in [{}, dict(ref_selection="max_snr")]:
        result = run(dict(lag_channel_ids=LAG_IDS, time_lag_range=ranges), run_kwargs)
        for ch_id, (lo, hi) in ranges.items():
            lag = result["coherent_time_lags"][ch_id]
            assert lo - 1e-9 <= lag <= hi + 1e-9, f"{run_kwargs}: channel {ch_id}: lag {lag} not in {(lo, hi)}"


def test_int_reference_channel():
    # The sum is aligned onto channel 2, the lags are still stored relative to channel 0
    fixed = run(dict(lag_channel_ids=LAG_IDS))
    result = run(dict(lag_channel_ids=LAG_IDS), dict(ref_selection=2))
    assert result["coherent_ref_channel_id"] == 2
    check_lags(result, SUM_IDS + LAG_IDS)
    for ch_id in SUM_IDS + LAG_IDS:
        assert abs(result["coherent_time_lags"][ch_id] - fixed["coherent_time_lags"][ch_id]) <= 1 / SAMPLING_RATE


def test_time_lag_ref_defaults_to_sum_reference():
    # As before time_lag_ref_ch_id existed: lags relative to the reference channel of the sum
    result = run(dict(time_lag_ref_ch_id=None), dict(ref_selection=1))
    check_lags(result, SUM_IDS, fix=1)
    with pytest.raises(ValueError):
        run(dict(time_lag_ref_ch_id=None), dict(ref_selection="max_snr"))


def test_deprecated_ref_ch_id():
    expected = run(dict(time_lag_ref_ch_id=None), dict(ref_selection=1))
    with pytest.warns(DeprecationWarning):
        result = run(dict(time_lag_ref_ch_id=None), dict(ref_ch_id=1))
    assert result == expected


def test_constant_channels():
    # Channels 20 and 23 read 0, channel 21 a constant: NaN lag and correlation, the rest unchanged
    expected = run(dict(lag_channel_ids=LAG_IDS))
    for run_kwargs in [{}, dict(ref_selection="max_snr")]:
        result = run(dict(lag_channel_ids=LAG_IDS), run_kwargs, constant={20: 0, 21: 255, 23: 0})
        for ch_id in [20, 21, 23]:
            assert np.isnan(result["coherent_time_lags"][ch_id]) and np.isnan(result["coherent_correlations"][ch_id])
        check_lags(result, SUM_IDS + [22])
        assert result["coherent_snr"] == run({}, run_kwargs)["coherent_snr"]
    assert result["coherent_time_lags"][22] == pytest.approx(expected["coherent_time_lags"][22], abs=1 / SAMPLING_RATE)

    # Constant fixed channel (the reference of the sum here): no lag is defined
    result = run(dict(lag_channel_ids=LAG_IDS), constant={0: 0})
    assert result["coherent_time_lags"][0] == 0
    assert all(np.isnan(lag) for ch_id, lag in result["coherent_time_lags"].items() if ch_id != 0)
    assert all(np.isnan(corr) for corr in result["coherent_correlations"].values())


if __name__ == "__main__":
    warnings.simplefilter("always")
    for name, func in list(globals().items()):
        if name.startswith("test_"):
            func()
            print(f"{name}: ok")
