"""
Reference-value test for the trace quantities of ``trace_utilities`` and ``channelSignalReconstructor``.

The reference values were obtained with the scipy-based implementations (``scipy.stats.kurtosis`` / ``entropy``)
and fixed-seed synthetic traces (noise + a short wavelet). Traces with and without a length divisible by 4 are used,
as the noise RMS is estimated from 4 segments.

Run with ``--generate`` to print the reference dictionaries for the current implementation.
"""
import logging
import pprint
import sys

import numpy as np

from NuRadioReco.framework import channel, event, station
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.modules.channelSignalReconstructor import channelSignalReconstructor
from NuRadioReco.utilities import trace_utilities, units

SAMPLING_RATE = 3.2 * units.GHz
RTOL = 1e-9

# trace_utilities functions: {n_samples: {quantity: value}}
TRACE_REFERENCE = {2048: {'split_noise_rms': 9.898692049030199,
        'impulsivity': 0.06342103193285986,
        'entropy': 4.741095212558004,
        'kurtosis': 0.2843839755880788,
        'root_power_ratio': 1.6301143531420976},
 2047: {'split_noise_rms': 9.862525689802917,
        'impulsivity': 0.0783866824963344,
        'entropy': 4.446984496983265,
        'kurtosis': 0.39681458829264793,
        'root_power_ratio': 1.764072041408105}}

# channelSignalReconstructor: {(n_samples, seed): {channelParameters name: value}}, the SNR entry is a dict itself
CHANNEL_REFERENCE = {(2048, 1): {'signal_time': 255.9375,
             'maximum_amplitude': 47.19653222384383,
             'maximum_amplitude_envelope': 50.50799683423189,
             'P2P_amplitude': 86.22295624185294,
             'noise_rms': 9.566571484540564,
             'impulsivity': 0.09061484588673685,
             'root_power_ratio': 1.7990358722755064,
             'entropy': 4.615777885456574,
             'kurtosis': 0.4156402362795646,
             'SNR': {'integrated_power': 0.7023584797174502,
                     'peak_2_peak_amplitude': 4.323724777608998,
                     'peak_amplitude': 4.733421925851372,
                     'peak_2_peak_amplitude_split_noise_rms': 4.506471120881078}},
 (2048, 2): {'signal_time': 256.25,
             'maximum_amplitude': 41.76348041039488,
             'maximum_amplitude_envelope': 49.570696670206395,
             'P2P_amplitude': 80.88092963238556,
             'noise_rms': 9.846739907073152,
             'impulsivity': 0.07187782483949379,
             'root_power_ratio': 1.6055523497790265,
             'entropy': 4.698914100744641,
             'kurtosis': 0.279999838678342,
             'SNR': {'integrated_power': 0.61020913294339,
                     'peak_2_peak_amplitude': 4.0721094750128115,
                     'peak_amplitude': 4.205329119278197,
                     'peak_2_peak_amplitude_split_noise_rms': 4.10699025239241}},
 (2050, 3): {'signal_time': 255.9375,
             'maximum_amplitude': 54.822860818960635,
             'maximum_amplitude_envelope': 69.3039506748754,
             'P2P_amplitude': 100.89467541907001,
             'noise_rms': 9.554552919586817,
             'impulsivity': 0.07956597359373063,
             'root_power_ratio': 1.9827991984444633,
             'entropy': 4.3760135359807935,
             'kurtosis': 0.8913525808426734,
             'SNR': {'integrated_power': 0.7844579970312248,
                     'peak_2_peak_amplitude': 5.142379812464713,
                     'peak_amplitude': 5.588401401085243,
                     'peak_2_peak_amplitude_split_noise_rms': 5.279926557957312}}}

# {stationParameters name: value}
STATION_REFERENCE = {'channels_max_amplitude': 54.822860818960635,
 'channels_max_amplitude_norm': 9.741411475729404}

CHANNEL_PARAMETERS = [chp.signal_time, chp.maximum_amplitude, chp.maximum_amplitude_envelope, chp.P2P_amplitude,
                      chp.noise_rms, chp.impulsivity, chp.root_power_ratio, chp.entropy, chp.kurtosis]
STATION_PARAMETERS = [stnp.channels_max_amplitude, stnp.channels_max_amplitude_norm]


def make_trace(n_samples, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / 3.2
    pulse = 40 * np.exp(-0.5 * ((t - 0.4 * n_samples / 3.2) / 3) ** 2) * np.sin(2 * np.pi * 0.25 * t)
    return rng.normal(0, 10, n_samples) + pulse


def compute_trace_values(n_samples):
    trace = make_trace(n_samples, n_samples)
    times = np.arange(n_samples) / 3.2
    rms = trace_utilities.get_split_trace_noise_RMS(trace)
    return {key: float(value) for key, value in {
        "split_noise_rms": rms,
        "impulsivity": trace_utilities.get_impulsivity(trace),
        "entropy": trace_utilities.get_entropy(trace),
        "kurtosis": trace_utilities.get_kurtosis(trace),
        "root_power_ratio": trace_utilities.get_root_power_ratio(trace, times, rms),
    }.items()}


def run_reconstructor(channel_keys):
    evt = event.Event(1, 1)
    stn = station.Station(1)
    for channel_id, (n_samples, seed) in enumerate(channel_keys):
        ch = channel.Channel(channel_id)
        ch.set_trace(make_trace(n_samples, seed), SAMPLING_RATE)
        stn.add_channel(ch)
    evt.set_station(stn)

    reconstructor = channelSignalReconstructor(log_level=logging.WARNING)
    reconstructor.begin()
    reconstructor.run(evt, stn, None)
    return stn


def compute_channel_values(channel_keys):
    stn = run_reconstructor(channel_keys)
    channel_values = {}
    for key, ch in zip(channel_keys, stn.iter_channels()):
        values = {param.name: float(ch[param]) for param in CHANNEL_PARAMETERS}
        values[chp.SNR.name] = {k: float(v) for k, v in ch[chp.SNR].items()}
        channel_values[key] = values
    station_values = {param.name: float(stn[param]) for param in STATION_PARAMETERS}
    return channel_values, station_values


def check(name, value, expected):
    assert np.isclose(value, expected, rtol=RTOL, atol=0), f"{name}: {value!r} != {expected!r}"


def check_nested(name, values, reference):
    assert values.keys() == reference.keys(), f"{name}: keys {sorted(values)} != {sorted(reference)}"
    for key, ref in reference.items():
        if isinstance(ref, dict):
            check_nested(f"{name}, {key}", values[key], ref)
        else:
            check(f"{name}, {key}", values[key], ref)


def test_trace_utilities():
    for n_samples, reference in TRACE_REFERENCE.items():
        check_nested(f"{n_samples} samples", compute_trace_values(n_samples), reference)


def test_channel_signal_reconstructor():
    channel_values, station_values = compute_channel_values(list(CHANNEL_REFERENCE))
    for key, reference in CHANNEL_REFERENCE.items():
        check_nested(f"channel {key}", channel_values[key], reference)
    check_nested("station", station_values, STATION_REFERENCE)


if __name__ == "__main__":
    if "--generate" in sys.argv:
        channel_keys = [(2048, 1), (2048, 2), (2050, 3)]
        channel_values, station_values = compute_channel_values(channel_keys)
        for name, values in (("TRACE_REFERENCE", {n: compute_trace_values(n) for n in (2048, 2047)}),
                             ("CHANNEL_REFERENCE", channel_values), ("STATION_REFERENCE", station_values)):
            print(f"{name} = {pprint.pformat(values, sort_dicts=False)}\n")
    else:
        test_trace_utilities()
        test_channel_signal_reconstructor()
        print("OK")
