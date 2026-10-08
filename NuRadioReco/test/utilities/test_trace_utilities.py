"""
Reference-value test for the trace quantities of ``trace_utilities`` and ``channelSignalReconstructor``.

The reference values were obtained with the scipy-based implementations (``scipy.stats.kurtosis`` / ``entropy``)
and fixed-seed synthetic traces (noise + a short wavelet). Traces with and without a length divisible by 4 are used,
as the noise RMS is estimated from 4 segments.
"""
import logging

import numpy as np

from NuRadioReco.framework import channel, event, station
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.modules.channelSignalReconstructor import channelSignalReconstructor
from NuRadioReco.utilities import trace_utilities, units

SAMPLING_RATE = 3.2 * units.GHz
RTOL = 1e-9


def make_trace(n_samples, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / 3.2
    pulse = 30 * np.exp(-0.5 * ((t - 0.4 * n_samples / 3.2) / 3) ** 2) * np.sin(2 * np.pi * 0.25 * t)
    return rng.normal(0, 10, n_samples) + pulse


def check(name, value, expected):
    assert np.isclose(value, expected, rtol=RTOL, atol=0), f"{name}: {value!r} != {expected!r}"


# n_samples: split_rms, impulsivity, entropy, kurtosis, root_power_ratio
TRACE_REFERENCE = {
    2048: (9.898692049030199, 0.038980919328958974, 4.837476760695864, 0.06481952970242189, 1.3878072073209522),
    2047: (9.862525689802917, 0.05242199126075442, 4.702943431309963, 0.0728080491479659, 1.5254028984495318),
}

# (n_samples, seed): signal_time, max_amp, max_amp_env, p2p, noise_rms, impulsivity, root_power_ratio, entropy,
# kurtosis, snr integrated_power, snr peak_2_peak_amplitude, snr peak_amplitude, snr split_noise_rms
CHANNEL_REFERENCE = {
    (2048, 1): (255.9375, 37.736937534776175, 40.512531868031864, 75.25660244428872, 9.566571484540564,
                0.06454505709044245, 1.542855296939273, 4.800124239134754, 0.13625800862038107,
                0.5736877187569194, 3.7738074736655345, 3.784702797536285, 3.7276426030669656),
    (2048, 2): (256.25, 32.30388572132722, 39.61349613896421, 63.82592385609562, 9.846739907073152,
                0.04841741013164791, 1.3724587181981862, 5.017050832426379, 0.021296504079047374,
                0.48006650029660336, 3.213441666251374, 3.252805320696405, 3.2409672875713875),
    (2050, 3): (255.9375, 45.22555915352886, 59.579814337298984, 82.12882019631795, 9.554552919586817,
                0.047921368713744084, 1.710373562529575, 4.657833219450146, 0.34154673421900794,
                0.6445765594790573, 4.185925424160332, 4.610094664214129, 4.297889230795614),
}
STATION_REFERENCE = (45.22555915352886, 8.066199558956638)  # channels_max_amplitude, channels_max_amplitude_norm

SNR_KEYS = ("integrated_power", "peak_2_peak_amplitude", "peak_amplitude", "peak_2_peak_amplitude_split_noise_rms")
CHANNEL_PARAMETERS = (chp.signal_time, chp.maximum_amplitude, chp.maximum_amplitude_envelope, chp.P2P_amplitude,
                      chp.noise_rms, chp.impulsivity, chp.root_power_ratio, chp.entropy, chp.kurtosis)


def test_trace_utilities():
    for n_samples, expected in TRACE_REFERENCE.items():
        trace = make_trace(n_samples, n_samples)
        times = np.arange(n_samples) / 3.2
        rms = trace_utilities.get_split_trace_noise_RMS(trace)
        values = (rms, trace_utilities.get_impulsivity(trace), trace_utilities.get_entropy(trace),
                  trace_utilities.get_kurtosis(trace), trace_utilities.get_root_power_ratio(trace, times, rms))
        for name, value, ref in zip(("split_rms", "impulsivity", "entropy", "kurtosis", "rpr"), values, expected):
            check(f"{n_samples} samples, {name}", value, ref)


def test_channel_signal_reconstructor():
    evt = event.Event(1, 1)
    stn = station.Station(1)
    for channel_id, (n_samples, seed) in enumerate(CHANNEL_REFERENCE):
        ch = channel.Channel(channel_id)
        ch.set_trace(make_trace(n_samples, seed), SAMPLING_RATE)
        stn.add_channel(ch)
    evt.set_station(stn)

    reconstructor = channelSignalReconstructor(log_level=logging.WARNING)
    reconstructor.begin()
    reconstructor.run(evt, stn, None)

    for ch, expected in zip(stn.iter_channels(), CHANNEL_REFERENCE.values()):
        for param, ref in zip(CHANNEL_PARAMETERS, expected[:len(CHANNEL_PARAMETERS)]):
            check(f"channel {ch.get_id()}, {param.name}", ch[param], ref)
        for key, ref in zip(SNR_KEYS, expected[len(CHANNEL_PARAMETERS):]):
            check(f"channel {ch.get_id()}, snr {key}", ch[chp.SNR][key], ref)

    check("channels_max_amplitude", stn[stnp.channels_max_amplitude], STATION_REFERENCE[0])
    check("channels_max_amplitude_norm", stn[stnp.channels_max_amplitude_norm], STATION_REFERENCE[1])


if __name__ == "__main__":
    test_trace_utilities()
    test_channel_signal_reconstructor()
    print("OK")
