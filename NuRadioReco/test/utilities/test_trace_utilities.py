"""
Reference-value test for the trace quantities of ``trace_utilities``.

The reference values were obtained with the scipy-based implementations (``scipy.stats.kurtosis`` / ``entropy``)
and fixed-seed synthetic traces (noise + a short wavelet). Traces with and without a length divisible by 4 are used,
as the noise RMS is estimated from 4 segments.
The ``channelSignalReconstructor`` is tested via the parameters in the NuRadioMC single event tests.

Run with ``--generate`` to print the reference dictionaries for the current implementation.
"""
import pprint
import sys

import numpy as np

from NuRadioReco.utilities import trace_utilities

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


if __name__ == "__main__":
    if "--generate" in sys.argv:
        print(f"TRACE_REFERENCE = {pprint.pformat({n: compute_trace_values(n) for n in (2048, 2047)}, sort_dicts=False)}")
    else:
        test_trace_utilities()
        print("OK")
