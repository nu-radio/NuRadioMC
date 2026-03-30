"""Compute per-channel noise RMS distribution statistics from simulated thermal noise.

Loads a NUR file containing pure thermal noise events passed through the
NuRadioMC signal chain and reports the per-channel RMS distribution shape
(skewness, kurtosis, tail counts). These values serve as the Gaussian
reference baseline for calibrating the FT noise cleaning threshold.

Usage:
    python analyze_sim_noise.py --input nur_sim_noise/noise_season2023_st23_1000events.nur
    python analyze_sim_noise.py --input /path/to/noise.nur --station 23 --max_events 500
"""

import argparse
import numpy as np
from scipy import stats as spstats
from NuRadioReco.modules.io import eventReader


def load_sim_noise(nur_path, station_id, max_events=None):
    """Load per-channel RMS values from a simulated noise NUR file.

    Args:
        nur_path: Path to the NUR file.
        station_id: Station ID to read.
        max_events: Maximum number of events to read. None for all.

    Returns:
        Dict mapping channel ID to array of per-event RMS values.
    """
    reader = eventReader.eventReader()
    reader.begin(nur_path)

    rms_data = {}
    n = 0
    for evt in reader.run():
        stn = evt.get_station(station_id)
        if stn is None:
            continue
        n += 1
        for ch in stn.iter_channels():
            ch_id = ch.get_id()
            rms = np.std(ch.get_trace())
            rms_data.setdefault(ch_id, []).append(rms)
        if max_events and n >= max_events:
            break

    for ch_id in rms_data:
        rms_data[ch_id] = np.array(rms_data[ch_id])

    return rms_data, n


def print_stats(rms_data, n_events):
    """Print per-channel distribution statistics."""
    print(f"{'Ch':>4} {'N':>5} {'Median RMS':>12} {'MAD sigma':>12} "
          f"{'Skew':>8} {'Kurt':>8} | "
          f"{'Obs>3sig':>9} {'Exp':>6} {'Ratio':>7}")
    print('-' * 95)

    all_skew = []
    all_kurt = []

    for ch_id in sorted(rms_data.keys()):
        vals = rms_data[ch_id]
        med = np.median(vals)
        mad = 1.4826 * np.median(np.abs(vals - med))
        sk = spstats.skew(vals)
        ku = spstats.kurtosis(vals)
        all_skew.append(sk)
        all_kurt.append(ku)

        if mad > 0:
            zscores = (vals - med) / mad
            obs_3sig = int(np.sum(zscores > 3))
        else:
            obs_3sig = 0
        exp_3sig = len(vals) * (1 - spstats.norm.cdf(3))
        ratio = obs_3sig / exp_3sig if exp_3sig > 0 else 0

        print(f"{ch_id:>4} {len(vals):>5} {med:>12.6f} {mad:>12.6f} "
              f"{sk:>8.3f} {ku:>8.3f} | "
              f"{obs_3sig:>9} {exp_3sig:>6.1f} {ratio:>6.1f}x")

    print()
    print(f"Summary across {len(rms_data)} channels:")
    print(f"  Skewness range:  [{min(all_skew):.3f}, {max(all_skew):.3f}]")
    print(f"  Kurtosis range:  [{min(all_kurt):.3f}, {max(all_kurt):.3f}]")
    print(f"  Events: {n_events}")


def main():
    """Load sim noise and print per-channel RMS distribution stats."""
    parser = argparse.ArgumentParser(
        description="Analyze simulated thermal noise RMS distributions.")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to NUR file with simulated noise events")
    parser.add_argument("--station", type=int, default=23,
                        help="Station ID (default: 23)")
    parser.add_argument("--max_events", type=int, default=None,
                        help="Max events to read (default: all)")
    args = parser.parse_args()

    print(f"Loading simulated noise from {args.input}...")
    rms_data, n_events = load_sim_noise(args.input, args.station, args.max_events)
    print(f"Loaded {n_events} events, {len(rms_data)} channels\n")
    print_stats(rms_data, n_events)


if __name__ == "__main__":
    main()
