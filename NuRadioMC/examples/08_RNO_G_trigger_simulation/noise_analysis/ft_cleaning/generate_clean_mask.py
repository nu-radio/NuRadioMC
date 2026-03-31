"""Generate a clean event mask for the FT noise pool.

Reads per-channel noise RMS from the NPZ produced by extract_ft_rms.py,
flags non-thermal events using a per-channel MAD-sigma cut, and saves a
clean mask NPZ for use with simulate.py (--ft_clean_mask).

An event is flagged if ANY channel exceeds the threshold.

Output: NPZ file with fields:
    runNum:     int32, run number per event
    eventNum:   int32, event number per event
    is_clean:   int8, 1 = clean, 0 = flagged
    station_id: int32

Usage:
    python generate_clean_mask.py \\
        --rms_npz ft_rms_station23.npz
"""

import argparse
import numpy as np

from NuRadioReco.utilities import units


def robust_stats(arr):
    """Return median and MAD-based sigma for an array.

    Args:
        arr: 1D numeric array.

    Returns:
        Tuple of (median, mad_sigma).
    """
    med = np.nanmedian(arr)
    mad = np.nanmedian(np.abs(arr - med))
    sigma = 1.4826 * mad
    return med, sigma


def per_channel_rms_flag(rms_data, channels, threshold):
    """Flag events where any channel's noise RMS exceeds the threshold.

    Args:
        rms_data: Dict mapping channel ID to array of RMS values.
        channels: List of channel IDs to check.
        threshold: MAD-sigma threshold for flagging.

    Returns:
        Tuple of (flag array, per-channel stats list).
    """
    n = len(next(iter(rms_data.values())))
    flag = np.zeros(n, dtype=bool)
    per_ch_stats = []
    for ch in channels:
        vals = rms_data[ch]
        med, sig = robust_stats(vals)
        if sig > 0:
            ch_flag = (vals - med) / sig > threshold
            flag |= ch_flag
            per_ch_stats.append((ch, med, sig, ch_flag.sum()))
    return flag, per_ch_stats


def main():
    """Generate clean mask from pre-extracted RMS data."""
    parser = argparse.ArgumentParser(
        description="Generate FT noise pool clean mask from RMS NPZ.")
    parser.add_argument("--rms_npz", type=str, required=True,
                        help="Path to NPZ from extract_ft_rms.py")
    parser.add_argument("--output", type=str, default=None,
                        help="Output NPZ path "
                             "(default: clean_mask_station{id}.npz)")
    parser.add_argument("--threshold", type=float, default=4.0,
                        help="Per-channel MAD-sigma threshold (default: 4.0)")
    args = parser.parse_args()

    data = np.load(args.rms_npz)
    run_nums = data["runNum"]
    event_nums = data["eventNum"]
    station_id = int(data["station_id"])
    channels = data["channels"]

    if args.output is None:
        args.output = f"clean_mask_station{station_id}.npz"

    rms_data = {}
    for ch in channels:
        key = f"ch{ch}_noise_RMS"
        if key in data:
            rms_data[ch] = data[key]

    flag, per_ch_stats = per_channel_rms_flag(
        rms_data, list(channels), args.threshold)

    print(f"\n  {'Ch':>4} {'Median (mV)':>12} {'MAD sig (mV)':>13} "
          f"{'Flagged':>10}")
    print(f"  {'-'*43}")
    for ch, med, sig, n in per_ch_stats:
        print(f"  {ch:>4} {med/units.mV:>12.3f} {sig/units.mV:>13.3f} "
              f"{n:>10,}")

    n_clean = (~flag).sum()
    n_total = len(run_nums)
    print(f"\nTotal flagged: {flag.sum():,} "
          f"({100 * flag.mean():.2f}%)")
    print(f"Clean: {n_clean:,} ({100 * n_clean / n_total:.2f}%)")

    np.savez_compressed(
        args.output,
        runNum=run_nums,
        eventNum=event_nums,
        is_clean=(~flag).astype(np.int8),
        station_id=np.int32(station_id),
    )
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
