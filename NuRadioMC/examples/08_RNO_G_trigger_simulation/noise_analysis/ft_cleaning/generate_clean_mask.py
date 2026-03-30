"""Generate a clean event mask for the FT noise pool.

Flags forced-trigger events with non-thermal noise using a per-channel
4-sigma MAD cut on noise_RMS. For each of 15 channels independently,
events where noise_RMS exceeds median + 4 * MAD_sigma are flagged. An
event is removed from the clean pool if ANY channel exceeds the threshold.

The 4-sigma threshold is calibrated against simulated thermal noise passed
through the NuRadioMC signal chain (see validate_threshold.py). It is the
value where the post-cut kurtosis converges to the simulated thermal
reference. Tighter cuts sculpt the thermal distribution itself; looser
cuts leave measurable non-Gaussian tails.

Output: NPZ file with fields:
    runNum:     int32, run number per event
    eventNum:   int32, event number per event
    is_clean:   int8, 1 = clean, 0 = flagged

Usage:
    python generate_clean_mask.py \\
        --feature_file /path/to/merged_feature_output.h5 \\
        --output clean_mask_station23.npz

    python generate_clean_mask.py \\
        --feature_file /path/to/merged_feature_output.h5 \\
        --output clean_mask_station23.npz \\
        --threshold 4.0 \\
        --channels 0 1 2 3 4 5 6 7 8 9 10 11 21 22 23
"""

import argparse
import numpy as np
import pandas as pd


DEFAULT_CHS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23]


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


def per_channel_rms_flag(df, channels, threshold):
    """Flag events where any channel's noise_RMS exceeds the threshold.

    Args:
        df: DataFrame with ch{N}_noise_RMS columns.
        channels: List of channel IDs to check.
        threshold: MAD-sigma threshold for flagging.

    Returns:
        Boolean array, True = flagged.
    """
    flag = np.zeros(len(df), dtype=bool)
    per_ch_stats = []
    for ch in channels:
        col = f"ch{ch}_noise_RMS"
        if col not in df.columns:
            continue
        vals = df[col].values
        med, sig = robust_stats(vals)
        if sig > 0:
            ch_flag = (vals - med) / sig > threshold
            n_flagged = ch_flag.sum()
            flag |= ch_flag
            per_ch_stats.append((ch, med, sig, n_flagged))

    return flag, per_ch_stats


def main():
    """Generate clean mask from FT feature data."""
    parser = argparse.ArgumentParser(
        description="Generate FT noise pool clean mask.")
    parser.add_argument("--feature_file", type=str, required=True,
                        help="Path to merged feature HDF5")
    parser.add_argument("--station_id", type=int, required=True,
                        help="Station ID (used in default output filename)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output NPZ path (default: clean_mask_station{id}.npz)")
    parser.add_argument("--threshold", type=float, default=4.0,
                        help="Per-channel MAD-sigma threshold (default: 4.0)")
    parser.add_argument("--channels", type=int, nargs="+", default=DEFAULT_CHS,
                        help="Channels to apply per-channel RMS cut "
                             f"(default: {DEFAULT_CHS})")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"clean_mask_station{args.station_id}.npz"

    print(f"Loading features from {args.feature_file}...")
    df = pd.read_hdf(args.feature_file, key="data")
    print(f"  {len(df):,} events, {len(df.columns)} columns")

    print(f"\nPer-channel RMS cut at {args.threshold}-sigma "
          f"on {len(args.channels)} channels...")
    flag, per_ch_stats = per_channel_rms_flag(df, args.channels, args.threshold)

    print(f"\n  {'Ch':>4} {'Median (mV)':>12} {'MAD sig (mV)':>13} {'Flagged':>10}")
    print(f"  {'-'*43}")
    for ch, med, sig, n in per_ch_stats:
        print(f"  {ch:>4} {med*1e3:>12.3f} {sig*1e3:>13.3f} {n:>10,}")

    n_clean = (~flag).sum()
    print(f"\nTotal flagged: {flag.sum():,} ({100 * flag.mean():.2f}%)")
    print(f"Clean: {n_clean:,} ({100 * n_clean / len(df):.2f}%)")

    np.savez_compressed(
        args.output,
        runNum=df["runNum"].values.astype(np.int32),
        eventNum=df["eventNum"].values.astype(np.int32),
        is_clean=(~flag).astype(np.int8),
    )
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
