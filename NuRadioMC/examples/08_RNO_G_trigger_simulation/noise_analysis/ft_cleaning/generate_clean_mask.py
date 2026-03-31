"""Generate a clean event mask for the FT noise pool.

Reads FT events directly from ROOT files, computes per-channel noise RMS,
and flags non-thermal events using a 4-sigma MAD cut. An event is flagged
if ANY deep channel exceeds the threshold.

Output: NPZ file with fields:
    runNum:     int32, run number per event
    eventNum:   int32, event number per event
    is_clean:   int8, 1 = clean, 0 = flagged
    station_id: int32

Usage:
    python generate_clean_mask.py \\
        --ft_noise_dir /path/to/forced_triggers/station23 \\
        --station_id 23
"""

import argparse
import glob
import logging
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from NuRadioReco.utilities import units

logger = logging.getLogger("generate_clean_mask")
logging.basicConfig(level=logging.INFO)

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


def extract_noise_rms_from_file(fpath, station_id, channels):
    """Read one ROOT file and compute per-channel noise RMS for each FT event.

    Args:
        fpath: Path to a ROOT file.
        station_id: Station ID to select.
        channels: List of channel IDs.

    Returns:
        List of dicts with runNum, eventNum, and ch{N}_noise_RMS per event.
        Returns empty list if the file can't be read.
    """
    from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData

    try:
        reader = readRNOGData()
        reader.begin(
            [fpath],
            convert_to_voltage=True,
            apply_baseline_correction="median",
            selectors=[lambda einfo: einfo.triggerType == "FORCE"],
            select_runs=False,
        )
    except Exception:
        return []

    rows = []
    for evt in reader.run():
        sid = evt.get_station_ids()[0]
        if sid != station_id:
            continue
        station = evt.get_station(sid)
        row = {
            "runNum": evt.get_run_number(),
            "eventNum": evt.get_id(),
        }
        for ch_id in channels:
            if station.has_channel(ch_id):
                trace = station.get_channel(ch_id).get_trace()
                row[f"ch{ch_id}_noise_RMS"] = np.std(trace)
            else:
                row[f"ch{ch_id}_noise_RMS"] = np.nan
        rows.append(row)

    return rows


def per_channel_rms_flag(run_nums, event_nums, rms_data, channels, threshold):
    """Flag events where any channel's noise RMS exceeds the threshold.

    Args:
        run_nums: Array of run numbers.
        event_nums: Array of event numbers.
        rms_data: Dict mapping channel ID to array of RMS values.
        channels: List of channel IDs to check.
        threshold: MAD-sigma threshold for flagging.

    Returns:
        Tuple of (flag array, per-channel stats list).
    """
    n = len(run_nums)
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
    """Generate clean mask from FT ROOT files."""
    parser = argparse.ArgumentParser(
        description="Generate FT noise pool clean mask from ROOT files.")
    parser.add_argument("--ft_noise_dir", type=str, required=True,
                        help="Directory with FT ROOT files")
    parser.add_argument("--station_id", type=int, required=True)
    parser.add_argument("--output", type=str, default=None,
                        help="Output NPZ path (default: clean_mask_station{id}.npz)")
    parser.add_argument("--threshold", type=float, default=4.0,
                        help="Per-channel MAD-sigma threshold (default: 4.0)")
    parser.add_argument("--channels", type=int, nargs="+", default=DEFAULT_CHS,
                        help=f"Channels to check (default: {DEFAULT_CHS})")
    parser.add_argument("--n_workers", type=int, default=None,
                        help="Max parallel workers (default: all available CPUs)")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"clean_mask_station{args.station_id}.npz"

    n_workers = args.n_workers or int(os.environ.get("SLURM_CPUS_PER_TASK",
                                                      os.cpu_count() or 4))

    # Discover FT files
    ft_files = sorted(glob.glob(os.path.join(
        args.ft_noise_dir, f"station{args.station_id}_run*.root")))
    if not ft_files:
        ft_files = sorted(glob.glob(os.path.join(
            args.ft_noise_dir, "run*/waveforms.root")))
    if not ft_files:
        raise FileNotFoundError(f"No FT ROOT files in {args.ft_noise_dir}")
    logger.info(f"Found {len(ft_files)} ROOT files, using {n_workers} workers")

    # Process files in parallel
    all_rows = []
    n_failed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(extract_noise_rms_from_file, f,
                               args.station_id, args.channels): f
                   for f in ft_files}
        for i, future in enumerate(as_completed(futures)):
            rows = future.result()
            if rows:
                all_rows.extend(rows)
            else:
                n_failed += 1
            if (i + 1) % 50 == 0 or (i + 1) == len(ft_files):
                logger.info(f"  {i+1}/{len(ft_files)} files processed, "
                            f"{len(all_rows)} events so far")

    if n_failed:
        logger.warning(f"Skipped {n_failed} files that couldn't be read")
    logger.info(f"Total: {len(all_rows)} FT events from {len(ft_files) - n_failed} files")

    # Build arrays
    run_nums = np.array([r["runNum"] for r in all_rows], dtype=np.int32)
    event_nums = np.array([r["eventNum"] for r in all_rows], dtype=np.int32)
    rms_data = {}
    for ch in args.channels:
        col = f"ch{ch}_noise_RMS"
        rms_data[ch] = np.array([r[col] for r in all_rows])

    # Apply cut
    flag, per_ch_stats = per_channel_rms_flag(
        run_nums, event_nums, rms_data, args.channels, args.threshold)

    print(f"\n  {'Ch':>4} {'Median (mV)':>12} {'MAD sig (mV)':>13} {'Flagged':>10}")
    print(f"  {'-'*43}")
    for ch, med, sig, n in per_ch_stats:
        print(f"  {ch:>4} {med/units.mV:>12.3f} {sig/units.mV:>13.3f} {n:>10,}")

    n_clean = (~flag).sum()
    print(f"\nTotal flagged: {flag.sum():,} ({100 * flag.mean():.2f}%)")
    print(f"Clean: {n_clean:,} ({100 * n_clean / len(all_rows):.2f}%)")

    np.savez_compressed(
        args.output,
        runNum=run_nums,
        eventNum=event_nums,
        is_clean=(~flag).astype(np.int8),
        station_id=np.int32(args.station_id),
    )
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
