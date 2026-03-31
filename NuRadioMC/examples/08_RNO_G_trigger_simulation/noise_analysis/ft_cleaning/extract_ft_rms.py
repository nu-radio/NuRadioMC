"""Extract per-channel noise RMS from FT ROOT files.

Reads forced-trigger events from ROOT files via readRNOGDataMattak,
computes per-channel waveform RMS, and saves the result as an NPZ.
This is the slow step (reads all ROOT files); downstream scripts
(generate_clean_mask.py, validate_threshold.py)
read the NPZ output.

Output: NPZ file with fields:
    runNum:              int32, run number per event
    eventNum:            int32, event number per event
    station_id:          int32
    channels:            int32 array of channel IDs
    ch{N}_noise_RMS:     float32, per-channel noise RMS per event

Usage:
    python extract_ft_rms.py \\
        --ft_noise_dir /path/to/forced_triggers/station23 \\
        --station_id 23
"""

import argparse
import glob
import logging
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger("extract_ft_rms")
logging.basicConfig(level=logging.INFO)

DEFAULT_CHS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23]


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


def main():
    """Extract per-channel noise RMS from FT ROOT files."""
    parser = argparse.ArgumentParser(
        description="Extract per-channel noise RMS from FT ROOT files.")
    parser.add_argument("--ft_noise_dir", type=str, required=True,
                        help="Directory with FT ROOT files")
    parser.add_argument("--station_id", type=int, required=True)
    parser.add_argument("--output", type=str, default=None,
                        help="Output NPZ path "
                             "(default: ft_rms_station{id}.npz)")
    parser.add_argument("--channels", type=int, nargs="+",
                        default=DEFAULT_CHS,
                        help=f"Channels to extract (default: {DEFAULT_CHS})")
    parser.add_argument("--n_workers", type=int, default=None,
                        help="Max parallel workers "
                             "(default: all available CPUs)")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"ft_rms_station{args.station_id}.npz"

    n_workers = args.n_workers or int(os.environ.get(
        "SLURM_CPUS_PER_TASK", os.cpu_count() or 4))

    ft_files = sorted(glob.glob(os.path.join(
        args.ft_noise_dir, f"station{args.station_id}_run*.root")))
    if not ft_files:
        ft_files = sorted(glob.glob(os.path.join(
            args.ft_noise_dir, "run*/waveforms.root")))
    if not ft_files:
        raise FileNotFoundError(
            f"No FT ROOT files in {args.ft_noise_dir}")
    logger.info(f"Found {len(ft_files)} ROOT files, "
                f"using {n_workers} workers")

    all_rows = []
    n_failed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(extract_noise_rms_from_file, f,
                        args.station_id, args.channels): f
            for f in ft_files
        }
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
    logger.info(f"Total: {len(all_rows)} FT events from "
                f"{len(ft_files) - n_failed} files")

    run_nums = np.array([r["runNum"] for r in all_rows], dtype=np.int32)
    event_nums = np.array([r["eventNum"] for r in all_rows], dtype=np.int32)

    save_dict = dict(
        runNum=run_nums,
        eventNum=event_nums,
        station_id=np.int32(args.station_id),
        channels=np.array(args.channels, dtype=np.int32),
    )
    for ch in args.channels:
        col = f"ch{ch}_noise_RMS"
        save_dict[col] = np.array(
            [r[col] for r in all_rows], dtype=np.float32)

    np.savez_compressed(args.output, **save_dict)
    print(f"\nSaved {len(all_rows)} events to {args.output}")


if __name__ == "__main__":
    main()
