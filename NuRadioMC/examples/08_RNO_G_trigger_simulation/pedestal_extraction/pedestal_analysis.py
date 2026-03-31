#!/usr/bin/env python3
"""Extract per-channel ADC clip thresholds from pedestal.root files.

Crawls a data directory for run*/pedestal.root files, extracts the
per-channel mean pedestal voltage, and computes asymmetric clip
thresholds for the RADIANT 12-bit ADC (0-2.5V range). Outputs a
YAML file whose median pedestal can be passed to simulate.py via
the --pedestal_voltage flag.

Usage:
    # All pedestals in a directory:
    python pedestal_analysis.py --data_dir /path/to/station23/

    # Only 2022 pedestals (filtered by timestamp):
    python pedestal_analysis.py --data_dir /path/to/station23/ --year 2022

    # Compare against reference pedestal files:
    python pedestal_analysis.py --data_dir /path/to/station23/ --year 2022 \
        --compare_files /path/to/run1000/pedestal.root /path/to/run3400/pedestal.root
"""

import argparse
import datetime
import glob
import os
import sys
import numpy as np
import yaml
from joblib import Parallel, delayed

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

ADC_TO_MV = 2500.0 / 4095.0
N_CHANNELS = 24


def process_one_file(ped_file):
    """Return (run_num, timestamp, per-channel mean mV) from a single pedestal ROOT file."""
    import uproot
    try:
        run_str = os.path.basename(os.path.dirname(ped_file)).replace("run", "")
        run_num = int(run_str)

        f = uproot.open(ped_file)
        tree = f['pedestals']
        data = tree.arrays(library='np')
        peds_adc = data['pedestals[24][4096]'][0]  # (24, 4096) uint16
        when = int(data['when'][0])

        means_mv = np.array([np.mean(peds_adc[ch]) * ADC_TO_MV for ch in range(N_CHANNELS)])
        return run_num, when, means_mv
    except Exception:
        return None


def filter_by_year(results, year):
    """Filter results to only include entries from the specified year (UTC)."""
    filtered = []
    for run_num, when, means_mv in results:
        dt = datetime.datetime.utcfromtimestamp(when)
        if dt.year == year:
            filtered.append((run_num, when, means_mv))
    return filtered


def main():
    """Process pedestal files in parallel and output clip thresholds as YAML."""
    parser = argparse.ArgumentParser(
        description="Extract per-channel ADC clip thresholds from pedestal.root files.")
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing run*/pedestal.root files")
    parser.add_argument("--year", type=int, default=None,
                        help="Filter pedestals by year (uses UTC timestamp from ROOT file)")
    parser.add_argument("--compare_files", nargs="*", default=None,
                        help="Optional pedestal.root files to compare against")
    parser.add_argument("--station_id", type=int, required=True,
                        help="Station ID (used in output filenames)")
    parser.add_argument("--outdir", default=".",
                        help="Output directory for YAML and NPZ (default: current directory)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    ped_files = sorted(glob.glob(os.path.join(args.data_dir, "run*/pedestal.root")))
    print(f"Found {len(ped_files)} pedestal files in {args.data_dir}")

    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 20))
    print(f"Processing with {n_cpus} workers...")

    results = Parallel(n_jobs=n_cpus, verbose=5)(
        delayed(process_one_file)(pf) for pf in ped_files
    )

    results = [r for r in results if r is not None]
    results.sort(key=lambda x: x[0])
    print(f"Successfully read {len(results)} / {len(ped_files)} files")

    if args.year:
        n_before = len(results)
        results = filter_by_year(results, args.year)
        print(f"Filtered to year {args.year}: {len(results)} / {n_before} runs")
        if not results:
            print(f"No pedestal files found for year {args.year}")
            sys.exit(1)

    run_nums = np.array([r[0] for r in results])
    timestamps = np.array([r[1] for r in results])
    all_peds = np.array([r[2] for r in results])

    dt_min = datetime.datetime.utcfromtimestamp(timestamps.min())
    dt_max = datetime.datetime.utcfromtimestamp(timestamps.max())

    npz_path = os.path.join(args.outdir, "pedestal_analysis_results.npz")
    np.savez(npz_path, run_nums=run_nums, timestamps=timestamps, pedestals_mv=all_peds)
    print(f"Saved NPZ to {npz_path}")

    compare_avg = None
    if args.compare_files:
        import uproot
        compare_peds = []
        for pf in args.compare_files:
            f = uproot.open(pf)
            data = f['pedestals']['pedestals[24][4096]'].array(library='np')
            peds_adc = data[0]
            compare_peds.append([np.mean(peds_adc[ch]) * ADC_TO_MV for ch in range(N_CHANNELS)])
        compare_avg = np.mean(compare_peds, axis=0)

    year_str = str(args.year) if args.year else "all"
    print(f"\n{'=' * 100}")
    print(f"PEDESTAL ANALYSIS: {len(results)} runs, year={year_str}, "
          f"date range: {dt_min:%Y-%m-%d} to {dt_max:%Y-%m-%d}")
    print(f"{'=' * 100}\n")

    header = (f"{'ch':>4} {'median':>8} {'mean':>9} {'std':>8} "
              f"{'p5':>8} {'p95':>8} {'min':>8} {'max':>8} ")
    if compare_avg is not None:
        header += f"{'compare':>8} {'diff':>8} "
    header += f"{'clip-':>10} {'clip+':>10}"
    print(header)
    print("-" * len(header))

    clip_thresholds = {}
    for ch in range(N_CHANNELS):
        col = all_peds[:, ch]
        med = np.median(col)
        mean = np.mean(col)
        std = np.std(col)
        p5 = np.percentile(col, 5)
        p95 = np.percentile(col, 95)
        mn = np.min(col)
        mx = np.max(col)

        new_lo = -round(med)
        new_hi = round(2500 - med)
        clip_thresholds[ch] = [int(new_lo), int(new_hi)]

        line = (f"{ch:>4} {med:>7.1f}m {mean:>8.1f}m {std:>7.1f}m "
                f"{p5:>7.1f}m {p95:>7.1f}m {mn:>7.1f}m {mx:>7.1f}m ")
        if compare_avg is not None:
            line += f"{compare_avg[ch]:>7.1f}m {med - compare_avg[ch]:>+7.1f}m "
        line += f"{new_lo:>9}m {new_hi:>9}m"
        print(line)

    yaml_data = {
        "clip_thresholds_mV": clip_thresholds,
        "metadata": {
            "station_id": args.station_id,
            "n_runs": len(results),
            "year": args.year,
            "date_range": [dt_min.strftime("%Y-%m-%d"), dt_max.strftime("%Y-%m-%d")],
            "data_dir": args.data_dir,
            "method": "median pedestal per channel across all runs",
        }
    }

    yaml_name = f"clip_thresholds_station{args.station_id}_{year_str}.yaml"
    yaml_path = os.path.join(args.outdir, yaml_name)
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    print(f"\nSaved YAML to {yaml_path}")

    print(f"\n{'=' * 100}")
    print(f"CLIP_THRESHOLDS_MV (year={year_str}, {len(results)} runs)")
    print(f"{'=' * 100}\n")
    print("CLIP_THRESHOLDS_MV = {")
    for ch in range(N_CHANNELS):
        lo, hi = clip_thresholds[ch]
        print(f"    {ch}: ({lo}, +{hi}),")
    print("}")


if __name__ == "__main__":
    main()
