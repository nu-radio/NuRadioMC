"""Measure how trigger-path Vrms stabilizes as a function of FT event count.

Sweeps N = 10, 25, 50, 100, 200, 500 FT events and measures the per-channel
trigger Vrms at each N. Repeats each N with different random seeds to get
error bars. Outputs a CSV of results and prints a convergence summary.

Supports parallelization via --chunk_id / --n_chunks for SLURM array jobs.
Use --merge to combine chunk CSVs into a single result.
"""

import argparse
import numpy as np
import os
import logging
import datetime as dt
import glob as _glob

from NuRadioReco.detector.RNO_G import rnog_detector
from NuRadioReco.modules.RNO_G import hardwareResponseIncorporator
from NuRadioReco.modules.measured_noise.RNO_G.noiseImporter import noiseImporter
from NuRadioReco.utilities import units

logger = logging.getLogger("vrms_convergence")
logging.basicConfig(level=logging.INFO)

TRIGGER_CHANNELS = [0, 1, 2, 3]


def measure_trigger_vrms(noise_imp, hw_resp, det, station_id, n_events, seed):
    """Draw n_events FT events and measure trigger-path Vrms per channel.

    Returns dict mapping channel_id to Vrms in V.
    """
    rng = np.random.default_rng(seed)

    sum_sq = {ch: 0.0 for ch in TRIGGER_CHANNELS}
    n_samples_total = {ch: 0 for ch in TRIGGER_CHANNELS}

    event_indices = noise_imp._noiseImporter__event_index_list
    station_ids = noise_imp._noiseImporter__station_id_list
    mask = station_ids == station_id

    for i in range(n_events):
        idx = int(rng.choice(event_indices[mask]))
        noise_event = noise_imp._noise_reader.get_event_by_index(idx)
        if noise_event is None:
            continue
        sid = noise_event.get_station_ids()[0]
        noise_station = noise_event.get_station(sid)

        for ch_id in TRIGGER_CHANNELS:
            noise_ch = noise_station.get_channel(ch_id)
            noise_trace = noise_ch.get_trace()
            noise_sr = noise_ch.get_sampling_rate()

            trig_sr = 5.0 * units.GHz
            n_up = int(round(len(noise_trace) * trig_sr / noise_sr))
            noise_up = noiseImporter._upsample(noise_trace, n_up)

            transfer = noise_imp._get_readout_to_trigger_transfer(
                ch_id, n_up, det, station_id, trig_sr)
            noise_fft = np.fft.rfft(noise_up)
            trig_noise = np.fft.irfft(noise_fft * transfer, n=n_up)

            sum_sq[ch_id] += np.sum(trig_noise ** 2)
            n_samples_total[ch_id] += len(trig_noise)

    vrms = {}
    for ch_id in TRIGGER_CHANNELS:
        if n_samples_total[ch_id] > 0:
            vrms[ch_id] = np.sqrt(sum_sq[ch_id] / n_samples_total[ch_id])
        else:
            vrms[ch_id] = np.nan
    return vrms


def build_job_list(n_values, n_repeats):
    """Build list of (n_evt, repeat) pairs."""
    jobs = []
    for n_evt in n_values:
        for rep in range(n_repeats):
            jobs.append((n_evt, rep))
    return jobs


def merge_chunks(outdir, output_name="vrms_convergence.csv"):
    """Merge chunk CSVs into a single file and print summary."""
    import pandas as pd

    chunk_files = sorted(_glob.glob(os.path.join(outdir, "vrms_convergence_chunk*.csv")))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk CSVs found in {outdir}")

    dfs = [pd.read_csv(f) for f in chunk_files]
    df = pd.concat(dfs, ignore_index=True).sort_values(["n_events", "repeat"])

    csv_path = os.path.join(outdir, output_name)
    df.to_csv(csv_path, index=False)
    print(f"Merged {len(chunk_files)} chunks -> {csv_path} ({len(df)} rows)")

    print_summary(df)
    return df


def print_summary(df):
    """Print convergence summary table."""
    n_values = sorted(df["n_events"].unique())

    print("\n--- Mean +/- std (mV) ---")
    for n_evt in n_values:
        sub = df[df["n_events"] == n_evt]
        parts = []
        for ch_id in TRIGGER_CHANNELS:
            col = f"vrms_ch{ch_id}_mV"
            parts.append(f"ch{ch_id}: {sub[col].mean():.3f} +/- {sub[col].std():.3f}")
        print(f"N={n_evt:>4}: {', '.join(parts)}")

    print("\n--- Relative spread (std/mean %) ---")
    for n_evt in n_values:
        sub = df[df["n_events"] == n_evt]
        parts = []
        for ch_id in TRIGGER_CHANNELS:
            col = f"vrms_ch{ch_id}_mV"
            rel = 100 * sub[col].std() / sub[col].mean()
            parts.append(f"ch{ch_id}: {rel:.2f}%")
        print(f"N={n_evt:>4}: {', '.join(parts)}")


def main():
    """Run trigger Vrms convergence study across N values and repeats."""
    parser = argparse.ArgumentParser(description="Trigger Vrms convergence study")
    parser.add_argument("--ft_noise_dir", type=str, default=None)
    parser.add_argument("--station_id", type=int, default=23)
    parser.add_argument("--event_time", type=str, default="2022-10-01")
    parser.add_argument("--detector_file", type=str, default=None)
    parser.add_argument("--n_repeats", type=int, default=10)
    parser.add_argument("--outdir", type=str, default=".")

    parser.add_argument("--chunk_id", type=int, default=None,
                        help="Which chunk to run (0-indexed)")
    parser.add_argument("--n_chunks", type=int, default=None,
                        help="Total number of chunks")
    parser.add_argument("--merge", action="store_true",
                        help="Merge chunk CSVs instead of running")
    args = parser.parse_args()

    n_values = [10, 25, 50, 100, 200, 500]

    if args.merge:
        merge_chunks(args.outdir)
        return

    if args.ft_noise_dir is None:
        parser.error("--ft_noise_dir is required (unless --merge)")

    all_jobs = build_job_list(n_values, args.n_repeats)

    if args.chunk_id is not None and args.n_chunks is not None:
        chunk_jobs = [j for i, j in enumerate(all_jobs) if i % args.n_chunks == args.chunk_id]
        logger.info(f"Chunk {args.chunk_id}/{args.n_chunks}: {len(chunk_jobs)} jobs")
    else:
        chunk_jobs = all_jobs

    det = rnog_detector.Detector(
        detector_file=args.detector_file, log_level=logging.WARNING,
        always_query_entire_description=False,
        select_stations=args.station_id)
    det.update(dt.datetime.fromisoformat(args.event_time))

    hw_resp = hardwareResponseIncorporator.hardwareResponseIncorporator()
    hw_resp.begin(trigger_channels=TRIGGER_CHANNELS)

    ft_files = sorted(_glob.glob(os.path.join(args.ft_noise_dir, "station*_run*.root")))
    if not ft_files:
        ft_files = sorted(_glob.glob(os.path.join(args.ft_noise_dir, "run*/waveforms.root")))
    if not ft_files:
        raise FileNotFoundError(f"No FT ROOT files in {args.ft_noise_dir}")

    noise_imp = noiseImporter()
    noise_imp.begin(
        noise_files=ft_files,
        match_station_id=True,
        scramble_noise_file_order=True,
        random_seed=0,
        inject_trigger_copies=True,
        trigger_channels=TRIGGER_CHANNELS,
        hardware_response_incorporator=hw_resp,
        reader_kwargs={
            "select_runs": False,
            "convert_to_voltage": True,
            "apply_baseline_correction": "median",
        },
    )

    import pandas as pd
    rows = []
    for n_evt, rep in chunk_jobs:
        seed = rep * 1000 + n_evt
        vrms = measure_trigger_vrms(
            noise_imp, hw_resp, det, args.station_id, n_evt, seed)
        row = {"n_events": n_evt, "repeat": rep, "seed": seed}
        for ch_id in TRIGGER_CHANNELS:
            row[f"vrms_ch{ch_id}_mV"] = vrms[ch_id] / units.mV
        rows.append(row)
        logger.info(f"N={n_evt}, rep={rep}: "
                    + ", ".join(f"ch{c}={vrms[c]/units.mV:.3f}" for c in TRIGGER_CHANNELS))

    df = pd.DataFrame(rows)
    if args.chunk_id is not None:
        csv_name = f"vrms_convergence_chunk{args.chunk_id}.csv"
    else:
        csv_name = "vrms_convergence.csv"
    csv_path = os.path.join(args.outdir, csv_name)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved to {csv_path}")

    if args.chunk_id is None:
        print_summary(df)


if __name__ == "__main__":
    main()
