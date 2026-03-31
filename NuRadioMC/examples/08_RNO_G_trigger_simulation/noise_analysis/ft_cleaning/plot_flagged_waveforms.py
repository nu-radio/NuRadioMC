"""Plot waveforms of randomly sampled flagged FT events for visual inspection.

Samples N events caught by the composite flag but NOT by the per-channel
RMS cut, loads their waveforms from ROOT data, and plots all channels
for each event.

Usage:
    python plot_flagged_waveforms.py \
        --feature_file /path/to/merged_feature_output.h5 \
        --ft_dir /path/to/handcarry/station23 \
        --n_samples 20 \
        --output_dir test_figs/
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData


ALL_CHS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23]
VPOL_CHS = [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23]
KEY_FEATURES = [
    "chAvgSNR", "maxAmplitude", "impulsivity", "coherentSNR", "outlier_score",
]


def robust_stats(arr):
    """Return median and MAD-based sigma."""
    med = np.nanmedian(arr)
    mad = np.nanmedian(np.abs(arr - med))
    return med, 1.4826 * mad


def find_stage1_only_events(df):
    """Find events caught by composite flag but not 4-sigma per-channel RMS."""
    # Per-channel 4-sigma RMS flag
    rms_flag = np.zeros(len(df), dtype=bool)
    for ch in ALL_CHS:
        col = f"ch{ch}_noise_RMS"
        vals = df[col].values
        med, sig = robust_stats(vals)
        if sig > 0:
            rms_flag |= (vals - med) / sig > 4.0

    # Composite flag
    comp_flag = np.zeros(len(df), dtype=bool)
    for feat in KEY_FEATURES:
        vals = df[feat].values
        med, sig = robust_stats(vals)
        if sig > 0:
            comp_flag |= vals > med + 5 * sig

    rms_zscores = pd.DataFrame()
    for ch in VPOL_CHS:
        col = f"ch{ch}_noise_RMS"
        med, sig = robust_stats(df[col].values)
        if sig > 0:
            rms_zscores[col] = (df[col].values - med) / sig
    comp_flag |= (rms_zscores > 3).sum(axis=1).values >= 3

    stage1_only = comp_flag & ~rms_flag
    return stage1_only


def load_event_waveforms(ft_dir, station_id, run_num, event_num):
    """Load waveforms for a single event from handcarry ROOT data."""
    run_path = os.path.join(ft_dir, f"run{run_num}", "waveforms.root")
    if not os.path.exists(run_path):
        return None

    reader = readRNOGData()
    reader.begin(
        run_path,
        read_calibrated_data=False,
        apply_baseline_correction="median",
        convert_to_voltage=True,
        selectors=[lambda einfo: einfo.triggerType == "FORCE"],
    )

    evt = reader.get_event(run_nr=run_num, event_id=event_num)
    if evt is None:
        reader.end()
        return None

    stn = evt.get_station(station_id)
    if stn is None:
        reader.end()
        return None

    traces = {}
    for ch in stn.iter_channels():
        traces[ch.get_id()] = ch.get_trace().copy()

    reader.end()
    return traces


def plot_event(traces, run_num, event_num, output_dir, idx):
    """Plot all channels for a single event."""
    chs = sorted(traces.keys())
    n_chs = len(chs)
    fig, axes = plt.subplots(n_chs, 1, figsize=(12, 1.5 * n_chs), sharex=True)

    for i, ch_id in enumerate(chs):
        ax = axes[i]
        trace = traces[ch_id]
        t_ns = np.arange(len(trace)) / 3.2  # 3.2 GHz -> ns
        rms = np.std(trace)
        ax.plot(t_ns, trace * 1e3, color="C0", lw=0.5)
        ax.set_ylabel(f"ch{ch_id}\n({rms*1e3:.2f} mV)", fontsize=8)
        ax.tick_params(labelsize=7)

    axes[-1].set_xlabel("Time (ns)")
    fig.suptitle(f"Event {idx}: run {run_num}, event {event_num}\n"
                 f"(composite-flagged, NOT caught by 4-sigma RMS)",
                 fontsize=11)
    plt.tight_layout()

    outpath = os.path.join(output_dir, f"event_{idx:02d}_run{run_num}_evt{event_num}.png")
    plt.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close()
    return outpath


def main():
    """Sample and plot flagged event waveforms."""
    parser = argparse.ArgumentParser(
        description="Plot waveforms of composite-only flagged FT events.")
    parser.add_argument("--feature_file", type=str, required=True)
    parser.add_argument("--ft_dir", type=str, required=True,
                        help="Handcarry data dir (e.g. .../handcarry/station23)")
    parser.add_argument("--station", type=int, default=23)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="test_figs/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading features from {args.feature_file}...")
    df = pd.read_hdf(args.feature_file, key="data")
    print(f"  {len(df):,} events")

    print("Finding composite-only flagged events...")
    stage1_only = find_stage1_only_events(df)
    print(f"  {stage1_only.sum()} events flagged by composite but not 4-sigma RMS")

    s1_df = df[stage1_only]
    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(len(s1_df), size=min(args.n_samples, len(s1_df)),
                            replace=False)
    sample = s1_df.iloc[sample_idx]

    print(f"Sampling {len(sample)} events, loading waveforms...")
    for i, (_, row) in enumerate(sample.iterrows()):
        run_num = int(row["runNum"])
        event_num = int(row["eventNum"])
        print(f"  [{i+1}/{len(sample)}] run {run_num}, event {event_num}...",
              end=" ", flush=True)

        traces = load_event_waveforms(args.ft_dir, args.station, run_num, event_num)
        if traces is None:
            print("SKIP (not found)")
            continue

        outpath = plot_event(traces, run_num, event_num, args.output_dir, i)
        print(f"saved")

    print(f"\nPlots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
