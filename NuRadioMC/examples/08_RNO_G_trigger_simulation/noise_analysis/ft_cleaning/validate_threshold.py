"""Validate the per-channel RMS cleaning threshold against simulated thermal noise.

Sweeps the MAD-sigma threshold from 2.5 to 8.0 and measures the post-cut
distribution shape (kurtosis, skewness) on helper and PA channels. Compares
against a simulated thermal noise reference to identify the threshold where
the real FT data converges to thermal behavior.

The chosen threshold (default 4.0) is where the post-cut max helper kurtosis
matches the simulated thermal reference (~0.28). Tighter cuts sculpt the
thermal distribution (negative skewness appears). Looser cuts leave
measurable non-Gaussian tails.

Produces:
    figures/threshold_convergence.png - 4-panel convergence plot

Usage:
    python validate_threshold.py \\
        --feature_file /path/to/merged_feature_output.h5

    python validate_threshold.py \\
        --feature_file /path/to/merged_feature_output.h5 \\
        --sim_nur nur_sim_noise/noise_season2023_st23_1000events.nur \\
        --output_dir figures/
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy import stats as spstats
import matplotlib.pyplot as plt

HELPER_CHS = [9, 10, 11, 21, 22, 23]
PA_CHS = [0, 1, 2, 3]
ALL_CHS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23]
VPOL_CHS = [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23]

KEY_FEATURES = [
    "chAvgSNR", "maxAmplitude", "impulsivity", "coherentSNR", "outlier_score",
]


def robust_stats(arr):
    """Return median and MAD-based sigma for an array.

    Args:
        arr: 1D numeric array.

    Returns:
        Tuple of (median, mad_sigma).
    """
    med = np.nanmedian(arr)
    mad = np.nanmedian(np.abs(arr - med))
    return med, 1.4826 * mad


def composite_flag(df):
    """Apply the 5-sigma composite flag + multi-channel RMS criterion.

    Args:
        df: DataFrame with feature columns.

    Returns:
        Boolean array, True = flagged.
    """
    n = len(df)
    flag = np.zeros(n, dtype=bool)

    for feat in KEY_FEATURES:
        if feat not in df.columns:
            continue
        vals = df[feat].values
        med, sig = robust_stats(vals)
        if sig > 0:
            flag |= vals > med + 5 * sig

    rms_cols = [f"ch{ch}_noise_RMS" for ch in VPOL_CHS]
    rms_zscores = pd.DataFrame()
    for col in rms_cols:
        if col not in df.columns:
            continue
        med, sig = robust_stats(df[col].values)
        if sig > 0:
            rms_zscores[col] = (df[col].values - med) / sig

    n_chs_above = (rms_zscores > 3).sum(axis=1)
    flag |= n_chs_above >= 3
    return flag


def compute_zscores(df, channels):
    """Compute per-channel MAD z-scores for noise_RMS.

    Args:
        df: DataFrame with ch{N}_noise_RMS columns.
        channels: List of channel IDs.

    Returns:
        DataFrame of z-scores, one column per channel.
    """
    zscores = pd.DataFrame(index=df.index)
    for ch in channels:
        col = f"ch{ch}_noise_RMS"
        if col not in df.columns:
            continue
        vals = df[col].values
        med, sig = robust_stats(vals)
        if sig > 0:
            zscores[f"ch{ch}"] = (vals - med) / sig
    return zscores


def get_sim_reference(sim_nur, station_id):
    """Load sim noise NUR and compute reference kurtosis/skewness ranges.

    Args:
        sim_nur: Path to NUR file with simulated thermal noise.
        station_id: Station ID.

    Returns:
        Dict with max_kurtosis, max_skewness across all channels.
    """
    from NuRadioReco.modules.io import eventReader as er

    reader = er.eventReader()
    reader.begin(sim_nur)

    rms_data = {}
    for evt in reader.run():
        stn = evt.get_station(station_id)
        if stn is None:
            continue
        for ch in stn.iter_channels():
            ch_id = ch.get_id()
            rms_data.setdefault(ch_id, []).append(np.std(ch.get_trace()))

    max_kurt = max(spstats.kurtosis(np.array(v)) for v in rms_data.values())
    max_skew = max(abs(spstats.skew(np.array(v))) for v in rms_data.values())
    n_events = max(len(v) for v in rms_data.values())

    return {"max_kurtosis": max_kurt, "max_skewness": max_skew,
            "n_events": n_events}


def sweep_thresholds(df, base_flag, zscores, thresholds):
    """Sweep thresholds and compute post-cut distribution shape.

    Args:
        df: Full DataFrame.
        base_flag: Boolean array of composite-flagged events.
        zscores: DataFrame of per-channel z-scores.
        thresholds: Array of threshold values to sweep.

    Returns:
        Dict of arrays keyed by metric name.
    """
    results = {
        "threshold": thresholds,
        "helper_avg_kurt": [], "helper_max_kurt": [],
        "pa_avg_kurt": [], "helper_avg_skew": [], "pct_cut": [],
    }

    for thresh in thresholds:
        cut = (zscores > thresh).any(axis=1).values
        clean = ~cut
        results["pct_cut"].append(100 * cut.sum() / len(df))

        hk, hs, pk = [], [], []
        for ch in HELPER_CHS:
            col = f"ch{ch}_noise_RMS"
            if col in df.columns:
                vals = df.loc[clean, col].values
                hk.append(spstats.kurtosis(vals))
                hs.append(spstats.skew(vals))
        for ch in PA_CHS:
            col = f"ch{ch}_noise_RMS"
            if col in df.columns:
                vals = df.loc[clean, col].values
                pk.append(spstats.kurtosis(vals))

        results["helper_avg_kurt"].append(np.mean(hk))
        results["helper_max_kurt"].append(max(hk))
        results["pa_avg_kurt"].append(np.mean(pk))
        results["helper_avg_skew"].append(np.mean(hs))

    for key in results:
        results[key] = np.array(results[key])

    return results


def plot_convergence(results, sim_ref, output_dir, df=None, zscores=None):
    """Generate the threshold convergence plot with before/after distributions.

    Args:
        results: Dict from sweep_thresholds.
        sim_ref: Dict with max_kurtosis, max_skewness from sim.
        output_dir: Directory for output figure.
        df: Full DataFrame (needed for distribution panels).
        zscores: Per-channel z-score DataFrame (needed for distribution panels).
    """
    thresholds = results["threshold"]
    has_dists = df is not None and zscores is not None

    if has_dists:
        fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Top-left: helper kurtosis
    ax = axes[0, 0]
    ax.plot(thresholds, results["helper_max_kurt"], "o-", color="C0",
            markersize=4, label="Helper max")
    ax.plot(thresholds, results["helper_avg_kurt"], "s-", color="C1",
            markersize=4, label="Helper mean")
    ax.axhline(sim_ref["max_kurtosis"], color="C2", ls="--", lw=1.5,
               label=f"Sim thermal ref ({sim_ref['max_kurtosis']:.2f})")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.axvline(4.0, color="C3", ls="--", lw=1.5, alpha=0.7,
               label="Chosen threshold (4.0)")
    ax.set_xlabel("Threshold (MAD sigma)")
    ax.set_ylabel("Excess kurtosis")
    ax.set_title("Helper kurtosis vs threshold")
    ax.set_ylim(-0.1, 1.5)
    ax.legend(fontsize=8)

    # Bottom-left: helper skewness + PA kurtosis combined
    ax = axes[1, 0]
    ax.plot(thresholds, results["helper_avg_skew"], "o-", color="C0",
            markersize=4, label="Helper skewness")
    ax.plot(thresholds, results["pa_avg_kurt"], "s-", color="C1",
            markersize=4, label="PA kurtosis")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.axhline(sim_ref["max_skewness"], color="C2", ls="--", lw=1.5,
               label=f"Sim max ({sim_ref['max_skewness']:.2f})")
    ax.axvline(4.0, color="C3", ls="--", lw=1.5, alpha=0.7,
               label="Chosen threshold (4.0)")
    ax.set_xlabel("Threshold (MAD sigma)")
    ax.set_ylabel("Value")
    ax.set_title("Skewness and PA kurtosis vs threshold")
    ax.set_ylim(-0.1, 0.5)
    ax.legend(fontsize=8)

    if has_dists:
        cut_4sig = (zscores > 4.0).any(axis=1).values
        clean = ~cut_4sig
        n_before = len(df)
        n_after = clean.sum()

        # Top-middle: fraction removed
        ax = axes[0, 1]
        ax.plot(thresholds, results["pct_cut"], "o-", color="C0",
                markersize=4)
        ax.axvline(4.0, color="C3", ls="--", lw=1.5, alpha=0.7,
                   label="Chosen threshold (4.0)")
        ax.set_xlabel("Threshold (MAD sigma)")
        ax.set_ylabel("Events removed (%)")
        ax.set_title("Fraction removed")
        ax.legend(fontsize=8)

        # Bottom-middle: fraction removed log scale
        ax = axes[1, 1]
        ax.plot(thresholds, results["pct_cut"], "o-", color="C0",
                markersize=4)
        ax.axvline(4.0, color="C3", ls="--", lw=1.5, alpha=0.7,
                   label="Chosen threshold (4.0)")
        ax.set_xlabel("Threshold (MAD sigma)")
        ax.set_ylabel("Events removed (%)")
        ax.set_title("Fraction removed (log scale)")
        ax.set_yscale("log")
        ax.legend(fontsize=8)

        # Top-right: max z-score across channels
        max_z_all = zscores.max(axis=1).values
        max_z_clean = max_z_all[clean]

        ax = axes[0, 2]
        bins = np.linspace(0, 10, 120)
        ax.hist(max_z_all, bins=bins, alpha=0.6, color="C3", density=True,
                label=f"Before ({n_before:,})")
        ax.hist(max_z_clean, bins=bins, alpha=0.6, color="C0", density=True,
                label=f"After ({n_after:,})")
        ax.axvline(4.0, color="C3", ls="--", lw=1.5, label="4$\\sigma$ cut")
        ax.set_xlabel("Max z-score across channels")
        ax.set_ylabel("Density")
        ax.set_title("Max per-channel z-score")
        ax.set_yscale("log")
        ax.legend(fontsize=8)

        # Bottom-right: mean z-score across channels
        mean_z_all = zscores.mean(axis=1).values
        mean_z_clean = mean_z_all[clean]

        ax = axes[1, 2]
        lo = np.percentile(mean_z_all, 0.01)
        hi = np.percentile(mean_z_all, 99.99)
        bins = np.linspace(lo, hi, 120)
        ax.hist(mean_z_all, bins=bins, alpha=0.6, color="C3", density=True,
                label=f"Before ({n_before:,})")
        ax.hist(mean_z_clean, bins=bins, alpha=0.6, color="C0", density=True,
                label=f"After ({n_after:,})")
        ax.set_xlabel("Mean z-score across channels")
        ax.set_ylabel("Density")
        ax.set_title("Mean per-channel z-score")
        ax.set_yscale("log")
        ax.legend(fontsize=8)

    n_events = int(len(df)) if df is not None else ""
    fig.suptitle(f"FT noise cleaning threshold calibration\n"
                 f"Station 23, {n_events:,} events, "
                 f"per-channel noise_RMS MAD z-score cut",
                 fontsize=14)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, "threshold_convergence.png")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")


def print_sweep_table(results, sim_ref):
    """Print the threshold sweep as a formatted table."""
    print(f"\n{'Thresh':>7} {'% cut':>7} | "
          f"{'Helper avg kurt':>16} {'Helper max kurt':>16} "
          f"{'PA avg kurt':>13} | {'Helper avg skew':>16} {'Converged?':>11}")
    print('-' * 110)

    for i, thresh in enumerate(results["threshold"]):
        converged = ("YES" if results["helper_max_kurt"][i] < 2 * sim_ref["max_kurtosis"]
                     else "no")
        print(f"{thresh:>7.2f} {results['pct_cut'][i]:>6.2f}% | "
              f"{results['helper_avg_kurt'][i]:>16.3f} "
              f"{results['helper_max_kurt'][i]:>16.3f} "
              f"{results['pa_avg_kurt'][i]:>13.3f} | "
              f"{results['helper_avg_skew'][i]:>16.3f} {converged:>11}")

    print(f"\nSim thermal reference: kurtosis < {sim_ref['max_kurtosis']:.2f}, "
          f"skewness < {sim_ref['max_skewness']:.2f} "
          f"({sim_ref['n_events']} events)")


def main():
    """Run the threshold validation sweep."""
    parser = argparse.ArgumentParser(
        description="Validate FT noise cleaning threshold against sim reference.")
    parser.add_argument("--feature_file", type=str, required=True,
                        help="Path to merged feature HDF5")
    parser.add_argument("--sim_nur", type=str, default=None,
                        help="Path to simulated thermal noise NUR for reference. "
                             "If not provided, uses hardcoded reference values.")
    parser.add_argument("--station", type=int, default=23)
    parser.add_argument("--output_dir", type=str, default="figures/",
                        help="Output directory for plots (default: figures/)")
    args = parser.parse_args()

    print(f"Loading features from {args.feature_file}...")
    df = pd.read_hdf(args.feature_file, key="data")
    print(f"  {len(df):,} events")

    if args.sim_nur:
        print(f"Computing sim reference from {args.sim_nur}...")
        sim_ref = get_sim_reference(args.sim_nur, args.station)
        print(f"  Sim ref: max_kurtosis={sim_ref['max_kurtosis']:.3f}, "
              f"max_skewness={sim_ref['max_skewness']:.3f} "
              f"({sim_ref['n_events']} events)")
    else:
        sim_ref = {"max_kurtosis": 0.28, "max_skewness": 0.22, "n_events": 1000}
        print(f"  Using hardcoded sim reference: kurtosis={sim_ref['max_kurtosis']}, "
              f"skewness={sim_ref['max_skewness']}")

    print("Computing composite flag...")
    base_flag = composite_flag(df)
    print(f"  Composite flagged: {base_flag.sum():,}")

    print("Computing per-channel z-scores...")
    zscores = compute_zscores(df, ALL_CHS)

    thresholds = np.arange(2.5, 8.05, 0.25)
    print(f"Sweeping {len(thresholds)} thresholds from {thresholds[0]} to {thresholds[-1]}...")
    results = sweep_thresholds(df, base_flag, zscores, thresholds)

    print_sweep_table(results, sim_ref)
    plot_convergence(results, sim_ref, args.output_dir, df=df, zscores=zscores)


if __name__ == "__main__":
    main()
