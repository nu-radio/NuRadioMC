"""Validate the per-channel RMS cleaning threshold against simulated thermal noise.

Sweeps the MAD-sigma threshold from 2.5 to 8.0 and measures the post-cut
distribution shape (kurtosis, skewness) on helper and PA channels. Compares
against a simulated thermal noise reference to identify the threshold where
the real FT data converges to thermal behavior.

The chosen threshold (default 4.0) is where the post-cut max helper kurtosis
matches the simulated thermal reference (~0.28). Tighter cuts sculpt the
thermal distribution (negative skewness appears). Looser cuts leave
measurable non-Gaussian tails.

Reads per-channel noise RMS from the NPZ produced by extract_ft_rms.py.

Produces:
    figures/ft_cleaning_threshold_convergence.png - 4-panel convergence plot

Usage:
    python validate_threshold.py \\
        --rms_npz ft_rms_station23.npz

    python validate_threshold.py \\
        --rms_npz ft_rms_station23.npz \\
        --sim_nur nur_sim_noise/noise_season2023_st23_1000events.nur \\
        --output_dir figures/
"""

import argparse
import os
import numpy as np
from scipy import stats as spstats
import matplotlib.pyplot as plt

HELPER_CHS = [9, 10, 11, 21, 22, 23]
PA_CHS = [0, 1, 2, 3]
ALL_CHS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23]


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


def load_rms_from_npz(npz_path):
    """Load per-channel noise RMS arrays from the RMS NPZ.

    Args:
        npz_path: Path to NPZ from extract_ft_rms.py.

    Returns:
        Dict mapping channel ID to RMS array, plus n_events count.
    """
    data = np.load(npz_path)
    channels = data["channels"]
    rms_data = {}
    for ch in channels:
        key = f"ch{ch}_noise_RMS"
        if key in data:
            rms_data[ch] = data[key]
    n_events = len(data["runNum"])
    return rms_data, n_events


def compute_zscores(rms_data, channels):
    """Compute per-channel MAD z-scores for noise RMS.

    Args:
        rms_data: Dict mapping channel ID to RMS array.
        channels: List of channel IDs.

    Returns:
        Dict mapping channel ID to z-score array.
    """
    zscores = {}
    for ch in channels:
        if ch not in rms_data:
            continue
        vals = rms_data[ch]
        med, sig = robust_stats(vals)
        if sig > 0:
            zscores[ch] = (vals - med) / sig
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

    rms_by_ch = {}
    for evt in reader.run():
        stn = evt.get_station(station_id)
        if stn is None:
            continue
        for ch in stn.iter_channels():
            ch_id = ch.get_id()
            rms_by_ch.setdefault(ch_id, []).append(np.std(ch.get_trace()))

    max_kurt = max(spstats.kurtosis(np.array(v)) for v in rms_by_ch.values())
    max_skew = max(abs(spstats.skew(np.array(v))) for v in rms_by_ch.values())
    n_events = max(len(v) for v in rms_by_ch.values())

    return {"max_kurtosis": max_kurt, "max_skewness": max_skew,
            "n_events": n_events}


def sweep_thresholds(rms_data, zscores, thresholds):
    """Sweep thresholds and compute post-cut distribution shape.

    Args:
        rms_data: Dict mapping channel ID to RMS array.
        zscores: Dict mapping channel ID to z-score array.
        thresholds: Array of threshold values to sweep.

    Returns:
        Dict of arrays keyed by metric name.
    """
    n_events = len(next(iter(rms_data.values())))
    z_matrix = np.column_stack([zscores[ch] for ch in sorted(zscores)])

    results = {
        "threshold": thresholds,
        "helper_avg_kurt": [], "helper_max_kurt": [],
        "pa_avg_kurt": [], "helper_avg_skew": [], "pct_cut": [],
    }

    for thresh in thresholds:
        cut = (z_matrix > thresh).any(axis=1)
        clean = ~cut
        results["pct_cut"].append(100 * cut.sum() / n_events)

        hk, hs, pk = [], [], []
        for ch in HELPER_CHS:
            if ch in rms_data:
                vals = rms_data[ch][clean]
                hk.append(spstats.kurtosis(vals))
                hs.append(spstats.skew(vals))
        for ch in PA_CHS:
            if ch in rms_data:
                vals = rms_data[ch][clean]
                pk.append(spstats.kurtosis(vals))

        results["helper_avg_kurt"].append(np.mean(hk))
        results["helper_max_kurt"].append(max(hk))
        results["pa_avg_kurt"].append(np.mean(pk))
        results["helper_avg_skew"].append(np.mean(hs))

    for key in results:
        results[key] = np.array(results[key])

    return results


def plot_convergence(results, sim_ref, output_dir, n_events,
                     rms_data=None, zscores=None):
    """Generate the threshold convergence plot with before/after distributions.

    Args:
        results: Dict from sweep_thresholds.
        sim_ref: Dict with max_kurtosis, max_skewness from sim.
        output_dir: Directory for output figure.
        n_events: Total number of events.
        rms_data: Dict of per-channel RMS arrays (for distribution panels).
        zscores: Dict of per-channel z-score arrays (for distribution panels).
    """
    thresholds = results["threshold"]
    has_dists = rms_data is not None and zscores is not None

    if has_dists:
        fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

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
        z_matrix = np.column_stack([zscores[ch] for ch in sorted(zscores)])
        cut_4sig = (z_matrix > 4.0).any(axis=1)
        clean = ~cut_4sig
        n_after = clean.sum()

        ax = axes[0, 1]
        ax.plot(thresholds, results["pct_cut"], "o-", color="C0",
                markersize=4)
        ax.axvline(4.0, color="C3", ls="--", lw=1.5, alpha=0.7,
                   label="Chosen threshold (4.0)")
        ax.set_xlabel("Threshold (MAD sigma)")
        ax.set_ylabel("Events removed (%)")
        ax.set_title("Fraction removed")
        ax.legend(fontsize=8)

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

        max_z = z_matrix.max(axis=1)
        max_z_clean = max_z[clean]

        ax = axes[0, 2]
        bins = np.linspace(0, 10, 120)
        ax.hist(max_z, bins=bins, alpha=0.6, color="C3", density=True,
                label=f"Before ({n_events:,})")
        ax.hist(max_z_clean, bins=bins, alpha=0.6, color="C0", density=True,
                label=f"After ({n_after:,})")
        ax.axvline(4.0, color="C3", ls="--", lw=1.5, label="4$\\sigma$ cut")
        ax.set_xlabel("Max z-score across channels")
        ax.set_ylabel("Density")
        ax.set_title("Max per-channel z-score")
        ax.set_yscale("log")
        ax.legend(fontsize=8)

        mean_z = z_matrix.mean(axis=1)
        mean_z_clean = mean_z[clean]

        ax = axes[1, 2]
        lo = np.percentile(mean_z, 0.01)
        hi = np.percentile(mean_z, 99.99)
        bins = np.linspace(lo, hi, 120)
        ax.hist(mean_z, bins=bins, alpha=0.6, color="C3", density=True,
                label=f"Before ({n_events:,})")
        ax.hist(mean_z_clean, bins=bins, alpha=0.6, color="C0", density=True,
                label=f"After ({n_after:,})")
        ax.set_xlabel("Mean z-score across channels")
        ax.set_ylabel("Density")
        ax.set_title("Mean per-channel z-score")
        ax.set_yscale("log")
        ax.legend(fontsize=8)

    fig.suptitle(f"FT noise cleaning threshold calibration\n"
                 f"Station 23, {n_events:,} events, "
                 f"per-channel noise_RMS MAD z-score cut",
                 fontsize=14)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, "ft_cleaning_threshold_convergence.png")
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
    parser.add_argument("--rms_npz", type=str, required=True,
                        help="Path to NPZ from extract_ft_rms.py")
    parser.add_argument("--sim_nur", type=str, default=None,
                        help="Path to simulated thermal noise NUR for reference. "
                             "If not provided, uses hardcoded reference values.")
    parser.add_argument("--station", type=int, default=23)
    parser.add_argument("--output_dir", type=str, default="figures/",
                        help="Output directory for plots (default: figures/)")
    args = parser.parse_args()

    print(f"Loading RMS data from {args.rms_npz}...")
    rms_data, n_events = load_rms_from_npz(args.rms_npz)
    print(f"  {n_events:,} events, {len(rms_data)} channels")

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

    print("Computing per-channel z-scores...")
    zscores = compute_zscores(rms_data, ALL_CHS)

    thresholds = np.arange(2.5, 8.05, 0.25)
    print(f"Sweeping {len(thresholds)} thresholds from "
          f"{thresholds[0]} to {thresholds[-1]}...")
    results = sweep_thresholds(rms_data, zscores, thresholds)

    print_sweep_table(results, sim_ref)
    plot_convergence(results, sim_ref, args.output_dir, n_events,
                     rms_data=rms_data, zscores=zscores)


if __name__ == "__main__":
    main()
