"""Per-station outlier vs reco-zenith / impulsivity correlation analysis.

For each event in the merged feature table:
  - Compute robust per-feature z-scores (median + MAD scaled to a Gaussian
    sigma via 1.4826) over the entire merged dataset. Robust because the
    reference distribution is the data itself; classical mean+std would
    be pulled toward outliers.
  - Per-event "outlier score" = max |z| across all numeric features.
    (Ignoring NaN / zero-spread columns.)

Then join with the merged reco file to attach the reconstructed vertex
and recompute zenith in the PA frame (PA center at z = -95 m), so a
vertex at (rho=0, z=0) maps to zenith 0 deg directly above the array.

Outputs:
  - outlier_score_distribution.png
  - outlier_vs_reco_zenith.png         (2D hist + median trend)
  - outlier_vs_impulsivity_pa.png      (2D hist + median trend)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# PA center depth in the reco frame (z=0 at the ice surface, PA below).
# Hardcoded to -95 m (canonical mid-PA depth from CLAUDE.md ch0..ch3).
PA_Z_M = -95.0

# Columns excluded from the feature set used for z-scoring (metadata,
# IDs, anything not a per-event measurement). Kept conservative; if a
# feature column starts with one of these prefixes it's also dropped.
_EXCLUDE_EXACT = {
    "event_number", "run_number", "station_id", "station", "year",
    "trigger_time", "readout_time", "trigger_type", "trigger_info",
}
_EXCLUDE_PREFIXES = ("source_", "_")


def robust_z_columns(df, cols):
    """Return a 2D np array of robust z-scores for ``cols`` of ``df``."""
    arr = df[cols].to_numpy(dtype=np.float64, copy=False)
    med = np.nanmedian(arr, axis=0)
    mad = np.nanmedian(np.abs(arr - med), axis=0)
    sigma = 1.4826 * mad
    # Columns with zero spread are uninformative -- mask them out
    sigma_safe = np.where(sigma > 0, sigma, np.nan)
    return (arr - med) / sigma_safe


def per_event_outlier_score(z):
    """Per-event max |z| across all features, ignoring NaNs."""
    az = np.abs(z)
    az[~np.isfinite(az)] = -np.inf
    score = np.nanmax(az, axis=1)
    score[~np.isfinite(score)] = np.nan
    return score


def per_event_extreme_count(z, threshold=5.0):
    """Per-event count of features with |z| > ``threshold``."""
    az = np.abs(z)
    az[~np.isfinite(az)] = 0.0
    return np.sum(az > threshold, axis=1)


def per_event_sum_z2(z):
    """Per-event sum of z^2 (Mahalanobis-like, assumes feature independence)."""
    return np.nansum(z * z, axis=1)


PHYSICAL_FEATURE_KEYS = (
    "impulsivity", "snr", "kurtosis", "entropy", "max_amplitude",
    "spectral_centroid", "spectral_bandwidth",
)


def select_physical_features(cols):
    """Subset of feature columns matching diagnostic-physics keywords."""
    return [c for c in cols if any(k in c for k in PHYSICAL_FEATURE_KEYS)]


def find_impulsivity_column(df):
    """Pick the most informative coherent impulsivity column available."""
    for c in ("coherent_impulsivity_pa", "coherent_impulsivity_vpol",
              "coherent_impulsivity_deep", "coherent_impulsivity_hpol"):
        if c in df.columns:
            return c
    # Fall back to a per-channel value if coherent ones aren't present
    per_ch = [c for c in df.columns if c.endswith("_impulsivity")
              and not c.startswith("coherent_")]
    return per_ch[0] if per_ch else None


def plot_score_dist(score, out_path, label):
    """1D histogram of the outlier score with key percentiles annotated."""
    finite = score[np.isfinite(score)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(finite, bins=80, range=(0, np.nanpercentile(finite, 99.5)),
            histtype="step", linewidth=1.4)
    for q, color in [(50, "#27ae60"), (95, "#f39c12"), (99, "#e74c3c")]:
        v = np.nanpercentile(finite, q)
        ax.axvline(v, color=color, linestyle="--", linewidth=1,
                   label=f"p{q}={v:.1f}")
    ax.set_xlabel(r"Per-event outlier score (max $|z|$ across features)")
    ax.set_ylabel("Events")
    ax.set_yscale("log")
    ax.set_title(f"{label}: outlier score distribution (N={len(finite):,})")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_score_vs_x(score, x, x_label, out_path, label,
                    x_range=None, score_range=None,
                    add_secondary_x_deg_to_rad=False):
    """2D hist of outlier score vs ``x`` plus per-bin median curve."""
    mask = np.isfinite(score) & np.isfinite(x)
    sc = score[mask]
    xv = x[mask]
    if x_range is None:
        x_range = (np.nanmin(xv), np.nanmax(xv))
    if score_range is None:
        score_range = (0, np.nanpercentile(sc, 99.5))

    fig, ax = plt.subplots(figsize=(7, 5))
    h, xedges, yedges, im = ax.hist2d(
        xv, sc, bins=(80, 80),
        range=[x_range, score_range],
        cmap="viridis", cmin=1,
    )
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Events / bin")

    # Per-bin median trend on top of the 2D histogram
    binned = pd.cut(xv, bins=xedges, include_lowest=True)
    medians = pd.Series(sc).groupby(binned, observed=False).median()
    centers = 0.5 * (xedges[1:] + xedges[:-1])
    ax.plot(centers, medians.to_numpy(), color="white",
            linewidth=2, label="Per-bin median")

    ax.set_xlabel(x_label)
    ax.set_ylabel(r"Outlier score (max $|z|$)")
    ax.set_title(f"{label}: outlier score vs {x_label} (N={len(sc):,})")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)

    if add_secondary_x_deg_to_rad:
        secax = ax.secondary_xaxis(
            "top", functions=(np.deg2rad, np.rad2deg))
        secax.set_xlabel("[rad]")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--reco", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--label", default="burn",
                   help="Used in titles + as a fallback if no station info.")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    feat = pd.read_hdf(args.features)
    reco = pd.read_hdf(args.reco)
    print(f"Loaded features {feat.shape} from {args.features}")
    print(f"Loaded reco     {reco.shape} from {args.reco}")

    # Pick numeric columns to score
    numeric_cols = [c for c in feat.columns if pd.api.types.is_numeric_dtype(feat[c])]
    feature_cols = [
        c for c in numeric_cols
        if c not in _EXCLUDE_EXACT
        and not any(c.startswith(pre) for pre in _EXCLUDE_PREFIXES)
    ]
    print(f"  scoring {len(feature_cols)} numeric features")

    z = robust_z_columns(feat, feature_cols)
    score_max = per_event_outlier_score(z)
    score_count5 = per_event_extreme_count(z, 5.0).astype(float)
    score_count10 = per_event_extreme_count(z, 10.0).astype(float)
    score_sumz2 = per_event_sum_z2(z)

    # Same statistics restricted to physical features only -- max|z| over
    # 559 mostly-redundant features is dominated by feature multiplicity.
    # Restricting to impulsivity/SNR/kurtosis/etc. cuts the multiple-
    # comparison floor and tends to reveal cleaner correlations.
    phys_cols = select_physical_features(feature_cols)
    z_phys = robust_z_columns(feat, phys_cols)
    score_phys_max = per_event_outlier_score(z_phys)
    print(f"  physical feature subset: {len(phys_cols)} cols")

    feat = feat.copy()
    feat["score_max"] = score_max
    feat["score_count5"] = score_count5
    feat["score_count10"] = score_count10
    feat["score_sumz2"] = score_sumz2
    feat["score_phys_max"] = score_phys_max
    # Backwards compat (existing snakemake rule + plot fns reference this)
    feat["outlier_score"] = score_max

    # Join with reco on (source_file, run_number, event_number).
    # source_file format differs between writers (feat: absolute path,
    # reco: basename like "run123"), so normalize to basename on both
    # sides before merging.
    if "source_file" in feat.columns:
        feat = feat.copy()
        feat["source_file"] = feat["source_file"].astype(str).map(os.path.basename)
    if "source_file" in reco.columns:
        reco = reco.copy()
        reco["source_file"] = reco["source_file"].astype(str).map(os.path.basename)
    join_keys = [k for k in ("source_file", "run_number", "event_number")
                 if k in feat.columns and k in reco.columns]
    if not join_keys:
        raise SystemExit(
            "No common join keys (need source_file/run_number/event_number)"
        )
    print(f"  joining on {join_keys}")
    merged = feat.merge(reco, on=join_keys, how="inner",
                        suffixes=("_feat", "_reco"))
    print(f"  merged shape: {merged.shape}")

    # Reco zenith in PA frame
    rho = pd.to_numeric(merged["peak_0_rho"], errors="coerce").to_numpy()
    z_reco = pd.to_numeric(merged["peak_0_z"], errors="coerce").to_numpy()
    zen_deg = np.degrees(np.arctan2(rho, z_reco - PA_Z_M))

    # Impulsivity proxy
    imp_col = find_impulsivity_column(merged)
    if imp_col is None:
        print("  WARNING: no impulsivity column found; skipping impulsivity plot")
        imp = None
    else:
        imp = pd.to_numeric(merged[imp_col], errors="coerce").to_numpy()
        print(f"  impulsivity column: {imp_col}")

    sc = merged["outlier_score"].to_numpy()

    # Plots
    plot_score_dist(
        sc,
        os.path.join(args.output_dir, f"outlier_{args.label}_score_distribution.png"),
        args.label,
    )
    plot_score_vs_x(
        sc, zen_deg, "Reconstructed zenith [deg]",
        os.path.join(args.output_dir, f"outlier_{args.label}_vs_reco_zenith.png"),
        args.label,
        x_range=(0, 180),
        add_secondary_x_deg_to_rad=True,
    )
    if imp is not None:
        plot_score_vs_x(
            sc, imp, f"{imp_col}",
            os.path.join(args.output_dir, f"outlier_{args.label}_vs_impulsivity.png"),
            args.label,
        )

    # Spearman across multiple outlier statistics so a non-monotonic
    # relationship doesn't get hidden by a single bad summary statistic.
    def _spearman(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 100:
            return float("nan")
        return float(pd.Series(a[m]).corr(pd.Series(b[m]), method="spearman"))

    metrics = {
        "max|z| (all)": merged["score_max"].to_numpy(),
        "max|z| (phys)": merged["score_phys_max"].to_numpy(),
        "count(|z|>5)": merged["score_count5"].to_numpy(),
        "count(|z|>10)": merged["score_count10"].to_numpy(),
        "sum(z^2)": merged["score_sumz2"].to_numpy(),
    }
    summary_lines = [
        f"# outlier-correlation summary ({args.label})",
        f"events scored:           {len(sc):,}",
        f"merged with reco:        {len(merged):,}",
        f"feature_cols (all): {len(feature_cols)}, physical: {len(phys_cols)}",
        f"max|z| median/p95/p99: "
        f"{np.nanmedian(sc):.2f} / {np.nanpercentile(sc, 95):.2f} / "
        f"{np.nanpercentile(sc, 99):.2f}",
        "",
        "Spearman correlations (per metric):",
    ]
    for name, vals in metrics.items():
        summary_lines.append(
            f"  {name:18s}  vs zenith    : {_spearman(vals, zen_deg):+.3f}"
            + (f"   vs impulsivity: {_spearman(vals, imp):+.3f}" if imp is not None else "")
        )

    # Top-1% outliers: are they spatially / impulsiveness-clustered
    # independent of any monotonic trend?
    cut = np.nanpercentile(sc, 99)
    topm = sc > cut
    summary_lines += [
        "",
        f"top-1% outliers (max|z| > {cut:.1f}): {topm.sum()} events",
    ]
    if topm.sum() > 50:
        summary_lines.append(
            f"  zenith of top-1%: median={np.nanmedian(zen_deg[topm]):.1f}deg, "
            f"p25={np.nanpercentile(zen_deg[topm], 25):.1f}, "
            f"p75={np.nanpercentile(zen_deg[topm], 75):.1f}"
        )
        summary_lines.append(
            f"  zenith of all:    median={np.nanmedian(zen_deg):.1f}deg, "
            f"p25={np.nanpercentile(zen_deg, 25):.1f}, "
            f"p75={np.nanpercentile(zen_deg, 75):.1f}"
        )
        if imp is not None:
            summary_lines.append(
                f"  impulsivity of top-1%: median={np.nanmedian(imp[topm]):.3f}, "
                f"p25={np.nanpercentile(imp[topm], 25):.3f}, "
                f"p75={np.nanpercentile(imp[topm], 75):.3f}"
            )
            summary_lines.append(
                f"  impulsivity of all:    median={np.nanmedian(imp):.3f}, "
                f"p25={np.nanpercentile(imp, 25):.3f}, "
                f"p75={np.nanpercentile(imp, 75):.3f}"
            )
    summary_path = os.path.join(args.output_dir, f"outlier_{args.label}_summary.txt")
    with open(summary_path, "w") as fh:
        fh.write("\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
