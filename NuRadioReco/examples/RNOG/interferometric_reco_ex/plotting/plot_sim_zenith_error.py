"""Sim-only reco-vs-truth zenith comparison plots.

Operates on a per-station ``combined_event_variables.h5`` and selects
``source == 'sim'`` rows.

Truth zenith here is the GEOMETRIC angle from the PA center to the
interaction VERTEX (PA at sim absolute coords; vertex from
``truth_vx, truth_vy, truth_vz``). This matches what the
interferometric reco actually targets -- NOT the thrown primary
zenith ``truth_zenith`` (which is the source neutrino's arrival
direction, capped at the sim's ``thetamax`` and unrelated to the
PA->vertex geometric angle).

The PA-centred geometry must already be present in the table. This script
never loads detector information or recalculates reconstruction quantities.

Outputs four PNGs:
  - sim_zenith_error_hist.png   1D histogram of reco - truth, deg
  - sim_zenith_2d.png           2D hist of reco vs truth, deg
  - sim_zenith_abs_error.png    1D histogram of |reco - truth|, deg
  - sim_truth_zenith_hist.png   1D histogram of truth PA->vertex zenith, deg
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="combined_event_variables.h5 (any station; we filter to source='sim')")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--label", default="sim")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_hdf(args.input)
    if "source" in df.columns:
        df = df[df["source"] == "sim"]
    print(f"Loaded {len(df):,} sim rows from {args.input}")

    required = {"reco_zenith_pa", "truth_zenith_pa", "zenith_error_pa"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            "combined table is missing saved PA-geometry columns: "
            + ", ".join(missing)
        )
    keep = df[sorted(required)].apply(pd.to_numeric, errors="coerce").dropna()
    if keep.empty:
        raise RuntimeError("no simulation rows have complete PA-zenith values")

    reco_deg = keep["reco_zenith_pa"].to_numpy()
    truth_deg = keep["truth_zenith_pa"].to_numpy()
    err_deg = keep["zenith_error_pa"].to_numpy()
    print("  using saved reco_zenith_pa, truth_zenith_pa, zenith_error_pa")
    print(f"  events with all three fields: {len(keep):,}")
    print(f"  err median / p16 / p84 / p95 (deg): "
          f"{np.median(err_deg):+.2f} / {np.percentile(err_deg, 16):+.2f} / "
          f"{np.percentile(err_deg, 84):+.2f} / {np.percentile(err_deg, 95):+.2f}")
    print(f"  |err| median / p68 / p90 (deg): "
          f"{np.median(np.abs(err_deg)):.2f} / "
          f"{np.percentile(np.abs(err_deg), 68):.2f} / "
          f"{np.percentile(np.abs(err_deg), 90):.2f}")

    # 1D signed-error hist
    rng = max(abs(np.percentile(err_deg, 0.5)), abs(np.percentile(err_deg, 99.5)))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(err_deg, bins=120, range=(-rng, rng), histtype="step", linewidth=1.4)
    for q, color in [(50, "#27ae60"), (16, "#888"), (84, "#888")]:
        v = np.percentile(err_deg, q)
        ax.axvline(v, color=color, linestyle="--", linewidth=1,
                   label=f"p{q}={v:+.2f} deg")
    ax.set_xlabel("Zenith error: reco - truth [deg]")
    ax.set_ylabel("Events")
    ax.set_title(f"{args.label}: zenith error (N={len(err_deg):,})")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    secax = ax.secondary_xaxis(
        "top", functions=(np.deg2rad, np.rad2deg))
    secax.set_xlabel("[rad]")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "sim_zenith_error_hist.png"), dpi=140)
    plt.close(fig)

    # 1D abs-error hist
    fig, ax = plt.subplots(figsize=(7, 4.5))
    abs_err = np.abs(err_deg)
    abs_max = np.percentile(abs_err, 99.5)
    ax.hist(abs_err, bins=120, range=(0, abs_max), histtype="step", linewidth=1.4)
    for q, color in [(50, "#27ae60"), (68, "#f39c12"), (90, "#e74c3c")]:
        v = np.percentile(abs_err, q)
        ax.axvline(v, color=color, linestyle="--", linewidth=1,
                   label=f"p{q}={v:.2f} deg")
    ax.set_xlabel("|Zenith error| = |reco - truth| [deg]")
    ax.set_ylabel("Events")
    ax.set_title(f"{args.label}: absolute zenith error (N={len(abs_err):,})")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    secax = ax.secondary_xaxis(
        "top", functions=(np.deg2rad, np.rad2deg))
    secax.set_xlabel("[rad]")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "sim_zenith_abs_error.png"), dpi=140)
    plt.close(fig)

    # 2D reco vs truth
    fig, ax = plt.subplots(figsize=(7, 5.5))
    h, xedges, yedges, im = ax.hist2d(
        truth_deg, reco_deg, bins=(90, 90),
        cmap="viridis", cmin=1,
    )
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Events / bin")
    lo = min(truth_deg.min(), reco_deg.min())
    hi = max(truth_deg.max(), reco_deg.max())
    diag = np.linspace(lo, hi, 50)
    ax.plot(diag, diag, color="red", linestyle="--", linewidth=1.5,
            label="reco = truth")
    ax.set_xlabel("Truth zenith from PA->vertex [deg]")
    ax.set_ylabel("Reco zenith (PA frame) [deg]")
    ax.set_title(f"{args.label}: reco vs truth zenith (N={len(truth_deg):,})")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    secx = ax.secondary_xaxis(
        "top", functions=(np.deg2rad, np.rad2deg))
    secx.set_xlabel("Truth [rad]")
    secy = ax.secondary_yaxis(
        "right", functions=(np.deg2rad, np.rad2deg))
    secy.set_ylabel("Reco [rad]")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "sim_zenith_2d.png"), dpi=140)
    plt.close(fig)

    # 1D truth PA->vertex zenith
    fig, ax = plt.subplots(figsize=(7, 4.5))
    tmax = np.percentile(truth_deg, 99.5)
    ax.hist(truth_deg, bins=120, range=(0, tmax), histtype="step", linewidth=1.4)
    for q, color in [(50, "#27ae60"), (16, "#888"), (84, "#888")]:
        v = np.percentile(truth_deg, q)
        ax.axvline(v, color=color, linestyle="--", linewidth=1,
                   label=f"p{q}={v:.2f} deg")
    ax.set_xlabel("Truth zenith from PA->vertex [deg]")
    ax.set_ylabel("Events")
    ax.set_title(f"{args.label}: truth PA->vertex zenith (N={len(truth_deg):,})")
    ax.grid(alpha=0.3)
    ax.legend()
    secax = ax.secondary_xaxis(
        "top", functions=(np.deg2rad, np.rad2deg))
    secax.set_xlabel("[rad]")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "sim_truth_zenith_hist.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
