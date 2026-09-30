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

Reco zenith is computed in the PA frame: ``arctan2(rho, z - PA_Z)`` with the
PA depth taken from the detector description; the truth zenith uses the station's
absolute position from the same description, because the simulation vertices are in
the absolute array frame.

Outputs four PNGs:
  - sim_zenith_error_hist.png   1D histogram of reco - truth, deg
  - sim_zenith_2d.png           2D hist of reco vs truth, deg
  - sim_zenith_abs_error.png    1D histogram of |reco - truth|, deg
  - sim_truth_zenith_hist.png   1D histogram of truth PA->vertex zenith, deg
"""
import argparse
import datetime
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from NuRadioReco.detector.RNO_G import rnog_detector

PA_CHANNELS = (0, 1, 2, 3)


def pa_position(station, source, detector_file, date):
    """Phased-array centre: absolute (x, y, z) and station-local depth from the detector description.

    The simulation vertices are in the absolute array frame, so the truth zenith needs the
    station's absolute position; the reconstruction works in the station-local frame, so the
    reco zenith needs only the depth. Both come from the mean of the four PA channels.
    """
    kwargs = dict(log_level=logging.ERROR, select_stations=[station])
    if source == "rnog_file":
        kwargs["detector_file"] = detector_file
    det = rnog_detector.Detector(**kwargs)
    det.update(datetime.datetime.fromisoformat(date))
    origin = np.asarray(det.get_absolute_position(station), float)
    rel = np.mean([np.asarray(det.get_relative_position(station, ch), float) for ch in PA_CHANNELS], axis=0)
    return origin + rel, float(rel[2])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="combined_event_variables.h5 (any station; we filter to source='sim')")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--label", default="sim")
    p.add_argument("--station", type=int, required=True)
    p.add_argument("--detector-source", default="rnog_mongo", choices=["rnog_mongo", "rnog_file"])
    p.add_argument("--detector-file", default=None, help="MongoDB export, used with --detector-source rnog_file")
    p.add_argument("--detector-date", default="2022-10-01")
    args = p.parse_args()
    pa_abs, pa_depth = pa_position(args.station, args.detector_source, args.detector_file, args.detector_date)
    print(f"station {args.station}: PA absolute position {np.round(pa_abs, 2).tolist()}, local depth {pa_depth:.2f} m")

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_hdf(args.input)
    if "source" in df.columns:
        df = df[df["source"] == "sim"]
    print(f"Loaded {len(df):,} sim rows from {args.input}")

    # Use the GEOMETRIC PA->vertex zenith as truth, NOT the thrown primary
    # zenith stored in `truth_zenith` (which is the neutrino arrival
    # direction, capped at the sim's thetamax and not what the antennas see).
    keep = df[["peak_0_rho", "peak_0_z", "truth_vx", "truth_vy", "truth_vz"]].dropna()
    rho = keep["peak_0_rho"].to_numpy()
    z = keep["peak_0_z"].to_numpy()
    dx = keep["truth_vx"].to_numpy() - pa_abs[0]
    dy = keep["truth_vy"].to_numpy() - pa_abs[1]
    dz = keep["truth_vz"].to_numpy() - pa_abs[2]
    truth_rad = np.arctan2(np.sqrt(dx**2 + dy**2), dz)
    reco_rad = np.arctan2(rho, z - pa_depth)

    err_deg = np.degrees(reco_rad - truth_rad)
    truth_deg = np.degrees(truth_rad)
    reco_deg = np.degrees(reco_rad)
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
    ax.set_yscale("log")
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
    ax.set_yscale("log")
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
