"""Generate summary panels from a merged 3D reco HDF5.

Input is one of the `station{S}_merged_3d_reco.h5` files produced by
`merge_reco3d_chunks.py`. Output is a set of PNGs tagged with the dataset
label ("burn" or "sim") so Snakemake can embed them in the report.

Plots:
- peak_0_z distribution               (reconstructed depth)
- peak_0_rho distribution             (reconstructed radial distance)
- peak_0_phi distribution             (reconstructed azimuth)
- peak_0_corr distribution            (map-peak correlation)
- peak_0_map_snr distribution         (map SNR at the peak)
- max_corr distribution               (best correlation across pols/multiray)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_ANGLE_LABEL_DEG = {
    "peak_phi": (r"Reconstructed $\phi$ [deg]", r"$\phi$ [rad]"),
    "peak_zen": ("Reconstructed zenith [deg]", "zenith [rad]"),
    "surf_zen": ("Surface-correlator zenith [deg]", "zenith [rad]"),
    "surf_zen_vpol": ("Surface-correlator zenith VPOL [deg]", "zenith [rad]"),
    "surf_zen_hpol": ("Surface-correlator zenith HPOL [deg]", "zenith [rad]"),
}

PANELS = [
    ("peak_0_z", "Reconstructed $z$ [m]", "peak_z"),
    ("peak_0_rho", r"Reconstructed $\rho$ [m]", "peak_rho"),
    ("peak_0_phi", _ANGLE_LABEL_DEG["peak_phi"][0], "peak_phi"),
    ("peak_0_corr", "Peak correlation", "peak_corr"),
    ("peak_0_map_snr", "Map SNR at peak", "peak_map_snr"),
    ("max_corr", "Best correlation across pols/multiray", "max_corr"),
    # Zenith of the reco vertex direction from the detector origin,
    # derived from peak_0_rho and peak_0_z. Computed below before
    # the panel loop. 0 rad = straight down (vertex directly below
    # detector); pi/2 = horizontal.
    ("peak_0_zen", _ANGLE_LABEL_DEG["peak_zen"][0], "peak_zen"),
    # Surface-correlator zenith columns (different reconstruction
    # quantity than peak_0_zen above; kept for completeness).
    ("surf_corr_zen", _ANGLE_LABEL_DEG["surf_zen"][0], "surf_zen"),
    ("surf_corr_zen_vpol", _ANGLE_LABEL_DEG["surf_zen_vpol"][0], "surf_zen_vpol"),
    ("surf_corr_zen_hpol", _ANGLE_LABEL_DEG["surf_zen_hpol"][0], "surf_zen_hpol"),
]


def plot_hist(df, column, xlabel, out_path, label, stem=None):
    """Plot a single-column histogram from df and save to out_path.

    For angle panels (those whose stem appears in ``_ANGLE_LABEL_DEG``)
    the primary x-axis is in DEGREES (data is converted from radians
    on the fly) and a secondary top axis shows the corresponding
    radians. Non-angle panels keep the data's native units on the
    bottom axis with no top axis.
    """
    vals = pd.to_numeric(df[column], errors="coerce").dropna().to_numpy()
    finite = vals[np.isfinite(vals)]

    is_angle = stem in _ANGLE_LABEL_DEG
    if is_angle:
        # Reco output IS already in degrees for these columns. Only
        # apply rad->deg conversion if the data range looks like radians
        # (max <= ~7) to be backwards-tolerant of any column that does
        # come in radians.
        finite_max = float(np.nanmax(np.abs(finite))) if len(finite) else 0.0
        if finite_max <= 2 * np.pi + 0.1:
            plot_vals = np.degrees(finite)
        else:
            plot_vals = finite
        bottom_label = _ANGLE_LABEL_DEG[stem][0]
        top_label = _ANGLE_LABEL_DEG[stem][1]
    else:
        plot_vals = finite
        bottom_label = xlabel
        top_label = None

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(plot_vals, bins=80, histtype="step", linewidth=1.4)
    ax.set_xlabel(bottom_label)
    ax.set_ylabel("Events")
    ax.set_yscale("log")
    ax.set_title(f"{label}: {column} (N = {len(finite):,})")
    ax.grid(alpha=0.3, which="both")

    if is_angle:
        # Top axis: convert displayed-degrees back to radians.
        secax = ax.secondary_xaxis(
            "top",
            functions=(np.deg2rad, np.rad2deg),
        )
        secax.set_xlabel(top_label)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--label", required=True,
                   help="Dataset label (e.g. 'burn' or 'sim') used in plot titles and filenames.")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_hdf(args.input, key="data")
    print(f"Loaded {len(df):,} events from {args.input}", flush=True)

    # Derive zenith from PA center (origin of incoming RF direction),
    # NOT from the reco's (rho=0, z=0) reference. The reco frame has
    # z=0 at the ice surface and z negative subsurface; the PA center
    # sits at z ~ -95 m. For a vertex at (rho, z), the line from the
    # PA to the vertex has vertical run (z - z_PA) and horizontal run
    # rho, so zenith from straight-up at the PA is arctan(rho/(z-z_PA)).
    # A vertex at (0, 0) is 95 m straight above the PA -> zen = 0.
    # A vertex at (rho=95, z=0) is 45 deg from vertical -> 45 deg.
    PA_Z_M = -95.0
    if "peak_0_zen" not in df.columns and {"peak_0_rho", "peak_0_z"} <= set(df.columns):
        rho = pd.to_numeric(df["peak_0_rho"], errors="coerce").to_numpy()
        z = pd.to_numeric(df["peak_0_z"], errors="coerce").to_numpy()
        df["peak_0_zen"] = np.arctan2(rho, z - PA_Z_M)

    for column, xlabel, stem in PANELS:
        if column not in df.columns:
            print(f"  skip {column}: not in dataframe")
            continue
        out = os.path.join(args.output_dir, f"reco_{args.label}_{stem}.png")
        plot_hist(df, column, xlabel, out, args.label, stem=stem)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
