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
- surface-region maximum correlations (overall, VPOL, HPOL)

PA-centred geometric quantities are intentionally not calculated here. They
are plotted from the canonical combined event table by ``plot_pa_geometry.py``.
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
}

PANELS = [
    ("peak_0_z", "Reconstructed $z$ [m]", "peak_z"),
    ("peak_0_rho", r"Reconstructed $\rho$ [m]", "peak_rho"),
    ("peak_0_phi", _ANGLE_LABEL_DEG["peak_phi"][0], "peak_phi"),
    ("peak_0_corr", "Peak correlation", "peak_corr"),
    ("peak_0_map_snr", "Map SNR at peak", "peak_map_snr"),
    ("max_corr", "Best correlation across pols/multiray", "max_corr"),
    # Despite their names, these are maximum correlation values inside a
    # configured surface-like zenith region, not angles.
    ("surf_corr_zen", "Maximum correlation in surface-zenith region", "surf_zen"),
    ("surf_corr_zen_vpol", "Maximum VPOL correlation in surface-zenith region", "surf_zen_vpol"),
    ("surf_corr_zen_hpol", "Maximum HPOL correlation in surface-zenith region", "surf_zen_hpol"),
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

    for column, xlabel, stem in PANELS:
        if column not in df.columns:
            print(f"  skip {column}: not in dataframe")
            continue
        out = os.path.join(args.output_dir, f"reco_{args.label}_{stem}.png")
        plot_hist(df, column, xlabel, out, args.label, stem=stem)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
