"""Plot saved PA-centred reconstruction zenith from a combined event table.

This script deliberately performs no detector lookup and no geometry
calculation. ``reco_zenith_pa`` is produced upstream by deep_cr_search and is
the single authoritative value used here.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                        help="combined_event_variables.h5")
    parser.add_argument("--source", required=True, choices=["burn", "sim"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dataframe = pd.read_hdf(args.input)
    if "source" not in dataframe.columns:
        raise RuntimeError("combined table is missing required column: source")
    if "reco_zenith_pa" not in dataframe.columns:
        raise RuntimeError(
            "combined table is missing saved column: reco_zenith_pa"
        )

    selected = dataframe.loc[dataframe["source"] == args.source]
    zenith = pd.to_numeric(
        selected["reco_zenith_pa"], errors="coerce"
    ).to_numpy()
    zenith = zenith[np.isfinite(zenith)]
    if not len(zenith):
        raise RuntimeError(
            f"no {args.source} rows have a finite reco_zenith_pa value"
        )

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.hist(zenith, bins=80, histtype="step", linewidth=1.4)
    axis.set_xlabel("Reconstructed zenith from PA center [deg]")
    axis.set_ylabel("Events")
    axis.set_yscale("log")
    axis.set_title(
        f"{args.label}: reco_zenith_pa (N = {len(zenith):,})"
    )
    axis.grid(alpha=0.3, which="both")
    secondary = axis.secondary_xaxis(
        "top", functions=(np.deg2rad, np.rad2deg)
    )
    secondary.set_xlabel("Zenith [rad]")
    figure.tight_layout()

    output = os.path.join(
        args.output_dir, f"reco_{args.label}_peak_zen.png"
    )
    figure.savefig(output, dpi=140)
    plt.close(figure)
    print(
        f"Loaded {len(zenith):,} {args.source} PA-zenith values from "
        f"{args.input}\n  wrote {output}"
    )


if __name__ == "__main__":
    main()
