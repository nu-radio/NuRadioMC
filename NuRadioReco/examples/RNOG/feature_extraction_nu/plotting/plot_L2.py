import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import yaml
import copy
import matplotlib.colors as colors

parser = argparse.ArgumentParser()
parser.add_argument("--indir")
parser.add_argument("--outdir")
parser.add_argument("--plot-config")

args = parser.parse_args()

cfg = yaml.safe_load(open(args.plot_config))

df = pd.read_hdf(args.indir, key="events")

datasets = {
    "burn": df[df["source"] == "burn"],
    "sim": df[df["source"] == "sim"],
}


def build_bins(bin_cfg):
    """Convert bin config into numpy bin edges."""
    mode = bin_cfg.get("mode", "linear")
    start = bin_cfg["start"]
    stop = bin_cfg["stop"]
    nbins = bin_cfg["nbins"]

    if mode == "linear":
        return np.linspace(start, stop, nbins + 1)

    elif mode == "log":
        return np.logspace(np.log10(start), np.log10(stop), nbins + 1)

    else:
        raise ValueError(f"Unknown bin mode: {mode}")


for dataset_name, data in datasets.items():

    for plot in cfg["plotting"]["plots"]:

        name = plot["name"]
        ptype = plot["type"]

        fig, ax = plt.subplots(figsize=(8, 6))

        # -------------------------
        # 2D histogram
        # -------------------------
        if ptype == "hist2d":

            x_bins = build_bins(plot["x_bins"])
            y_bins = build_bins(plot["y_bins"])
            
            cmap = copy.copy(plt.cm.viridis)
            cmap.set_under("white")
                        
            h = ax.hist2d(
                data[plot["x"]],
                data[plot["y"]],
                bins=[x_bins, y_bins],
                cmap=cmap,
                norm=colors.LogNorm(vmin=1)
            )

            fig.colorbar(h[3], ax=ax)

            ax.set_xlabel(plot["x"])
            ax.set_ylabel(plot["y"])

            # optional x limits
            if "x_lim" in plot:
                ax.set_xlim(plot["x_bins"]["start"], plot["x_lim"])

            # optional y limits
            if "y_lim" in plot:
                ax.set_ylim(plot["y_bins"]["start"], plot["y_lim"])

        # -------------------------
        # 1D histogram
        # -------------------------
        elif ptype == "hist1d":

            x_bins = build_bins(plot["x_bins"])

            ax.hist(
                data[plot["var"]],
                bins=x_bins,
            )

            ax.set_xlabel(plot["var"])

            # optional x limits
            if "x_lim" in plot:
                ax.set_xlim(plot["x_bins"]["start"], plot["x_lim"])

        else:
            raise ValueError(f"Unknown plot type: {ptype}")

        ax.set_title(f"{dataset_name}: {name}")

        # -------------------------
        # scales
        # -------------------------
        if plot.get("x_scale"):
            ax.set_xscale(plot["x_scale"])
        if plot.get("y_scale"):
            ax.set_yscale(plot["y_scale"])

        plt.tight_layout()

        plt.savefig(
            os.path.join(args.outdir, f"{dataset_name}_{name}.png")
        )

        plt.close()
