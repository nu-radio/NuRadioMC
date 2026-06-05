import yaml
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

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

for dataset_name, data in datasets.items():

    for plot in cfg["plotting"]["plots"]:

        name = plot["name"]
        ptype = plot["type"]

        fig, ax = plt.subplots(figsize=(8, 6))

        if ptype == "hist2d":

            h = ax.hist2d(
                data[plot["x"]],
                data[plot["y"]],
                bins=plot["x_bins"]["nbins"]
            )

            fig.colorbar(h[3], ax=ax)

            ax.set_xlabel(plot["x"])
            ax.set_ylabel(plot["y"])

        elif ptype == "hist1d":

            ax.hist(
                data[plot["var"]],
                bins=plot["x_bins"]["nbins"]
            )

            ax.set_xlabel(plot["var"])

        else:
            raise ValueError(f"Unknown plot type: {ptype}")

        ax.set_title(f"{dataset_name}: {name}")

        # scales if present
        if plot.get("x_scale"):
            ax.set_xscale(plot["x_scale"])
        if plot.get("y_scale"):
            ax.set_yscale(plot["y_scale"])

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                args.outdir,
                f"{dataset_name}_{name}.png"
            )
        )

        plt.close()
