import pandas as pd
import argparse
import matplotlib.pyplot as plt 
import os

parser = argparse.ArgumentParser()
parser.add_argument("--indir", required=True)
parser.add_argument("--outdir", required=True)
parser.add_argument("--x-vars", required=True)
parser.add_argument("--y-vars", required=True)
parser.add_argument("--names", required=True)

args = parser.parse_args()

x_vars = args.x_vars.split(",")
y_vars = args.y_vars.split(",")
names = args.names.split(",")

if not (len(x_vars) == len(y_vars) == len(names)):
    raise ValueError(
        "x_vars, y_vars, and names must have the same length"
    )

df = pd.read_hdf(args.indir, key="events")

datasets = {
    "burn": df[df["source"] == "burn"],
    "sim": df[df["source"] == "sim"],
}

for source_name, source_df in datasets.items():
    for x_var, y_var, plot_name in zip(x_vars, y_vars, names):

        fig, ax = plt.subplots(figsize=(8, 6))

        h = ax.hist2d(
            source_df[x_var],
            source_df[y_var],
            bins=50
        )

        fig.colorbar(h[3], ax=ax)

        ax.set_title(source_name.capitalize())
        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                args.outdir,
                f"{source_name}_{plot_name}.png"
            )
        )

        plt.close()
