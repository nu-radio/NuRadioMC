import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from itertools import permutations


parser = argparse.ArgumentParser()
parser.add_argument("--input_val")
parser.add_argument("--input_train")
parser.add_argument("--input_test")
parser.add_argument("--input_cuts")
parser.add_argument("--outdir")

args = parser.parse_args()

# Read files
dfs = [
    pd.read_hdf(args.input_train),
    pd.read_hdf(args.input_val),
    pd.read_hdf(args.input_test),
]

df = pd.concat(dfs, ignore_index=True)

snr = df["csw_snr_PA"]
passed = (df["Predicted_CR"] == 1).astype(int)

bins = np.linspace(snr.min(), snr.max(), 25)

total, edges = np.histogram(snr, bins=bins)
passed_hist, _ = np.histogram(
    snr[passed == 1],
    bins=bins
)

fraction = np.divide(
    passed_hist,
    total,
    out=np.zeros_like(passed_hist, dtype=float),
    where=total > 0
)

centers = 0.5 * (edges[:-1] + edges[1:])

plt.figure()
plt.plot(centers, fraction, marker="o")
plt.xlabel("csw_snr_PA")
plt.ylabel("Fraction Predicted_CR = 1")
plt.grid(True)
plt.tight_layout()
outfile = "efficiency_vs_snr_lda"
plt.savefig(
            os.path.join(
                args.outdir,
                f"{outfile}.png"))

plt.close()

cuts = pd.read_hdf(args.input_cuts)

cut_columns = [
    "wind_passed",
    "airplane_passed",
    "intrarun_rate_passed",
    "spatiotemporal_passed"
]

bins = np.linspace(
    cuts["csw_snr_PA"].min(),
    cuts["csw_snr_PA"].max(),
    25
)

fig, axes = plt.subplots(
    1, 2,
    figsize=(12,5),
    sharey=True
)

for ax, source in zip(axes, ["burn", "sim"]):

    df_src = cuts[cuts["source"] == source]

    snr = df_src["csw_snr_PA"]

    total, edges = np.histogram(
        snr,
        bins=bins
    )

    centers = 0.5 * (edges[:-1] + edges[1:])

    for cut in cut_columns:
        
        mask = df_src[cut].astype(bool)
        passed_snr = snr[mask]

        passed_hist, _ = np.histogram(
            passed_snr,
            bins=bins
        )

        frac = np.divide(
            passed_hist,
            total,
            out=np.zeros_like(passed_hist, dtype=float),
            where=total > 0
        )

        ax.plot(
            centers,
            frac,
            marker="o",
            label=cut
        )

    ax.set_title(source)
    ax.set_xlabel("csw_snr_PA")
    ax.grid(True)

axes[0].set_ylabel("Fraction passing")
axes[1].legend()

plt.tight_layout()
plt.savefig(
            os.path.join(
                args.outdir,
                "efficiency_vs_snr_analysis_cuts.png"))

plt.close()



df_sim = cuts[cuts["source"] == "sim"].copy()
energy = df_sim["energy"].astype(float)

bins = np.logspace(
    np.log10(energy.min()),
    np.log10(energy.max()),
    25,
)

total, edges = np.histogram(energy, bins=bins)
centers = np.sqrt(edges[:-1] * edges[1:])

plt.figure(figsize=(7, 5))

for cut in cut_columns:
    
    mask = df_sim[cut].astype(bool)
    passed_energy = energy[mask]
    
    passed_hist, _ = np.histogram(
        passed_energy,
        bins=bins,
    )

    frac = np.divide(
        passed_hist,
        total,
        out=np.zeros_like(passed_hist, dtype=float),
        where=total > 0,
    )

    plt.plot(
        centers,
        frac,
        marker="o",
        label=cut,
    )

plt.xscale("log")
plt.xlabel("Energy")
plt.ylabel("Fraction passing")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(
        args.outdir,
        "efficiency_vs_energy_analysis_cuts.png",
    )
)

plt.close()


cut_names = [
    "wind_passed",
    "airplane_passed",
    "intrarun_rate_passed",
    "spatiotemporal_passed",
]

orders = {}

for cut in cut_names:
    others = [c for c in cut_names if c != cut]

    # Cut first
    orders[f"{cut}_first"] = [cut] + others

    # Cut middle
    orders[f"{cut}_middle"] = [others[0], cut] + others[1:]

    # Cut last
    orders[f"{cut}_last"] = others + [cut]

results = {}

for name, order in orders.items():

    mask = np.ones(len(cuts), dtype=bool)

    removed_fracs = {}

    for cut in order:

        before = mask.sum()

        mask &= cuts[cut].astype(bool).to_numpy()

        after = mask.sum()

        removed_fracs[cut] = (before - after) / len(cuts)

    results[name] = removed_fracs



from matplotlib.patches import Patch

group_spacing = 4      # space allocated per cut
bar_positions = []
bar_labels = []
bar_values = []
bar_colors = []

color_map = {
    "first": "tab:blue",
    "middle": "tab:green",
    "last": "tab:orange",
}

for i, cut in enumerate(cut_names):
    base = i * group_spacing

    for j, pos in enumerate(["first", "middle", "last"]):
        key = f"{cut}_{pos}"

        bar_positions.append(base + j)
        bar_values.append(results[key][cut])
        bar_colors.append(color_map[pos])

    bar_labels.append(base + 1)

fig, ax = plt.subplots(figsize=(10, 6))

ax.barh(
    bar_positions,
    bar_values,
    color=bar_colors,
)

ax.set_yticks(bar_labels)
ax.set_yticklabels([
    "Wind",
    "Airplane",
    "Intrarun Rate",
    "Spatiotemporal",
])

ax.legend(handles=[
    Patch(color="tab:blue", label="Passed first"),
    Patch(color="tab:green", label="In sequence"),
    Patch(color="tab:orange", label="Passed last"),
])

ax.set_xlabel("Fraction of total events removed")

plt.tight_layout()
plt.savefig(
    os.path.join(args.outdir, "cut_order_analysis_cuts.png"),
    dpi=300,
)
plt.close()

cut_names = [
    "wind_passed",
    "airplane_passed",
    "intrarun_rate_passed",
    "spatiotemporal_passed",
]

frac_removed = [
    1 - cuts[c].astype(bool).mean()
    for c in cut_names
]

plt.figure(figsize=(7, 4))

plt.bar(
    ["Wind", "Airplane", "Intrarun Rate", "Spatiotemporal"],
    frac_removed,
)

plt.ylabel("Fraction of events removed")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(args.outdir, "cut_evt_frac.png"),
    dpi=300,
)

plt.close()

xvar = "csw_snr_PA"
yvar = "max_corr"

x_bins = np.linspace(0, 10, 50)
y_bins = np.linspace(0, 1, 50)

fig, axes = plt.subplots(
    1,
    len(cut_names),
    figsize=(4 * len(cut_names), 4),
    sharex=True,
    sharey=True,
)

if len(cut_names) == 1:
    axes = [axes]

pretty_names = {
    "wind_passed": "Wind",
    "airplane_passed": "Airplane",
    "intrarun_rate_passed": "Intrarun Rate",
    "spatiotemporal_passed": "Spatiotemporal",
}

for ax, cut in zip(axes, cut_names):

    mask = cuts[cut].astype(bool)

    ax.hist2d(
        cuts.loc[mask, xvar],
        cuts.loc[mask, yvar],
        bins=[x_bins, y_bins],
    )

    n_surviving = mask.sum()

    ax.set_title(
        f"{pretty_names.get(cut, cut)}\nN={n_surviving:,}"
    )
    ax.set_xlabel("csw_snr_PA")

axes[0].set_ylabel("max_corr")

plt.tight_layout()

plt.savefig(
    os.path.join(args.outdir, "maxcorr_vs_snr_after_cuts.png"),
    dpi=300,
)

plt.close()

xvar = "theta"
yvar = "surf_corr_ratio"

x_bins = np.linspace(-1.5708, 1.5708, 50)
y_bins = np.linspace(0, 1, 50)

fig, axes = plt.subplots(
    1,
    len(cut_names),
    figsize=(4 * len(cut_names), 4),
    sharex=True,
    sharey=True,
)

if len(cut_names) == 1:
    axes = [axes]

pretty_names = {
    "wind_passed": "Wind",
    "airplane_passed": "Airplane",
    "intrarun_rate_passed": "Intrarun Rate",
    "spatiotemporal_passed": "Spatiotemporal",
}

for ax, cut in zip(axes, cut_names):

    mask = cuts[cut].astype(bool)

    ax.hist2d(
        cuts.loc[mask, xvar],
        cuts.loc[mask, yvar],
        bins=[x_bins, y_bins],
    )

    n_surviving = mask.sum()

    ax.set_title(
        f"{pretty_names.get(cut, cut)}\nN={n_surviving:,}"
    )
    ax.set_xlabel("theta")

axes[0].set_ylabel("surf_corr_ratio")

plt.tight_layout()

plt.savefig(
    os.path.join(args.outdir, "surf_ratio_vs_theta_after_cuts.png"),
    dpi=300,
)
plt.close()


