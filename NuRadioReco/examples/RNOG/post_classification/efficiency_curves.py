import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from itertools import permutations
import copy
import matplotlib.colors as colors
from matplotlib.colors import LogNorm

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
    "spatiotemporal_passed", 
    "solar_flare_passed",
    "min_depth_sc_cut_passed",
    #"glitch_cut_passed",
]

bins = np.linspace(
    cuts["csw_snr_PA"].min(),
    cuts["csw_snr_PA"].max(),
    25
)

fig, axes = plt.subplots(
    2, 1,
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
    "solar_flare_passed",
    "min_depth_sc_cut_passed",
    #"glitch_cut_passed",
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
    "Solar",
    "Min Depth SC",
    #"Glitch",
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
    "solar_flare_passed",
    "min_depth_sc_cut_passed",
    #"glitch_cut_passed",
]

frac_removed = [
    1 - cuts[c].astype(bool).mean()
    for c in cut_names
]

plt.figure(figsize=(7, 4))

plt.bar(
    ["Wind", "Airplane", "Intrarun Rate", "Spatiotemporal", "Solar", "Min Depth SC"],
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

x_bins = np.linspace(0, 15, 50)
y_bins = np.linspace(0, 1, 50)
"""
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
    
    cmap = copy.copy(plt.cm.viridis)
    cmap.set_under("white")

    ax.hist2d(
        cuts.loc[mask, xvar],
        cuts.loc[mask, yvar],
        bins=[x_bins, y_bins],
        cmap=cmap,
        norm=colors.LogNorm(vmin=1)
    )

    n_surviving = mask.sum()
    ax.set_xlim(0, 15)

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

    cmap = copy.copy(plt.cm.viridis)
    cmap.set_under("white")

    ax.hist2d(
        cuts.loc[mask, xvar],
        cuts.loc[mask, yvar],
        bins=[x_bins, y_bins],
        cmap=cmap,
        norm=colors.LogNorm(vmin=1)
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
"""

fig, axes = plt.subplots(
    2,
    len(cut_names),
    figsize=(4 * len(cut_names), 8),
    sharex=True,
    sharey=True,
)

if len(cut_names) == 1:
    axes = axes.reshape(2, 1)

pretty_names = {
    "wind_passed": "Wind",
    "airplane_passed": "Airplane",
    "intrarun_rate_passed": "Intrarun Rate",
    "spatiotemporal_passed": "Spatiotemporal",
    "solar_flare_passed": "Solar",
    "min_depth_sc_cut_passed": "Min Depth SC",
    #"glitch_cut_passed": "Glitch",
}

for row, source in enumerate(["burn", "sim"]):

    df = cuts[cuts["source"] == source]

    for col, cut in enumerate(cut_names):

        ax = axes[row, col]

        mask = df[cut].astype(bool)

        cmap = copy.copy(plt.cm.viridis)
        cmap.set_under("white")

        ax.hist2d(
            df.loc[mask, xvar],
            df.loc[mask, yvar],
            bins=[x_bins, y_bins],
            cmap=cmap,
            norm=colors.LogNorm(vmin=1),
        )

        n_surviving = mask.sum()

        ax.set_xlim(0, 15)

        ax.set_title(
            f"{pretty_names.get(cut, cut)}\n"
            f"{source.capitalize()}  N={n_surviving:,}"
        )

        if row == 1:
            ax.set_xlabel("csw_snr_PA")

        if col == 0:
            ax.set_ylabel(f"{source.capitalize()}\nmax_corr")

plt.tight_layout()

plt.savefig(
    os.path.join(args.outdir, "maxcorr_vs_snr_after_cuts.png"),
    dpi=300,
)

plt.close()

fig, axes = plt.subplots(
    2,
    len(cut_names),
    figsize=(4 * len(cut_names), 8),
    sharex=True,
    sharey=True,
)

if len(cut_names) == 1:
    axes = axes.reshape(2, 1)

pretty_names = {
    "wind_passed": "Wind",
    "airplane_passed": "Airplane",
    "intrarun_rate_passed": "Intrarun Rate",
    "spatiotemporal_passed": "Spatiotemporal",
    "solar_flare_passed": "Solar",
    "min_depth_sc_cut_passed": "Min Depth SC",
    #"glitch_cut_passed": "Glitch",
}

xvar = "theta"
yvar = "surf_corr_ratio"

x_bins = np.linspace(-1.5708, 1.5708, 50)
y_bins = np.linspace(0, 1, 50)


for row, source in enumerate(["burn", "sim"]):

    df = cuts[cuts["source"] == source]

    for col, cut in enumerate(cut_names):

        ax = axes[row, col]

        mask = df[cut].astype(bool)

        cmap = copy.copy(plt.cm.viridis)
        cmap.set_under("white")

        ax.hist2d(
            df.loc[mask, xvar],
            df.loc[mask, yvar],
            bins=[x_bins, y_bins],
            cmap=cmap,
            norm=colors.LogNorm(vmin=1),
        )

        n_surviving = mask.sum()

        ax.set_title(
            f"{pretty_names.get(cut, cut)}\n"
            f"{source.capitalize()}  N={n_surviving:,}"
        )

        if row == 1:
            ax.set_xlabel("theta")

        if col == 0:
            ax.set_ylabel(
                f"{source.capitalize()}\n"
                "surf_corr_ratio"
            )

plt.tight_layout()

plt.savefig(
    os.path.join(args.outdir, "surf_ratio_vs_theta_after_cuts.png"),
    dpi=300,
)

plt.close()


mask = (
    (cuts["source"] == "burn")
    & (cuts["spatiotemporal_passed"].astype(bool))
    & (cuts["solar_flare_passed"].astype(bool))
    & (cuts["airplane_passed"].astype(bool))
    & (cuts["wind_passed"].astype(bool))
    & (cuts["min_depth_sc_cut_passed"].astype(bool))
    #& (cuts["glitch_cut_passed"].astype(bool))
    & (
        (cuts["max_corr"] > 0.05)
        | (cuts["csw_snr_PA"] >= 6.5)
    )
)

outliers = cuts.loc[
    mask,
    ["run_num", "event_id", "csw_snr_PA", "max_corr", "wind_speed", "trigger_times", "phi", "min_depth_z", "max_surf_corr", "bad_channels"]
].to_csv(
    os.path.join(args.outdir, "spatiotemporal_outliers.csv"),
    index=False,
)

selected = cuts.loc[mask]

fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

# Theta-Phi distribution
h1 = axs[0].hist2d(
    selected["theta"],
    selected["phi"],
    bins=100,
    cmap="viridis",
)
fig.colorbar(h1[3], ax=axs[0], label="Counts")
axs[0].set_xlabel(r"$\theta$")
axs[0].set_ylabel(r"$\phi$")
axs[0].set_title("Theta-Phi Distribution")

# R-Z distribution
h2 = axs[1].hist2d(
    selected["r"],
    selected["z"],
    bins=100,
    cmap="viridis",
)
fig.colorbar(h2[3], ax=axs[1], label="Counts")
axs[1].set_xlabel("R (m)")
axs[1].set_ylabel("Z (m)")
axs[1].set_title("R-Z Distribution")

plt.savefig(
    os.path.join(args.outdir, "spatiotemporal_outlier_distributions.png"),
    dpi=300,
)
plt.close(fig)


mask_2 = (
    (cuts["source"] == "burn")
    & (~cuts["spatiotemporal_passed"].astype(bool)))


df = cuts.loc[mask_2]

bad = df[~np.isfinite(df["trigger_times"])]
print(len(bad))

fig, axes = plt.subplots(
    1,
    3,
    figsize=(10, 4),
)

cmap = copy.copy(plt.cm.viridis)
cmap.set_under("white")

plots = [
    ("phi", "theta", "Theta vs Phi"),
    ("trigger_times", "phi", "Phi vs Trigger Time"),
    ("trigger_times", "theta", "Theta vs Trigger Time")
]

for ax, (xvar, yvar, title) in zip(axes, plots):
    
    plot_mask = (
        np.isfinite(df[xvar])
        & np.isfinite(df[yvar])
    )

    ax.hist2d(
        df.loc[plot_mask, xvar],
        df.loc[plot_mask, yvar],
        bins=100,
        cmap=cmap,
        norm=colors.LogNorm(vmin=1),
    )

    ax.set_xlabel(xvar)
    ax.set_ylabel(yvar)

    ax.set_title(
        f"{title}\n"
        f"N={len(df):,}"
    )

plt.tight_layout()
plt.savefig(
    os.path.join(args.outdir, "spatiotemporal_distributions.png"),
    dpi=300,
)

plt.close()


bad_run_counts = (
    cuts.loc[mask_2]
    .groupby("run_num")
    .size()
    .sort_values(ascending=False)
)

runs_to_plot = bad_run_counts.head(6).index

fig, axes = plt.subplots(
    len(runs_to_plot),
    3,
    figsize=(10, 4 * len(runs_to_plot)),
)

axes = np.atleast_2d(axes)

cmap = copy.copy(plt.cm.viridis)
cmap.set_under("white")

plots = [
    ("phi", "theta", "Theta vs Phi"),
    ("trigger_times", "phi", "Phi vs Trigger Time"),
    ("trigger_times", "theta", "Theta vs Trigger Time")
]

for row, run_num in enumerate(runs_to_plot):

    # ALL events from this run
    run_df = cuts[
        (cuts["source"] == "burn")
        & (cuts["run_num"] == run_num)
    ]

    for col, (xvar, yvar, title) in enumerate(plots):

        ax = axes[row, col]

        plot_mask = (
            np.isfinite(run_df[xvar])
            & np.isfinite(run_df[yvar])
        )

        ax.hist2d(
            run_df.loc[plot_mask, xvar],
            run_df.loc[plot_mask, yvar],
            bins=100,
            cmap=cmap,
            norm=colors.LogNorm(vmin=1),
        )

        ax.set_xlabel(xvar)
        ax.set_ylabel(yvar)

        ax.set_title(
            f"Run {run_num}\n"
            f"{title}\n"
            f"N={plot_mask.sum():,}"
        )

plt.tight_layout()
plt.savefig(
    os.path.join(args.outdir, "st_run_distributions.png"),
    dpi=300,
)
plt.close()


mask = cuts["source"] == "burn"

df = cuts.loc[mask]

wind_mask = np.isfinite(df["wind_speed"])

wind_speed = df.loc[wind_mask, "wind_speed"]

print(f"Number of burn events with valid wind speed: {len(wind_speed)}")

plt.figure(figsize=(7, 4))

plt.hist(
    wind_speed,
    bins=100,
    histtype="step",
)

plt.xlabel("Wind Speed")
plt.ylabel("Number of Events")
plt.title(
    f"Wind Speed Distribution (Burn Events)\n"
    f"N={len(wind_speed):,}"
)

plt.tight_layout()
plt.yscale("log")
plt.savefig(
    os.path.join(args.outdir, "burn_wind_speed_distribution.png"),
    dpi=300,
)

plt.close()

import matplotlib.dates as mdates
from datetime import datetime, timezone

mask = cuts["source"] == "burn"
df = cuts.loc[mask]

time_mask = np.isfinite(df["trigger_times"])
trigger_times = df.loc[time_mask, "trigger_times"]

# Convert Unix timestamps to datetime objects (UTC)
trigger_datetimes = [
    datetime.fromtimestamp(ts, tz=timezone.utc)
    for ts in trigger_times
]

print(f"Number of burn events with valid trigger times: {len(trigger_datetimes)}")

fig, ax = plt.subplots(figsize=(8, 4))

ax.hist(
    trigger_datetimes,
    bins=100,
    histtype="step",
)

ax.set_xlabel("Trigger Time (UTC)")
ax.set_ylabel("Number of Events")
ax.set_title(
    f"Burn Event Trigger Time Distribution\n"
    f"N={len(trigger_datetimes):,}"
)

# Format the datetime axis
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(args.outdir, "burn_trigger_time_distribution.png"),
    dpi=300,
)

plt.close()

import matplotlib.dates as mdates
from datetime import datetime, timezone

mask = cuts["source"] == "burn"
df = cuts.loc[mask]

valid = (
    np.isfinite(df["trigger_times"])
    & np.isfinite(df["wind_speed"])
)

trigger_times = df.loc[valid, "trigger_times"].to_numpy()
wind_speed = df.loc[valid, "wind_speed"].to_numpy()

print(f"Number of burn events: {len(trigger_times)}")

# Convert Unix timestamps -> datetime -> Matplotlib date numbers
trigger_datetimes = [
    datetime.fromtimestamp(ts, tz=timezone.utc)
    for ts in trigger_times
]
trigger_dates = mdates.date2num(trigger_datetimes)

fig, ax = plt.subplots(figsize=(10, 5))

h = ax.hist2d(
    trigger_dates,
    wind_speed,
    bins=[100, 50],  # adjust as desired
    cmap="viridis",
)

plt.colorbar(h[3], ax=ax, label="Number of Events")

ax.set_xlabel("Trigger Time (UTC)")
ax.set_ylabel("Wind Speed")
ax.set_title("Burn Events: Wind Speed vs Trigger Time")

ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
fig.autofmt_xdate()

plt.tight_layout()
plt.savefig(
    os.path.join(args.outdir, "burn_wind_speed_vs_trigger_time.png"),
    dpi=300,
)
plt.close()


outlier_mask = (
    (cuts["source"] == "burn")
    & (cuts["spatiotemporal_passed"].astype(bool))
    & (cuts["solar_flare_passed"].astype(bool))
    & (cuts["airplane_passed"].astype(bool))
    & (cuts["wind_passed"].astype(bool))
    & (cuts["min_depth_sc_cut_passed"].astype(bool))
    #& (cuts["glitch_cut_passed"].astype(bool))
    & (
        (cuts["max_corr"] > 0.1)
        | (cuts["csw_snr_PA"] >= 8)
    )
)

burn = cuts.loc[cuts["source"] == "burn", "max_surf_corr"]
sim = cuts.loc[cuts["source"] == "sim", "max_surf_corr"]
outliers = cuts.loc[outlier_mask, "max_surf_corr"]

# Common bins
"""
bins = np.linspace(
    cuts["max_surf_corr"].min(),
    cuts["max_surf_corr"].max(),
    75,
)
"""
bins = np.linspace(
    sim.min(),
    sim.max(),
    75,
)

plt.figure(figsize=(8, 5))
"""
plt.hist(
    burn,
    bins=bins,
    #density=True,
    histtype="step",
    linewidth=2,
    label=f"Burn ({len(burn)})",
)
"""
plt.hist(
    sim,
    bins=bins,
    #density=True,
    histtype="step",
    linewidth=2,
    label=f"Simulation ({len(sim)})",
)
"""
plt.hist(
    outliers,
    bins=bins,
    density=True,
    histtype="step",
    linewidth=2,
    label=f"Outliers ({len(outliers)})",
)
"""
plt.xlabel("Surface Correlation")
plt.ylabel("Probability Density")
plt.title("Surface Correlation Distributions")
plt.legend()
plt.yscale("log")

plt.tight_layout()
plt.savefig(
    os.path.join(args.outdir, "surf_corr_distributions_sim.png"),
    dpi=300,
)
plt.close()

cuts = pd.read_hdf(args.input_cuts)

# Masks
burn = cuts.loc[cuts["source"] == "burn"]
sim = cuts.loc[cuts["source"] == "sim"]
outliers = cuts.loc[outlier_mask]

fig, ax = plt.subplots(figsize=(8, 6))

theta_bins = np.linspace(cuts["theta"].min(), cuts["theta"].max(), 100)
ratio_bins = np.linspace(
    cuts["surf_corr_ratio"].min(),
    cuts["surf_corr_ratio"].max(),
    100,
)

for df, color, label in [
    (burn, "blue", "Burn"),
    (sim, "green", "Simulation"),
    (outliers, "red", "Outliers"),
]:
    H, xedges, yedges = np.histogram2d(
        df["theta"],
        df["surf_corr_ratio"],
        bins=[theta_bins, ratio_bins],
    )

    # Normalize each distribution
    H = H / H.max() if H.max() > 0 else H

    # Mask empty bins
    H = np.ma.masked_where(H == 0, H)

    X, Y = np.meshgrid(
        xedges[:-1],
        yedges[:-1],
        indexing="ij",
    )

    ax.contourf(
        X,
        Y,
        H,
        levels=[0.1, 1.0],
        colors=[color],
        alpha=1.0,
    )

from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color="blue", label="Burn"),
    Patch(color="green", label="Simulation"),
    Patch(color="red", label="Outliers"),
])

ax.set_xlabel(r"$\theta$")
ax.set_ylabel("Surface Correlation Ratio")
ax.set_title("Surface Correlation Ratio vs Theta")

plt.tight_layout()
plt.savefig(
    os.path.join(args.outdir, "surf_corr_ratio_vs_theta_overlay.png"),
    dpi=300,
)
plt.close()

total_burn = (cuts["source"] == "burn").sum()

# Events passing ALL cuts
pass_mask = (
    (cuts["source"] == "burn")
    & (cuts["spatiotemporal_passed"].astype(bool))
    & (cuts["solar_flare_passed"].astype(bool))
    & (cuts["airplane_passed"].astype(bool))
    & (cuts["wind_passed"].astype(bool))
    & (cuts["min_depth_sc_cut_passed"].astype(bool))
    #& (cuts["glitch_cut_passed"].astype(bool))
)
passed = cuts.loc[pass_mask].dropna(
    subset=["csw_snr_PA", "max_surf_corr"]
)
from matplotlib.colors import LogNorm

plt.figure(figsize=(8, 6))

# Calculate histogram first
counts, xedges, yedges = np.histogram2d(
    passed["csw_snr_PA"],
    passed["max_corr"],
    bins=[50, 50],
)

# Mask zero-count bins
counts = np.ma.masked_where(counts == 0, counts)

# Copy colormap and make masked bins white
cmap = plt.cm.viridis.copy()
cmap.set_bad("white")

plt.pcolormesh(
    xedges,
    yedges,
    counts.T,
    cmap=cmap,
    norm=LogNorm(vmin=counts.min(), vmax=counts.max()),
    shading="auto",
)

cbar = plt.colorbar()
cbar.set_label("Number of events")

#plt.xlim(0,15)
#plt.ylim(0,1)
plt.xlabel("CSW SNR PA")
plt.ylabel("Max Correlation")
plt.title(f"Burn events passing all cuts Passed: ({len(passed)}), Total: {total_burn}")

plt.tight_layout()
plt.savefig(
    os.path.join(args.outdir, "max_corr_vs_snr_pass_all_cuts.png"),
    dpi=300,
)
plt.close()
import ast
pass_mask = (
    (cuts["source"] == "burn")
    & (cuts["spatiotemporal_passed"].astype(bool))
    & (cuts["solar_flare_passed"].astype(bool))
    & (cuts["airplane_passed"].astype(bool))
    & (cuts["wind_passed"].astype(bool))
    & (cuts["min_depth_sc_cut_passed"].astype(bool))
    #& (cuts["glitch_cut_passed"].astype(bool))
)

passed = cuts.loc[pass_mask].copy()

passed["bad_channels"] = passed["bad_channels"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

passed["n_bad_channels"] = passed["bad_channels"].apply(
    lambda x: len(x) if isinstance(x, (list, tuple, np.ndarray)) else 0
)

plt.figure(figsize=(8, 6))

plt.hist(
    passed["n_bad_channels"],
    bins=np.arange(passed["n_bad_channels"].max() + 2) - 0.5,
    edgecolor="black",
)

plt.xlabel("Number of bad channels")
plt.ylabel("Number of events")
plt.yscale("log")
plt.title(
    f"Bad Channel Distribution for Burn Events Passing All Cuts "
    f"Passed: ({len(passed)}), Total: {total_burn}"
)

plt.xticks(
    np.arange(passed["n_bad_channels"].max() + 1)
)

plt.tight_layout()

plt.savefig(
    os.path.join(args.outdir, "bad_channels_distribution_pass_all_cuts.png"),
    dpi=300,
)

plt.close()


outlier_mask = (
    (cuts["source"] == "burn")
    & (cuts["spatiotemporal_passed"].astype(bool))
    & (cuts["solar_flare_passed"].astype(bool))
    & (cuts["airplane_passed"].astype(bool))
    & (cuts["wind_passed"].astype(bool))
    & (cuts["min_depth_sc_cut_passed"].astype(bool))
    #& (cuts["has_glitch"] == False)
    & (cuts["csw_snr_PA"] > 6.5)
)

outlier_events = cuts.loc[
    outlier_mask,
    ["event_id", "run_num", "csw_snr_PA", "max_surf_corr"]
]

outliers = cuts.loc[outlier_mask].copy()

print(f"Number of events passing all cuts with CSW SNR PA > 6: {len(outlier_events)}")
print(outlier_events.to_string(index=False))


plt.figure(figsize=(8, 6))

counts, xedges, yedges = np.histogram2d(
    outliers["phi"],
    outliers["theta"],
    bins=[50, 50],
)

counts = np.ma.masked_where(counts == 0, counts)

cmap = plt.cm.viridis.copy()
cmap.set_bad("white")

plt.pcolormesh(
    xedges,
    yedges,
    counts.T,
    cmap=cmap,
    shading="auto",
)

cbar = plt.colorbar()
cbar.set_label("Number of events")

plt.xlabel(r"$\phi$")
plt.ylabel(r"$\theta$")
plt.title(f"Outliers: $\\theta$ vs $\\phi$ ({len(outliers)} events)")

plt.tight_layout()
plt.savefig(
    os.path.join(args.outdir, "snr_6_ang.png"),
    dpi=300,
)
plt.close()

plt.figure(figsize=(8, 6))

counts, xedges, yedges = np.histogram2d(
    outliers["r"],
    outliers["z"],
    bins=[50, 50],
)

counts = np.ma.masked_where(counts == 0, counts)

cmap = plt.cm.viridis.copy()
cmap.set_bad("white")

plt.pcolormesh(
    xedges,
    yedges,
    counts.T,
    cmap=cmap,
    shading="auto",
)

cbar = plt.colorbar()
cbar.set_label("Number of events")

plt.xlabel("r [m]")
plt.ylabel("z [m]")
plt.title(f"Outliers: z vs r ({len(outliers)} events)")

plt.tight_layout()
plt.savefig(
    os.path.join(args.outdir, "snr_6_rz.png"),
    dpi=300,
)
plt.close()

plt.figure(figsize=(8, 6))

counts, xedges, yedges = np.histogram2d(
    outliers["theta"],
    outliers["surf_corr_ratio"],
    bins=[50, 50],
)

counts = np.ma.masked_where(counts == 0, counts)

cmap = plt.cm.viridis.copy()
cmap.set_bad("white")

plt.pcolormesh(
    xedges,
    yedges,
    counts.T,
    cmap=cmap,
    shading="auto",
)

cbar = plt.colorbar()
cbar.set_label("Number of events")

plt.xlabel(r"$\theta$")
plt.ylabel("Surface correlation ratio")
plt.title(
    f"Outliers: surface correlation ratio vs $\\theta$ "
    f"({len(outliers)} events)"
)

plt.tight_layout()
plt.savefig(
    os.path.join(args.outdir, "snr_6_scr_theta.png"),
    dpi=300,
)
plt.close()

plt.figure(figsize=(8, 5))

plt.hist(
    outliers["min_depth_z"].dropna(),
    bins=40,
    histtype="step",
    linewidth=2,
)

plt.xlabel("Minimum depth z [m]")
plt.ylabel("Number of events")
plt.title(f"Outliers: minimum depth ({len(outliers)} events)")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(args.outdir, "snr_6_min_depth.png"),
    dpi=300,
)
plt.close()

outliers["trigger_datetime"] = pd.to_datetime(
    outliers["trigger_times"],
    unit="s",
    utc=True,
)


plt.figure(figsize=(10, 5))

plt.hist(
    outliers["trigger_datetime"],
    bins=30,
    histtype="step",
    linewidth=2,
)

plt.xlabel("Trigger time (UTC)")
plt.ylabel("Number of events")
plt.title(f"Outlier trigger-time distribution ({len(outliers)} events)")

plt.xticks(rotation=45)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(args.outdir, "snr_6_trig_time.png"),
    dpi=300,
)
plt.close()

burn_mask = (
    (cuts["source"] == "burn"))

burn_events_all = cuts.loc[
    burn_mask,
    ["event_id", "run_num", "glitch_ts"]
]

burn_events_all["run_num"] = (
    burn_events_all["run_num"]
    .astype(str)
    .str.replace("run", "", regex=False)
    .astype(int)
)

rows = []

for _, row in burn_events_all.iterrows():

    glitch_dict = row["glitch_ts"]

    # In case the column was read from CSV and is a string
    if isinstance(glitch_dict, str):
        glitch_dict = ast.literal_eval(glitch_dict)

    for channel, ts in glitch_dict.items():
        rows.append({
            "run_num": row["run_num"],
            "event_id": row["event_id"],
            "channel": int(channel),
            "glitch_ts": float(ts),
        })

N_EVENTS = 5

glitch_df = pd.DataFrame(rows)

glitch_df["sign"] = np.sign(glitch_df["glitch_ts"])

flips = []

for (run, channel), group in glitch_df.groupby(
    ["run_num", "channel"]
):

    group = group.sort_values("event_id").reset_index(drop=True)

    signs = group["sign"].to_numpy()

    for i in range(N_EVENTS, len(group) - N_EVENTS):

        before = signs[i-N_EVENTS:i]
        after = signs[i:i+N_EVENTS]

        # Negative -> positive
        if (
            np.all(before < 0)
            and np.all(after > 0)
        ):
            flips.append({
                "run_num": run,
                "channel": channel,
                "previous_event_id": group.loc[i-1, "event_id"],
                "event_id": group.loc[i, "event_id"],
                "previous_glitch_ts": group.loc[i-1, "glitch_ts"],
                "glitch_ts": group.loc[i, "glitch_ts"],
                "flip": "negative -> positive",
            })
        elif (
            np.all(before > 0)
            and np.all(after < 0)
        ):
            flips.append({
                "run_num": run,
                "channel": channel,
                "previous_event_id": group.loc[i-1, "event_id"],
                "event_id": group.loc[i, "event_id"],
                "previous_glitch_ts": group.loc[i-1, "glitch_ts"],
                "glitch_ts": group.loc[i, "glitch_ts"],
                "flip": "positive -> negative",
            })

flips = pd.DataFrame(flips)

print(flips)



"""
fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

datasets = [
    (burn, "Burn", "Blues"),
    (sim, "Simulation", "Greens"),
    (outliers, "Outliers", "Reds"),
]

for ax, (df, label, cmap) in zip(axs, datasets):

    h = ax.hist2d(
        df["theta"],
        df["surf_corr_ratio"],
        bins=[theta_bins, ratio_bins],
        cmap=cmap,
        norm=LogNorm(),
    )

    fig.colorbar(h[3], ax=ax, label="Counts")

    ax.set_title(label)
    ax.set_xlabel(r"$\theta$")

axs[0].set_ylabel("Surface Correlation Ratio")

plt.tight_layout()

plt.savefig(
    os.path.join(args.outdir, "surf_corr_ratio_theta_three_histograms.png"),
    dpi=300,
)
plt.close()

fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

snr_bins = np.linspace(cuts["csw_snr_PA"].min(), 15, 100)


datasets = [
    (burn, "Burn", "Blues"),
    (sim, "Simulation", "Greens"),
    (outliers, "Outliers", "Reds"),
]

for ax, (df, label, cmap) in zip(axs, datasets):

    h = ax.hist2d(
        df["csw_snr_PA"],
        df["surf_corr_ratio"],
        bins=[snr_bins, ratio_bins],
        cmap=cmap,
        norm=LogNorm(),
    )

    fig.colorbar(h[3], ax=ax, label="Counts")

    ax.set_title(label)
    ax.set_xlabel("CSW SNR PA")

axs[0].set_ylabel("Surface Correlation Ratio")

plt.tight_layout()

plt.savefig(
    os.path.join(args.outdir, "surf_corr_ratio_snr_three_histograms.png"),
    dpi=300,
)
plt.close()

fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

snr_bins = np.linspace(cuts["csw_snr_PA"].min(), 15, 100)

surf_corr_bins = np.linspace(
    cuts["max_surf_corr"].min(),
    cuts["max_surf_corr"].max(),
    100,
)

datasets = [
    (burn, "Burn", "Blues"),
    (sim, "Simulation", "Greens"),
    (outliers, "Outliers", "Reds"),
]

for ax, (df, label, cmap) in zip(axs, datasets):

    h = ax.hist2d(
        df["max_surf_corr"],
        df["surf_corr_ratio"],
        bins=[surf_corr_bins, ratio_bins],
        cmap=cmap,
        norm=LogNorm(),
    )

    fig.colorbar(h[3], ax=ax, label="Counts")

    ax.set_title(label)
    ax.set_xlabel("Max Surf Correlation")

axs[0].set_ylabel("Surface Correlation Ratio")

plt.tight_layout()

plt.savefig(
    os.path.join(args.outdir, "surf_corr_ratio_surf_corr_three_histograms.png"),
    dpi=300,
)
plt.close()


sim_theta_positive = sim.loc[sim["theta"] > 0]
sim_theta_positive[
    ["run_num", "event_id", "theta", "surf_corr_ratio"]
].to_csv(
    os.path.join(args.outdir, "sim_theta_positive.csv"),
    index=False,
)

surface_mask = (
    (cuts["source"] == "burn")
    & (
        (~cuts["solar_flare_passed"].astype(bool))
        | (~cuts["airplane_passed"].astype(bool))
    )
)

fig, ax = plt.subplots(figsize=(6, 5))

surface = cuts.loc[surface_mask]

h = ax.hist2d(
    surface["theta"],
    surface["surf_corr_ratio"],
    bins=[theta_bins, ratio_bins],
    cmap="Blues",
    norm=LogNorm(vmin=1),
)

fig.colorbar(h[3], ax=ax, label="Counts")

ax.set_title("Surface")
ax.set_xlabel(r"$\theta$")
ax.set_ylabel("Surface Correlation Ratio")

plt.tight_layout()

plt.savefig(
    os.path.join(args.outdir, "surf_corr_ratio_theta_solar_airplane.png"),
    dpi=300,
)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5))

surface = cuts.loc[surface_mask]

h = ax.hist2d(
    surface["csw_snr_PA"],
    surface["surf_corr_ratio"],
    bins=[snr_bins, ratio_bins],
    cmap="Blues",
    norm=LogNorm(vmin=1),
)

fig.colorbar(h[3], ax=ax, label="Counts")

ax.set_title("Surface")
ax.set_xlabel("SNR")
ax.set_ylabel("Surface Correlation Ratio")

plt.tight_layout()

plt.savefig(
    os.path.join(args.outdir, "surf_corr_ratio_snr_solar_airplane.png"),
    dpi=300,
)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5))

ax.hist(
    sim["surf_corr_ratio"],
    bins=100,
    histtype="step",
    linewidth=2,
)

ax.set_xlabel("Surface Correlation Ratio")
ax.set_ylabel("Counts")
ax.set_title("Surface Correlation Ratio Projection")

plt.tight_layout()

plt.savefig(
    os.path.join(args.outdir, "surf_corr_ratio_projection_sim.png"),
    dpi=300,
)

plt.close(fig)


burn_filtered = burn[burn["max_surf_corr"] > 0.0416]
sim_filtered = sim[sim["max_surf_corr"] > 0.0416]

fig, ax = plt.subplots(figsize=(6, 5))

ax.hist(
    sim["max_corr"],
    bins=100,
    histtype="step",
    linewidth=2,
)

ax.set_xlabel("Max Correlation")
ax.set_ylabel("Counts")
ax.set_title("Max Correlation")

plt.tight_layout()
plt.yscale("log")

plt.savefig(
    os.path.join(args.outdir, "max_corr_sim.png"),
    dpi=300,
)

plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5))

ax.hist(
    burn["max_corr"],
    bins=100,
    histtype="step",
    linewidth=2,
)

ax.set_xlabel("Max Correlation")
ax.set_ylabel("Counts")
ax.set_title("Max Correlation")

plt.tight_layout()
plt.yscale("log")
plt.savefig(
    os.path.join(args.outdir, "max_corr_burn.png"),
    dpi=300,
)

plt.close(fig)



fig, ax = plt.subplots(figsize=(6, 5))

ax.hist(
    burn_filtered["min_depth_z"],
    bins=100,
    histtype="step",
    linewidth=2,
)

ax.set_xlabel("Min Depth Z")
ax.set_ylabel("Counts")
ax.set_title("Min Depth Z")

plt.tight_layout()
plt.yscale("log")
plt.savefig(
    os.path.join(args.outdir, "min_depth_z_burn_cut.png"),
    dpi=300,
)

plt.close(fig)
"""
"""
fig, ax = plt.subplots(figsize=(6, 5))

ax.hist(
    sim_filtered["min_depth_z"],
    bins=100,
    histtype="step",
    linewidth=2,
)

ax.set_xlabel("Min Depth Z")
ax.set_ylabel("Counts")
ax.set_title("Min Depth Z")

plt.tight_layout()
plt.yscale("log")
plt.savefig(
    os.path.join(args.outdir, "min_depth_z_sim_cut.png"),
    dpi=300,
)

plt.close(fig)

depth = sim_filtered["min_depth_z"].dropna().values

# Sort values
depth_sorted = np.sort(depth)

# Cumulative fraction
cdf = np.arange(1, len(depth_sorted) + 1) / len(depth_sorted)

fig, ax = plt.subplots(figsize=(6, 5))

ax.plot(
    depth_sorted,
    cdf,
    linewidth=2,
)

ax.set_xlabel("Min Depth Z")
ax.set_ylabel("CDF")
ax.set_title("Min Depth Z CDF")

ax.set_ylim(0, 1)

plt.tight_layout()

plt.savefig(
    os.path.join(args.outdir, "min_depth_z_sim_cut_cdf.png"),
    dpi=300,
)

plt.close(fig)


burn_z = burn_filtered["min_depth_z"].dropna().values
sim_z = sim_filtered["min_depth_z"].dropna().values

fig, ax = plt.subplots(figsize=(7, 5))

ax.hist(
    burn_z,
    bins=100,
    density=True,
    histtype="step",
    linewidth=2,
    label="Burn",
)

ax.hist(
    sim_z,
    bins=100,
    density=True,
    histtype="step",
    linewidth=2,
    label="Simulation",
)

ax.set_xlabel("Min Depth Z")
ax.set_ylabel("Normalized density")
ax.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(args.outdir, "min_depth_z_both_cut.png"),
    dpi=300,
)

plt.close(fig)

burn_boundary = burn_z == 300
burn_continuous = burn_z < 300

sim_boundary = sim_z == 300
sim_continuous = sim_z < 300

print(
    f"Burn at boundary: "
    f"{np.sum(burn_boundary)}/{len(burn_z)} "
    f"= {np.mean(burn_boundary):.3f}"
)

print(
    f"Sim at boundary: "
    f"{np.sum(sim_boundary)}/{len(sim_z)} "
    f"= {np.mean(sim_boundary):.3f}"
)

cuts = np.linspace(
    min(burn_z.min(), sim_z.min()),
    300,
    500,
)

burn_eff = np.array([
    np.mean(burn_z > cut)
    for cut in cuts
])

sim_eff = np.array([
    np.mean(sim_z > cut)
    for cut in cuts
])

burn_eff = np.array([
    np.mean(burn_z > cut)
    for cut in cuts
])

sim_eff = np.array([
    np.mean(sim_z > cut)
    for cut in cuts
])

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(
    cuts,
    burn_eff,
    label="Burn survival",
)

ax.plot(
    cuts,
    sim_eff,
    label="Simulation efficiency",
)

ax.set_xlabel("Min Depth Z cut")
ax.set_ylabel("Fraction surviving")
ax.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(args.outdir, "min_depth_z_frac_surv.png"),
    dpi=300,
)

plt.close(fig)

N_burn = len(burn_z)

expected_burn = N_burn * burn_eff

valid = expected_burn < 1

cut = cuts[valid][0]

idx = np.argmin(np.abs(expected_burn - 1))

cut = cuts[idx]

print(f"Cut = {cut:.2f}")
print(f"Expected burn = {expected_burn[idx]:.3f}")
print(f"Simulation efficiency = {sim_eff[idx]:.3f}")


# Events that don't surviving

burn_final = burn[
    (burn["max_surf_corr"] > 0.0416) &
    (burn["min_depth_z"] > 0)
]

sim_final = sim[
    (sim["max_surf_corr"] > 0.0416) &
    (sim["min_depth_z"] > 0)
]

# Total efficiencies relative to original samples
burn_eff = (len(burn) - len(burn_final)) / len(burn)
sim_eff = (len(sim) - len(sim_final)) / len(sim)

print(f"Burn:")
print(f"  Initial: {len(burn)}")
print(f"  Surviving both cuts: {len(burn) - len(burn_final)}")
print(f"  Total efficiency: {burn_eff:.4f} ({100*burn_eff:.2f}%)")
print(f"  Total efficiency loss: {100*(1-burn_eff):.2f}%")

print()

print(f"Simulation:")
print(f"  Initial: {len(sim)}")
print(f"  Surviving both cuts: {len(sim) - len(sim_final)}")
print(f"  Total efficiency: {sim_eff:.4f} ({100*sim_eff:.2f}%)")
print(f"  Total efficiency loss: {100*(1-sim_eff):.2f}%")


import numpy as np
import matplotlib.pyplot as plt

def calculate_efficiency(df, bins):

    # Keep only events with SNR between 0 and 15
    df = df[
        (df["csw_snr_PA"] >= 0) &
        (df["csw_snr_PA"] <= 15)
    ].copy()

    snr = df["csw_snr_PA"].values

    # Events inside the cut region = rejected
    rejected = (
        (df["max_surf_corr"] > 0.0416) &
        (df["min_depth_z"] > 0)
    ).values

    # Everything outside the cut region is retained
    passed = ~rejected

    bin_idx = np.digitize(snr, bins) - 1

    efficiency = []
    efficiency_err = []
    centers = []

    for i in range(len(bins) - 1):

        mask = bin_idx == i
        n_total = np.sum(mask)

        if n_total == 0:
            continue

        n_pass = np.sum(passed[mask])

        eff = n_pass / n_total

        # Binomial uncertainty
        err = np.sqrt(eff * (1 - eff) / n_total)

        efficiency.append(eff)
        efficiency_err.append(err)
        centers.append(
            0.5 * (bins[i] + bins[i + 1])
        )

    return (
        np.array(centers),
        np.array(efficiency),
        np.array(efficiency_err),
    )


# SNR bins from 0 to 15
snr_bins = np.linspace(0, 15, 16)

burn_snr, burn_eff, burn_err = calculate_efficiency(
    burn,
    snr_bins
)

sim_snr, sim_eff, sim_err = calculate_efficiency(
    sim,
    snr_bins
)

fig, ax = plt.subplots(figsize=(7, 5))

ax.errorbar(
    burn_snr,
    burn_eff,
    yerr=burn_err,
    fmt="o-",
    capsize=3,
    label="Burn",
)

ax.errorbar(
    sim_snr,
    sim_eff,
    yerr=sim_err,
    fmt="o-",
    capsize=3,
    label="Simulation",
)

ax.set_xlabel("CSW SNR PA")
ax.set_ylabel("Fraction retained")
ax.set_xlim(0, 15)
ax.set_ylim(0, 1.05)

ax.legend()

plt.tight_layout()
plt.savefig(
    os.path.join(args.outdir, "cut_vs_snr.png"),
    dpi=300,
)

plt.close(fig)


"""

"""
print("Burn events being plotted:")
print(
    burn_filtered[["event_id", "run_num", "min_depth_z"]].to_string(index=False)
)

print(f"\nTotal burn events: {len(burn_filtered)}")


print("\nSim events being plotted:")
print(
    sim_filtered[["event_id", "run_num", "min_depth_z"]].to_string(index=False)
)

print(f"\nTotal sim events: {len(sim_filtered)}")
"""



import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf

# SCR values
scr = burn["max_surf_corr"].dropna().values

# Sort data and construct empirical CDF
x = np.sort(scr)
y = np.arange(1, len(x) + 1) / len(x)

# Gaussian CDF
def gaussian_cdf(x, mu, sigma):
    return 0.5 * (1 + erf((x - mu) / (np.sqrt(2) * sigma)))

# Initial guesses
p0 = [
    np.mean(scr),
    np.std(scr),
]

# Fit
params, cov = curve_fit(
    gaussian_cdf,
    x,
    y,
    p0=p0,
)

mu, sigma = params

print(f"Mean = {mu:.5f}")
print(f"Sigma = {sigma:.5f}")

# Plot CDF fit
plt.figure(figsize=(7,5))

plt.plot(
    x,
    y,
    label="Empirical CDF",
)

x_fit = np.linspace(x.min(), x.max(), 500)

plt.plot(
    x_fit,
    gaussian_cdf(x_fit, mu, sigma),
    linewidth=2,
    label=rf"Gaussian CDF fit ($\mu={mu:.3f}, \sigma={sigma:.3f}$)",
)

plt.xlabel("Surface Correlation")
plt.ylabel("CDF")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(args.outdir, "sc_cdf_burn.png"),
    dpi=300,
)

plt.close(fig)

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

scr = burn["max_surf_corr"].dropna().values

# Histogram
counts, bins, _ = plt.hist(
    scr,
    bins=500,
    histtype="step",
    linewidth=2,
    label="SCR distribution",
)

bin_centers = 0.5 * (bins[:-1] + bins[1:])

def asymmetric_gaussian(x, A, mu, sigma_left, sigma_right):
    left = x < mu
    right = ~left

    y = np.zeros_like(x)

    y[left] = A * np.exp(
        -0.5 * ((x[left] - mu) / sigma_left)**2
    )

    y[right] = A * np.exp(
        -0.5 * ((x[right] - mu) / sigma_right)**2
    )

    return y


# Initial guesses
peak_idx = np.argmax(counts)

p0 = [
    counts.max(),              # amplitude
    bin_centers[peak_idx],     # mean
    np.std(scr[scr < bin_centers[peak_idx]]),   # left width
    np.std(scr[scr > bin_centers[peak_idx]])    # right width
]


counts_err = np.sqrt(np.maximum(counts, 1))


params, cov = curve_fit(
    asymmetric_gaussian,
    bin_centers,
    counts,
    p0=p0,
    sigma=counts_err,
    absolute_sigma=True,
    maxfev=10000
)


A, mu, sigma_left, sigma_right = params

print(f"A = {A:.2f}")
print(f"mu = {mu:.5f}")
print(f"sigma_left = {sigma_left:.5f}")
print(f"sigma_right = {sigma_right:.5f}")


# Plot
x_fit = np.linspace(scr.min(), scr.max(), 500)
"""
plt.step(
    bin_centers,
    counts,
    where="mid",
    label="Data"
)
"""
plt.plot(
    x_fit,
    asymmetric_gaussian(
        x_fit,
        A,
        mu,
        sigma_left,
        sigma_right
    ),
    lw=2,
    label=(
        rf"Asym Gaussian "
        rf"$\sigma_L={sigma_left:.3f}$, "
        rf"$\sigma_R={sigma_right:.3f}$"
    )
)

"""
plt.axvline(
    mu + 3*sigma,
    linestyle="--",
    label=rf"$\mu+3\sigma={mu+3*sigma:.3f}$",
)
"""
plt.yscale("log")
plt.ylim(1, counts.max() * 2)
plt.xlabel("Surface Correlation")
plt.ylabel("Counts")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(args.outdir, "sc_fit_burn.png"),
    dpi=300,
)

plt.close()



# SCR values

scr = burn["max_surf_corr"].dropna().values

# Histogram
counts, bins, _ = plt.hist(
    scr,
    bins=500,
    histtype="step",
    linewidth=2,
    label="SCR distribution",
)

bin_centers = 0.5 * (bins[:-1] + bins[1:])



import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Find peak
peak_idx = np.argmax(counts)
peak_height = counts[peak_idx]
peak_x = bin_centers[peak_idx]

# Find 50% height point on falling edge
half_height = 0.3 * peak_height

falling_edge = np.arange(peak_idx, len(counts))
half_idx = falling_edge[np.where(counts[falling_edge] <= half_height)[0][0]]

fit_start_x = bin_centers[half_idx]

print(f"Peak at SCR = {peak_x:.5f}")
print(f"Starting exponential fit at 50% height: SCR = {fit_start_x:.5f}")


# Exponential decay model
def exp_decay(x, A, lamb):
    return A * np.exp(-lamb * (x - fit_start_x))


# Select falling edge after 50% point
tail_mask = (bin_centers >= fit_start_x) & (counts > 0)

x_tail = bin_centers[tail_mask]
y_tail = counts[tail_mask]


# Fit
params, cov = curve_fit(
    exp_decay,
    x_tail,
    y_tail,
    p0=[
        half_height,
        10
    ],
    sigma=np.sqrt(y_tail),
    absolute_sigma=True,
    maxfev=10000
)

A, lamb = params

x0 = fit_start_x

# Desired expected number of events above the cut
N_expected = 1

bin_width = bins[1] - bins[0]

x_cut = x0 + np.log(A / (lamb * bin_width * N_expected)) / lamb

y_fit = exp_decay(x_tail, A, lamb)

residuals = y_tail - y_fit
y_err = np.sqrt(y_tail)

# Chi-square
chi2 = np.sum((residuals / y_err)**2)

# Degrees of freedom
ndof = len(y_tail) - len(params)

# Reduced chi-square
chi2_red = chi2 / ndof

# R^2
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y_tail - np.mean(y_tail))**2)

r_squared = 1 - ss_res / ss_tot

#x_cut = x0 + np.log(A / (lamb * N_expected)) / lamb

print(f"Exponential fit:")
print(f"  A              = {A:.3f}")
print(f"  Lambda         = {lamb:.5f}")
print(f"  Lambda error   = {np.sqrt(cov[1,1]):.5f}")
print()
print(f"Goodness of fit:")
print(f"  Chi-square      = {chi2:.2f}")
print(f"  NDF             = {ndof}")
print(f"  Reduced chi2    = {chi2_red:.2f}")
print(f"  R^2             = {r_squared:.4f}")
print()
print(f"Cut = {x_cut:.4f}")

print(f"Cut = {x_cut:.4f}")

print(f"A = {A:.2f}")
print(f"Lambda = {lamb:.5f}")


# Plot
x_fit = np.linspace(fit_start_x, x_tail.max(), 500)
"""
plt.step(
    bin_centers,
    counts,
    where="mid",
    label="SCR distribution"
)
"""
plt.plot(
    x_fit,
    exp_decay(x_fit, A, lamb),
    linewidth=2,
    label=rf"Exponential fit ($\lambda={lamb:.3f}$)"
)

plt.axvline(
    x_cut,
    color="red",
    linestyle="--",
    label=f"Expected tail events = {N_expected}"
)

"""
plt.axvline(
    fit_start_x,
    linestyle="--",
    label="50% peak point"
)
"""

plt.yscale("log")
plt.ylim(1, counts.max()*2)
plt.xlim(0.025, 0.05)
plt.xlabel("SCR")
plt.ylabel("Counts")
plt.legend()
plt.savefig(
    os.path.join(args.outdir, "sc_fit_burn_exp.png"),
    dpi=300,
)

plt.close(fig)
