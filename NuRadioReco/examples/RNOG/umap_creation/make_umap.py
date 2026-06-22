import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import umap
import yaml
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import hdbscan

sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})

###############################################################################
# Args
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--burn", required=True)
parser.add_argument("--config", required=True)
parser.add_argument("--outdir", required=True)
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

###############################################################################
# Load data
###############################################################################

burn = pd.read_csv(args.burn)

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

features = config["features"]

###############################################################################
# Feature matrix
###############################################################################

X = burn[features].copy()

# keep only numeric
X = X.select_dtypes(include=np.number)

# drop constant / empty columns
good_cols = []
for c in X.columns:
    if X[c].isna().all():
        continue
    if X[c].nunique() <= 1:
        continue
    good_cols.append(c)

X = X[good_cols]

print(f"Using {len(good_cols)} features")

# fill NaNs
X = X.fillna(X.median())

###############################################################################
# Scale
###############################################################################

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

###############################################################################
# UMAP
###############################################################################

"""
embedding = umap.UMAP(
    n_neighbors=30,
    min_dist=0.1,
    random_state=42,
).fit_transform(X_scaled)
"""

embedding = umap.UMAP(
    n_neighbors=10,
    min_dist=0.0,
    random_state=42,
).fit_transform(X_scaled)

burn["umap_x"] = embedding[:, 0]
burn["umap_y"] = embedding[:, 1]

###############################################################################
# HDBSCAN clustering
###############################################################################
"""
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=100,
    min_samples=20
)
"""
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=100,
    min_samples=50,
)

burn["cluster"] = clusterer.fit_predict(embedding)

###############################################################################
# Save full output
###############################################################################

burn.to_csv(
    os.path.join(args.outdir, "burn_umap_clusters.csv"),
    index=False
)

###############################################################################
# Cluster summary
###############################################################################

summary = burn.groupby("cluster").agg(
    n_events=("cluster", "count"),
)

# optional physics flags (only if they exist)
for col in ["airplane_passed", "wind_passed", "spatiotemporal_passed", "intrarun_rate_passed"]:
    if col in burn.columns:
        summary[f"{col}_frac"] = burn.groupby("cluster")[col].mean()

summary.to_csv(
    os.path.join(args.outdir, "cluster_summary.csv")
)

print(summary)

###############################################################################
# UMAP plot colored by cluster
###############################################################################

PLOT_N = min(60000, len(burn))
plot_df = burn.sample(PLOT_N, random_state=42)

###############################################################################
# UMAP cluster plot
###############################################################################

plt.figure(figsize=(8, 6))

plt.scatter(
    plot_df["umap_x"],
    plot_df["umap_y"],
    c=plot_df["cluster"],
    s=3,
    cmap="tab20",
    alpha=0.7,
    edgecolors="none",
    rasterized=True,
)

plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP + HDBSCAN Clusters")

plt.tight_layout()
plt.savefig(
    os.path.join(args.outdir, "umap_clusters.png"),
    dpi=300,
)

plt.close()

###############################################################################
# Diagnostic plots (physics labels)
###############################################################################

labels = [
    "airplane_passed",
    "wind_passed",
    "intrarun_rate_passed",
    "spatiotemporal_passed",
]

for col in labels:
    if col not in burn.columns:
        continue

    plt.figure(figsize=(8, 6))

    # reuse same subsample for consistency
    sc = plt.scatter(
        plot_df["umap_x"],
        plot_df["umap_y"],
        c=plot_df[col],
        s=3,
        cmap="viridis",
        alpha=0.7,
        edgecolors="none",
        rasterized=True,
    )

    plt.colorbar(sc, label=col)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title(f"UMAP colored by {col}")

    plt.tight_layout()
    plt.savefig(
        os.path.join(args.outdir, f"umap_{col}.png"),
        dpi=300,
    )

    plt.close()



import numpy as np
import matplotlib.pyplot as plt
import math

###############################################################################
# Setup
###############################################################################

clusters = sorted(c for c in burn["cluster"].unique())

n_clusters = len(clusters)
ncols = 3
nrows = math.ceil(n_clusters / ncols)

###############################################################################
# 1) max_corr_vs_csw_snr
###############################################################################

fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(5 * ncols, 4 * nrows),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)

axes = np.atleast_1d(axes).ravel()

x_bins = np.linspace(0, 15, 50)
y_bins = np.linspace(0, 1, 50)

for ax, cluster in zip(axes, clusters):

    df = burn[burn["cluster"] == cluster]

    h = ax.hist2d(
        df["csw_snr_PA"],
        df["max_corr"],
        bins=[x_bins, y_bins],
        cmap="viridis",
    )

    ax.set_title(
        f"Cluster {cluster}\nN={len(df):,}"
    )

    ax.set_xlim(0, 15)
    ax.set_ylim(0, 1)

for ax in axes[len(clusters):]:
    ax.set_visible(False)

fig.supxlabel("csw_snr_PA")
fig.supylabel("max_corr")

cbar = fig.colorbar(
    h[3],
    ax=axes.tolist(),
    shrink=0.8,
)
cbar.set_label("Counts")
plt.savefig(
    os.path.join(args.outdir, "max_corr_vs_csw_snr_by_cluster.png"),
    dpi=300,
)
plt.close()

###############################################################################
# 2) surf_corr_ratio_vs_theta
###############################################################################

fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(5 * ncols, 4 * nrows),
    sharex=True,
    sharey=True,
    constrained_layout=True
)

axes = np.atleast_1d(axes).ravel()

x_bins = np.linspace(
    -np.pi / 2,
    np.pi / 2,
    50,
)

y_bins = np.linspace(
    0,
    1,
    50,
)

for ax, cluster in zip(axes, clusters):

    df = burn[burn["cluster"] == cluster]

    h = ax.hist2d(
        df["theta"],
        df["surf_corr_ratio"],
        bins=[x_bins, y_bins],
        cmap="viridis",
    )

    ax.set_title(
        f"Cluster {cluster}\nN={len(df):,}"
    )

    ax.set_ylim(0, 1)

for ax in axes[len(clusters):]:
    ax.set_visible(False)

fig.supxlabel("theta [rad]")
fig.supylabel("surf_corr_ratio")

cbar = fig.colorbar(
    h[3],
    ax=axes.tolist(),
    shrink=0.8,
)
cbar.set_label("Counts")
plt.savefig(
    os.path.join(args.outdir, "surf_corr_ratio_vs_theta_by_cluster.png"),
    dpi=300,
)
plt.close()

###############################################################################
# 3) snr_1d
###############################################################################

fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(5 * ncols, 4 * nrows),
    sharex=True,
    sharey=True,
)

axes = np.atleast_1d(axes).ravel()

bins = np.logspace(
    np.log10(1),
    np.log10(100),
    50,
)

for ax, cluster in zip(axes, clusters):

    df = burn[burn["cluster"] == cluster]

    ax.hist(
        df["csw_snr_PA"],
        bins=bins,
        histtype="stepfilled",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_title(
        f"Cluster {cluster}\nN={len(df):,}"
    )

for ax in axes[len(clusters):]:
    ax.set_visible(False)

fig.supxlabel("csw_snr_PA")
fig.supylabel("Counts")

plt.tight_layout()
plt.savefig(
    os.path.join(args.outdir, "snr_1d_by_cluster.png"),
    dpi=300,
)
plt.close()
