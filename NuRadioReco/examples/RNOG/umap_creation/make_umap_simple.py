import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import umap
import yaml
import os
sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

parser = argparse.ArgumentParser()
parser.add_argument("--burn", required=True)
parser.add_argument("--config", required=True)
parser.add_argument("--outdir", required=True)
args = parser.parse_args()

burn = pd.read_csv(args.burn)

with open(args.config, "r") as f:
    config = yaml.safe_load(f)



reducer = umap.UMAP()
features = config["features"]

burn_data = burn[features]
scaled_burn = StandardScaler().fit_transform(burn_data)

embedding = umap.UMAP(
    n_neighbors=30,
    min_dist=0.1,
    random_state=42,
).fit_transform(scaled_burn)

plt.figure(figsize=(8,6))
plt.scatter(
    embedding[:,0],
    embedding[:,1],
    s=2,
    alpha=0.5
)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.savefig(
    os.path.join(args.outdir, "umap.png"),
    dpi=300,
)

plt.close()



