import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

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

        passed_snr = snr[df_src[cut]]

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

orders = {
    "Wind→Airplane→Rate": [
        "wind_passed",
        "airplane_passed",
        "intrarun_rate_passed",
    ],

    "Airplane→Wind→Rate": [
        "airplane_passed",
        "wind_passed",
        "intrarun_rate_passed",
    ],

    "Rate→Wind→Airplane": [
        "intrarun_rate_passed",
        "wind_passed",
        "airplane_passed",
    ],
}

results = {}

for name, order in orders.items():

    mask = np.ones(len(df_src), dtype=bool)

    for cut in order:
        mask &= df_src[cut]

    results[name] = mask.mean()

plt.bar(
    results.keys(),
    results.values()
)

plt.ylabel("Final surviving fraction")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(
            os.path.join(
                args.outdir,
                "cut_order_analysis_cuts.png"))

plt.close()



