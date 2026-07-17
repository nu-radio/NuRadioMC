"""Plot the shipped trigger-Vrms values against the measured distributions behind them.

Reads the per-event measurement npz written by measure_trigger_vrms_full.py
(vrms_ch{0..3} per event, runmean_ch{0..3} per run) and a trigger-vrms YAML,
and shows, per trigger channel, the shipped value sitting on the measured
per-event trigger-path RMS distribution with the per-run means overlaid.
"""
import argparse
import os

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11,
})


def main():
    """Write vrms_check.png next to this script (or --output)."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", required=True,
                   help="measurement npz from measure_trigger_vrms_full.py")
    p.add_argument("--vrms_yaml", required=True)
    p.add_argument("--output", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "vrms_check.png"))
    args = p.parse_args()

    d = np.load(args.npz)
    y = yaml.safe_load(open(args.vrms_yaml))
    vals = {int(k): float(v) for k, v in y["trigger_vrms_V"].items()}
    st = y.get("metadata", {}).get("station_id", "?")

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    for ch, ax in zip(sorted(vals), axs.ravel()):
        ev = d[f"vrms_ch{ch}"] * 1e3
        rm = d[f"runmean_ch{ch}"] * 1e3
        v = vals[ch] * 1e3
        med = float(np.median(ev))
        bins = np.linspace(np.percentile(ev, 0.2), np.percentile(ev, 99.8), 60)
        ax.hist(ev, bins=bins, density=True, histtype="stepfilled", alpha=0.45,
                color="C0", label=f"per-event ({len(ev)})")
        ax.hist(rm, bins=bins, density=True, histtype="step", lw=2.0,
                color="C2", label=f"per-run mean ({len(rm)})")
        ax.axvline(v, color="C3", lw=2.4,
                   label=f"shipped {v:.2f} mV ({v / med:.3f} x median)")
        ax.set_title(f"Station {st} ch{ch}: trigger-path noise RMS")
        ax.set_xlabel("RMS (mV)")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
