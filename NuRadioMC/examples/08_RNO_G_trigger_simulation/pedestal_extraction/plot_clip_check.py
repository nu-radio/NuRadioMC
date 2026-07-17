"""Plot the shipped ADC clip bounds against the measured pedestals behind them.

Reads the per-run pedestal npz written by pedestal_analysis.py
(pedestals_mv: n_runs x 24) and a clip-thresholds YAML, and shows:

- the per-run pedestal distributions of the trigger channels with the median
  each clip bound derives from;
- the shipped bounds against bounds re-derived from this npz's medians, for
  all channels in the YAML (diagonal = agreement);
- the mechanism: a loud waveform clipped at the asymmetric bounds.
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
    """Write clip_check.png next to this script (or --output)."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", required=True,
                   help="pedestal_analysis_results.npz from pedestal_analysis.py")
    p.add_argument("--clip_yaml", required=True)
    p.add_argument("--channels", type=int, nargs="*", default=[0, 1, 2, 3])
    p.add_argument("--output", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "clip_check.png"))
    args = p.parse_args()

    peds = np.load(args.npz)["pedestals_mv"]
    y = yaml.safe_load(open(args.clip_yaml))
    key = "clip_thresholds_mV" if "clip_thresholds_mV" in y else "clip_thresholds_mv"
    bounds = {int(k): [float(v[0]), float(v[1])] for k, v in y[key].items()}
    st = y.get("metadata", {}).get("station_id", "?")

    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))

    for k, ch in enumerate(args.channels):
        pv = peds[:, ch]
        med = float(np.median(pv))
        axs[0].hist(pv, bins=40, density=True, histtype="step", lw=2.0,
                    color=f"C{k}", label=f"ch{ch} (median {med:.0f} mV)")
        axs[0].axvline(med, color=f"C{k}", ls=":", lw=1.5)
    axs[0].set_title(f"Station {st}: per-run pedestals ({peds.shape[0]} runs)")
    axs[0].set_xlabel("Pedestal (mV)")
    axs[0].set_ylabel("Density")
    axs[0].legend(fontsize=10)

    ship, derived = [], []
    for ch, (lo, hi) in bounds.items():
        med = float(np.median(peds[:, ch]))
        ship += [lo, hi]
        derived += [-med, 2500 - med]
    lim = [min(ship + derived) - 60, max(ship + derived) + 60]
    axs[1].plot(lim, lim, color="0.5", lw=1.0)
    axs[1].plot(derived[::2], ship[::2], "o", ms=8, color="C0",
                label="clip- (per channel)")
    axs[1].plot(derived[1::2], ship[1::2], "s", ms=8, color="C1",
                label="clip+ (per channel)")
    axs[1].set_title("Shipped bounds vs bounds from this npz's medians")
    axs[1].set_xlabel("Re-derived bound (mV)")
    axs[1].set_ylabel("Shipped bound (mV)")
    axs[1].legend(fontsize=10)

    ch0 = args.channels[0]
    lo, hi = bounds[ch0]
    t = np.linspace(0, 1, 2000)
    wf = 1800 * np.sin(2 * np.pi * 12 * t) * np.exp(-((t - 0.5) / 0.22) ** 2)
    axs[2].plot(wf, lw=1.2, color="0.6", label="waveform before clip")
    axs[2].plot(np.clip(wf, lo, hi), lw=2.0, color="C0", label="after clip")
    axs[2].axhline(lo, color="C3", ls="--", lw=2.0, label=f"clip- {lo:.0f} mV")
    axs[2].axhline(hi, color="C3", ls=":", lw=2.0, label=f"clip+ {hi:.0f} mV")
    axs[2].set_title(f"Asymmetric saturation, ch{ch0}")
    axs[2].set_xlabel("Sample")
    axs[2].set_ylabel("Amplitude (mV)")
    axs[2].legend(fontsize=10, loc="lower right")
    for ax in axs:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
