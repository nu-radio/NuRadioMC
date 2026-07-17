"""Plot the FT-noise tiling: stitched trace, RMS profile, and seam-vs-random statistics.

Draws real forced-trigger events, builds trigger-copy noise traces exactly as
simulate.py does (upsample to the internal rate, stitch with the equal-power
crossfade), and shows:

- one stitched trace with the crossfade regions shaded;
- the per-sample RMS over many stitched traces, flat through seams and edges;
- the distribution of local RMS in crossfade windows vs random windows of the
  same width (with a KS test), and the ensemble RMS of every seam window
  against the distribution over random windows.

Together the bottom row is the quantitative statement that crossfade regions
are statistically identical to any other part of the trace.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from simulate import FTNoisePool, upsample_trace, tile_noise_overlap_add, TILE_OVERLAP

matplotlib.rcParams.update({
    "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11,
})


def main():
    """Write tiling_example.png next to this script."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ft_noise_dir", required=True)
    p.add_argument("--station", type=int, default=23)
    p.add_argument("--clean_mask", default=None)
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--n_traces", type=int, default=200)
    p.add_argument("--target_length", type=int, default=12000)
    p.add_argument("--upsample_factor", type=float, default=5.0 / 3.2)
    p.add_argument("--n_control_windows", type=int, default=2000)
    p.add_argument("--output", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "tiling_example.png"))
    args = p.parse_args()

    pool = FTNoisePool(args.ft_noise_dir, station_id=args.station, seed=0,
                       clean_mask_path=args.clean_mask)
    rng = np.random.default_rng(0)

    def stitched():
        tiles = []
        n = 0
        while n < args.target_length + TILE_OVERLAP:
            tr = pool.get_noise_event()[args.channel]
            up = upsample_trace(tr, int(round(len(tr) * args.upsample_factor)))
            tiles.append(up)
            n += len(up) - TILE_OVERLAP
        return tile_noise_overlap_add(tiles, args.target_length), len(tiles[0])

    traces = []
    for _ in range(args.n_traces):
        tr, n_tile = stitched()
        traces.append(tr)
    traces = np.array(traces) * 1e3
    w = TILE_OVERLAP
    seams = np.arange(n_tile - w, args.target_length - w, n_tile - w)

    seam_rms = np.array([traces[i, s:s + w].std()
                         for i in range(args.n_traces) for s in seams])
    starts = rng.integers(0, args.target_length - w, args.n_control_windows)
    idx = rng.integers(0, args.n_traces, args.n_control_windows)
    ctrl_rms = np.array([traces[i, s:s + w].std() for i, s in zip(idx, starts)])
    ks = stats.ks_2samp(seam_rms, ctrl_rms)
    ens_seam = np.array([traces[:, s:s + w].std() for s in seams])
    ens_ctrl = np.array([traces[:, s:s + w].std() for s in starts])

    fig = plt.figure(figsize=(14, 13))
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.25)
    ax_tr = fig.add_subplot(gs[0, :])
    ax_pr = fig.add_subplot(gs[1, :])
    ax_d1 = fig.add_subplot(gs[2, 0])
    ax_d2 = fig.add_subplot(gs[2, 1])

    ax_tr.plot(traces[0], lw=0.6, color="C0")
    for s in seams:
        ax_tr.axvspan(s, s + w, color="C1", alpha=0.25)
    ax_tr.set_title(f"Stitched trigger-copy noise trace, station {args.station} "
                    f"ch{args.channel} (crossfade regions shaded)")
    ax_tr.set_ylabel("Amplitude (mV)")

    rms = traces.std(axis=0)
    ax_pr.plot(rms, lw=2.0, color="C0",
               label=f"per-sample RMS over {args.n_traces} traces")
    ax_pr.axhline(float(np.median(rms)), color="C3", ls="--", lw=2.0,
                  label=f"median {np.median(rms):.2f} mV")
    for s in seams:
        ax_pr.axvspan(s, s + w, color="C1", alpha=0.25)
    ax_pr.set_ylim(0, 1.5 * float(np.median(rms)))
    ax_pr.set_title("Ensemble RMS profile: flat through seams and edges")
    ax_pr.set_ylabel("RMS (mV)")
    ax_pr.legend(loc="lower right")
    for ax in (ax_tr, ax_pr):
        ax.set_xlabel("Sample (internal rate)")
        ax.set_xlim(0, args.target_length)

    bins = np.linspace(min(ctrl_rms.min(), seam_rms.min()),
                       max(ctrl_rms.max(), seam_rms.max()), 40)
    ax_d1.hist(ctrl_rms, bins=bins, density=True, histtype="stepfilled",
               alpha=0.45, color="C0",
               label=f"random windows (n={len(ctrl_rms)})")
    ax_d1.hist(seam_rms, bins=bins, density=True, histtype="step", lw=2.2,
               color="C1", label=f"crossfade windows (n={len(seam_rms)})")
    ax_d1.set_title(f"Local window RMS: KS p = {ks.pvalue:.2f}")
    ax_d1.set_xlabel(f"RMS in a {w}-sample window (mV)")
    ax_d1.set_ylabel("Density")
    ax_d1.legend()

    ax_d2.hist(ens_ctrl, bins=30, density=True, histtype="stepfilled",
               alpha=0.45, color="C0",
               label=f"random windows (n={len(ens_ctrl)})")
    for k, r in enumerate(ens_seam):
        ax_d2.axvline(r, color="C1", lw=2.4,
                      label="crossfade windows" if k == 0 else None)
    ax_d2.set_title(f"Ensemble window RMS over {args.n_traces} traces")
    ax_d2.set_xlabel(f"Ensemble RMS in a {w}-sample window (mV)")
    ax_d2.set_ylabel("Density")
    ax_d2.legend()
    for ax in (ax_tr, ax_pr, ax_d1, ax_d2):
        ax.grid(True, alpha=0.3)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"wrote {args.output}")
    print(f"local RMS: crossfade {seam_rms.mean():.3f} vs random {ctrl_rms.mean():.3f} mV, "
          f"KS p = {ks.pvalue:.3f}")


if __name__ == "__main__":
    main()
