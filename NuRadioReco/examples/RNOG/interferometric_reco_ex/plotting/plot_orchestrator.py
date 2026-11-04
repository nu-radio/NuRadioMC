"""Reco plotting orchestrator. Dispatches enabled plots from a YAML config."""
import argparse
import os
import subprocess
import sys

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))


def _run(script, *args):
    subprocess.run([sys.executable, os.path.join(_HERE, script), *args], check=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True,
                    help="Path to reco_plotting.yaml.")
    ap.add_argument("--reco-merged",
                    help="Merged 3D reco H5 (consumed by reco_summary).")
    ap.add_argument("--combined",
                    help="combined_event_variables.h5 (consumed by PA-geometry plots).")
    ap.add_argument("--combined-source", choices=["burn", "sim"],
                    help="Rows to use for the PA-geometry distribution.")
    ap.add_argument(
        "--only",
        action="append",
        choices=["reco_summary", "pa_geometry", "sim_zenith_error"],
        help=("Run only this enabled plot group. May be repeated. By default "
              "all groups enabled in the config are considered."),
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--label", default="burn")
    args = ap.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    enabled = set(cfg.get("enabled", []))
    if args.only:
        requested = set(args.only)
        disabled = sorted(requested - enabled)
        if disabled:
            raise RuntimeError(
                "requested plot groups are not enabled in the config: "
                + ", ".join(disabled)
            )
        enabled = requested

    os.makedirs(args.output_dir, exist_ok=True)

    if "reco_summary" in enabled:
        if args.reco_merged:
            _run("plot_reco_summary.py",
                 "--input", args.reco_merged,
                 "--output-dir", args.output_dir,
                 "--label", args.label)
        else:
            print("[plot_all] skipping reco_summary: --reco-merged not given")

    if "pa_geometry" in enabled:
        if args.combined and args.combined_source:
            _run("plot_pa_geometry.py",
                 "--input", args.combined,
                 "--source", args.combined_source,
                 "--output-dir", args.output_dir,
                 "--label", args.label)
        else:
            print("[plot_all] skipping pa_geometry: "
                  "--combined or --combined-source not given")

    if "sim_zenith_error" in enabled:
        if args.combined:
            _run("plot_sim_zenith_error.py",
                 "--input", args.combined,
                 "--output-dir", args.output_dir,
                 "--label", args.label)
        else:
            print("[plot_all] skipping sim_zenith_error: --combined not given")


if __name__ == "__main__":
    main()
