"""Reco plotting orchestrator. Dispatches enabled plots from a YAML config."""
import argparse
import os
import re
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
                    help="combined_event_variables.h5 (consumed by sim_zenith_error).")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--label", default="burn")
    ap.add_argument("--station", type=int,
                    help="Station id (needed by sim_zenith_error); inferred from a "
                         "station{N} component of --output-dir when omitted.")
    ap.add_argument("--detector-source", default="rnog_mongo")
    ap.add_argument("--detector-file", default=None)
    ap.add_argument("--detector-date", default="2022-10-01")
    args = ap.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    enabled = set(cfg.get("enabled", []))
    if args.station is None:
        m = re.search(r"station(\d+)", os.path.abspath(args.output_dir))
        if m:
            args.station = int(m.group(1))
            print(f"[plot_all] station {args.station} inferred from the output path")

    os.makedirs(args.output_dir, exist_ok=True)

    if "reco_summary" in enabled:
        if args.reco_merged:
            _run("plot_reco_summary.py",
                 "--input", args.reco_merged,
                 "--output-dir", args.output_dir,
                 "--label", args.label)
        else:
            print("[plot_all] skipping reco_summary: --reco-merged not given")

    if "sim_zenith_error" in enabled:
        if args.combined and args.station is not None:
            extra = ["--detector-source", args.detector_source, "--detector-date", args.detector_date]
            if args.detector_file:
                extra += ["--detector-file", args.detector_file]
            _run("plot_sim_zenith_error.py",
                 "--input", args.combined,
                 "--output-dir", args.output_dir,
                 "--label", args.label,
                 "--station", str(args.station), *extra)
        else:
            print("[plot_all] skipping sim_zenith_error: --combined or --station not given")


if __name__ == "__main__":
    main()
