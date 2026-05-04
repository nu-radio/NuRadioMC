"""Feature plotting orchestrator. Dispatches enabled plots from a YAML config."""
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
                    help="Path to feature_plotting.yaml.")
    ap.add_argument("--features",
                    help="Merged feature H5 (consumed by outlier_correlation).")
    ap.add_argument("--reco-merged",
                    help="Merged 3D reco H5 (consumed by outlier_correlation).")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--label", default="burn")
    args = ap.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    enabled = set(cfg.get("enabled", []))

    os.makedirs(args.output_dir, exist_ok=True)

    if "outlier_correlation" in enabled:
        if args.features and args.reco_merged:
            _run("outlier_correlation_analysis.py",
                 "--features", args.features,
                 "--reco", args.reco_merged,
                 "--output-dir", args.output_dir,
                 "--label", args.label)
        else:
            print("[plot_all] skipping outlier_correlation: needs --features and --reco-merged")


if __name__ == "__main__":
    main()
