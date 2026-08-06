#!/usr/bin/env python3
"""
compare_raytracing_modules.py

Flexible testing + plotting framework for comparing NuRadioMC raytracing propagation modules.
"""

from pathlib import Path
from datetime import datetime
import logging
import numpy as np

from NuRadioReco.utilities import units
from NuRadioMC.SignalProp.AnalyticRayTracingImpl.MultilayerAnalyticRayTracing.testutils import (
    make_grid,
    run_batch_comparison_full,
    flatten_full,
    make_tracer
)

import time

def timed_call(fn):
    t0 = time.perf_counter()
    result = fn()
    t1 = time.perf_counter()
    return result, (t1 - t0)

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    # Modules
    "module_a": "analytic",
    "module_b": "analytic",

    # Ice models
    "ice_model_a": "greenland_simple",
    "ice_model_b": "greenland_simple_layered",

    # Grid
    "x_range": (-3000.0, -0.11),
    "z_range": (-3000.0, -0.11),
    "n_x": 33,
    "n_z": 33,

    # Geometry
    "end_point": np.array([0.0, 0.0, -100.013]) * units.m,

    # Tracer settings
    "attenuation_model": "GL1",
    "n_freq": 25,
    "n_reflections": 0,

    # Output
    "output_root": "raytracing_output",
}

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("compare_raytracing_modules")

# ============================================================
# OUTPUT DIRECTORY
# ============================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(CONFIG["output_root"]) / timestamp
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Output directory: {OUTPUT_DIR}")

# ============================================================
# STATISTICS & TESTING
# ============================================================

def run_tests(data_dict, keys):
    logger.info("Running statistics")
    stats = {}

    for key in [
        "time_diff", "path_diff", "angle_diff",
        "solving_time_a", "solving_time_b", "solving_time_ratio",
        "attenuation_diff" , "focusing_diff"
    ]:
        if key not in data_dict:
            continue
        vals = data_dict[key]
        mask = ~np.isnan(vals)
        if np.sum(mask) == 0:
            continue
        valid_vals = vals[mask]

        stats[key] = {
            "mean": float(np.mean(valid_vals)),
            "median": float(np.median(valid_vals)),
            "std": float(np.std(valid_vals)),
            "max_abs": float(np.max(np.abs(valid_vals))),
        }

    print("\nStatistics:")
    print("{:<25} {:<15} {:<15} {:<15} {:<15}".format('Metric', 'Mean', 'Median', 'Std', 'Max Abs'))
    for key in stats:
        s = stats[key]
        print("{:<25} {:<15.4g} {:<15.4g} {:<15.4g} {:<15.4g}".format(
            key, s['mean'], s['median'], s['std'], s['max_abs']))

# ============================================================
# VALIDATION
# ============================================================

TIME_TOL = 1e-3
ANGLE_TOL = 1e-3
ATTENUATION_TOL = 1e-2
FOCUSING_TOL = 1e-1

def run_assertions(data_dict):
    """Assert that all relative differences are below tolerances."""
    checks = [
        ("time_diff", "time_a", TIME_TOL),
        ("angle_diff", "receive_angle_a", ANGLE_TOL),
        ("attenuation_diff", "attenuation_a", ATTENUATION_TOL),
        ("focusing_diff", "focusing_a", FOCUSING_TOL),
    ]

    for diff_col, ref_col, tol in checks:
        if diff_col not in data_dict or ref_col not in data_dict:
            continue
        diff_vals = np.abs(data_dict[diff_col])
        ref_vals = data_dict[ref_col]

        # Mask for valid entries where both modules succeeded
        valid_mask = (data_dict.get("has_a", np.ones_like(diff_vals, dtype=bool)) == 1) & \
                     (data_dict.get("has_b", np.ones_like(diff_vals, dtype=bool)) == 1)

        #vmax = np.nanpercentile(diff_vals[valid_mask], 95)
        #mask = (
        #    (~np.isnan(diff_vals)) &
        #    (diff_vals <= vmax) &
        #    (~np.isnan(ref_vals)) &
        #    (ref_vals != 0) &
        #    valid_mask
        #)
        mask = valid_mask
        if np.sum(mask) == 0:
            continue

        rel = np.divide(diff_vals[mask], ref_vals[mask])
        rel = np.abs(rel)
        max_rel = np.max(rel)

        assert (rel <= tol).all(), (
            f"{diff_col}/{ref_col} exceeded tolerance.\n"
            f"Maximum relative error: {max_rel:.3e}\n"
            f"Tolerance: {tol:.3e}"
        )
        print(f"✓ {diff_col}: max relative error = {max_rel:.3e}")

# ============================================================
# MAIN PIPELINE
# ============================================================

def run_full_pipeline():
    logger.info("Building tracers")
    tracer_a = make_tracer(
        CONFIG["module_a"],
        ice_model=CONFIG["ice_model_a"],
        attenuation_model=CONFIG["attenuation_model"],
        n_freq=CONFIG["n_freq"],
        n_reflections=CONFIG["n_reflections"],
    )

    tracer_b = None
    if CONFIG["module_b"] is not None:
        tracer_b = make_tracer(
            CONFIG["module_b"],
            ice_model=CONFIG["ice_model_b"],
            attenuation_model=CONFIG["attenuation_model"],
            n_freq=CONFIG["n_freq"],
            n_reflections=CONFIG["n_reflections"],
        )

    logger.info("Creating grid")
    grid, x_vals, z_vals = make_grid(
        CONFIG["x_range"], CONFIG["z_range"],
        CONFIG["n_x"], CONFIG["n_z"]
    )

    logger.info("Running comparison")
    results = run_batch_comparison_full(tracer_a, tracer_b, grid, CONFIG["end_point"])
    flattened = flatten_full(results)

    # Normalize flattened data into a dict of arrays
    if isinstance(flattened, list):
        all_keys = set()
        for d in flattened:
            all_keys.update(d.keys())
        keys = sorted(all_keys)
        data_dict = {k: [] for k in keys}
        for d in flattened:
            for k in keys:
                data_dict[k].append(d.get(k, np.nan))
    elif isinstance(flattened, dict):
        keys = list(flattened.keys())
        data_dict = {k: v for k, v in flattened.items()}
    else:
        raise TypeError(f"Unexpected type from flatten_full: {type(flattened)}")

    # Convert to numpy arrays
    data_dict = {k: np.array(v) for k, v in data_dict.items()}

    # Filter data to only include points where both modules succeeded
    if "has_a" in data_dict and "has_b" in data_dict:
        valid_mask = (data_dict["has_a"] == 1) & (data_dict["has_b"] == 1)
        for key in data_dict:
            data_dict[key] = data_dict[key][valid_mask]


    run_tests(data_dict, keys)
    run_assertions(data_dict)

def main():
    run_full_pipeline()

if __name__ == "__main__":
    main()