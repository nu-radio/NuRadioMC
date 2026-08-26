#!/usr/bin/env python3
"""
compare_raytracing_modules.py

Flexible testing + plotting framework for comparing NuRadioMC
ray tracing propagation modules.

Features
--------
- Easy configuration section
- Compare one or two modules
- Optional plotting
- Optional testing/statistics
- Automatic timestamped output folders
- Multiple plotted quantities
- Safe handling of missing solutions
- Saves all plots automatically

Example
-------
python compare_raytracing_modules.py
"""

from datetime import datetime
import logging



import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.titleweight": "bold", 
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 18,
    "figure.titlesize": 20,
    "savefig.dpi": 600,
})

from NuRadioReco.utilities import units
from NuRadioMC.utilities import medium
from NuRadioMC.SignalProp import propagation
import time

# ------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("raytracing_utils")


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def log_timing(level=logging.DEBUG):

    def decorator(func):

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):

            start = time.perf_counter()

            try:
                result = func(self, *args, **kwargs)

            except Exception:

                elapsed_ms = (time.perf_counter() - start) * 1000

                self._logger.exception(
                    "%s failed after %.3f ms",
                    func.__name__,
                    elapsed_ms
                )

                raise

            elapsed_ms = (time.perf_counter() - start) * 1000

            self._logger.info(
                level,
                "%s completed in %.3f ms",
                func.__name__,
                elapsed_ms
            )

            return result

        return wrapper

    return decorator

def timed_call(fn):
    t0 = time.perf_counter()
    result = fn()
    t1 = time.perf_counter()
    return result, (t1 - t0)

def make_tracer(
    module_name,
    ice_model="greenland_simple",
    attenuation_model="GL1",
    n_freq=25,
    n_reflections=0,
):
    ice = medium.get_ice_model(ice_model)

    prop = propagation.get_propagation_module(module_name)

    return prop(
        ice,
        attenuation_model,
        n_frequencies_integration=n_freq,
        n_reflections=n_reflections,
    )

def make_grid(x_range, z_range, n_x, n_z, y=0.0):

    x_vals = np.linspace(*x_range, n_x)
    z_vals = np.linspace(*z_range, n_z)

    grid = []

    for iz, z in enumerate(z_vals):
        for ix, x in enumerate(x_vals):

            grid.append({
                "ix": ix,
                "iz": iz,
                "x": x,
                "z": z,
                "start": np.array([x, y, z]) * units.m,
            })

    return grid, x_vals, z_vals

def angle_between(v1, v2):

    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    return np.arccos(
        np.clip(np.dot(v1, v2), -1, 1)
    )

def safe_get(func, default=np.nan, warning=""):

    try:

        val = func()

        # guard against None
        if val is None:
            raise ValueError("returned None")

        # guard against invalid scalars
        if np.isscalar(val) and not np.isfinite(val):
            raise ValueError(f"non-finite value: {val}")

        return val

    except Exception as e:

        logger.warning(f"{warning}: {e}")

        return default

def compare_point_full(
    tracer_a,
    tracer_b,
    start,
    end,
):

    out = {
        "solutions": {}
    }

    try:

        # ======================================================
        # RUN TRACER A
        # ======================================================
        tracer_a.set_start_and_end_point(start, end)

        _, time_find_a = timed_call(tracer_a.find_solutions)

        results_a = tracer_a.get_results()
        n_a = tracer_a.get_number_of_solutions()

        # ======================================================
        # BUILD GROUPING FOR A
        # ======================================================
        group_map_a = {}

        sorted_indices_a = sorted(
            range(n_a),
            key=lambda i: results_a[i]['C0']
        )

        for group_id, i in enumerate(sorted_indices_a):
            group_map_a[i] = group_id

        # ======================================================
        # RUN TRACER B
        # ======================================================
        group_map_b = {}
        n_b = 0
        time_find_b = None
        results_b = None

        if tracer_b is not None:

            tracer_b.set_start_and_end_point(start, end)

            _, time_find_b = timed_call(tracer_b.find_solutions)

            results_b = tracer_b.get_results()
            n_b = tracer_b.get_number_of_solutions()

            # ==================================================
            # BUILD GROUPING FOR B
            # ==================================================
            sorted_indices_b = sorted(
                range(n_b),
                key=lambda i: results_b[i]['C0']
            )

            for group_id, i in enumerate(sorted_indices_b):
                group_map_b[i] = group_id

        # ======================================================
        # OUTPUT
        # ======================================================
        out["n_a"] = n_a
        out["n_b"] = n_b

        all_groups = (
            set(group_map_a.values())
            | set(group_map_b.values())
        )

        for group in sorted(all_groups):

            entry = {
                "group": group,
                "a": None,
                "b": None,
                "diff": None,
            }

            # ==================================================
            # MODULE A
            # ==================================================
            if group in group_map_a.values():

                i_a = next(
                    i for i in range(n_a)
                    if group_map_a.get(i) == group
                )

                tt_a = tracer_a.get_travel_time(i_a)
                pl_a = tracer_a.get_path_length(i_a)
                rv_a = tracer_a.get_receive_vector(i_a)
                freqs = np.linspace(100*units.MHz, 700*units.MHz, 1)
                att_a = tracer_a.get_attenuation(i_a,freqs)[0]
                foc_a = tracer_a.get_focusing(i_a)
                #ra_a = tracer_a.get_receive_angle(i_a)

                entry["a"] = {
                    "travel_time_ns": safe_get(
                        lambda: tt_a / units.ns,
                        warning=f"travel time failed (A, group {group})",
                    ),
                    "path_length_m": safe_get(
                        lambda: pl_a / units.m,
                        warning=f"path length failed (A, group {group})",
                    ),
                    "receive_vector": safe_get(
                        lambda: rv_a,
                        default=None,
                        warning=f"receive vector failed (A, group {group})",
                    ),
                    "solving_time_ms": safe_get(
                        lambda: time_find_a  * 1e3,
                        warning=f"solving time failed (A, group {group})",
                    ),
                    "receive_angle": safe_get(
                        lambda : np.degrees(np.arccos(rv_a[2] / np.linalg.norm(rv_a))),
                        default=None,
                        warning=f"recive angle failed (A, group {group})",
                    ),
                    "attenuation": safe_get(
                        lambda : att_a,
                        default=None,
                        warning=f"attenuation failed (A, group {group})",
                    ),
                    "focusing": safe_get(
                        lambda : foc_a,
                        default=None,
                        warning=f"focusing failed (A, group {group})",
                    )
                }

            # ==================================================
            # MODULE B
            # ==================================================
            if tracer_b is not None and group in group_map_b.values():

                i_b = next(
                    i for i in range(n_b)
                    if group_map_b.get(i) == group
                )

                tt_b = tracer_b.get_travel_time(i_b)
                pl_b = tracer_b.get_path_length(i_b)
                rv_b = tracer_b.get_receive_vector(i_b)
                freqs = np.linspace(100*units.MHz, 700*units.MHz, 1)
                att_b = tracer_b.get_attenuation(i_b,freqs)[0]
                foc_b = tracer_b.get_focusing(i_b)
                #ra_b = tracer_b.get_receive_angle(i_b)

                entry["b"] = {
                    "travel_time_ns": safe_get(
                        lambda: tt_b / units.ns,
                        warning=f"travel time failed (B, group {group})",
                    ),
                    "path_length_m": safe_get(
                        lambda: pl_b / units.m,
                        warning=f"path length failed (B, group {group})",
                    ),
                    "receive_vector": safe_get(
                        lambda: rv_b,
                        default=None,
                        warning=f"receive vector failed (B, group {group})",
                    ),
                    "solving_time_ms": safe_get(
                        lambda: time_find_b * 1e3,
                        warning=f"solving time failed (B, group {group})",
                    ),
                    "receive_angle": safe_get(
                        lambda : np.degrees(np.arccos(rv_b[2] / np.linalg.norm(rv_b))),
                        default=None,
                        warning=f"recive angle failed (B, group {group})",
                    ),
                    "attenuation": safe_get(
                        lambda : att_b,
                        default=None,
                        warning=f"attenuation failed (B, group {group})",
                    ),
                    "focusing": safe_get(
                        lambda : foc_b,
                        default=None,
                        warning=f"focusing failed (B, group {group})",
                    )
                    
                }

            # ==================================================
            # DIFFERENCES
            # ==================================================
            if entry["a"] is not None and entry["b"] is not None:

                entry["diff"] = {
                    "time_diff": (
                        entry["b"]["travel_time_ns"]
                        - entry["a"]["travel_time_ns"]
                    ),
                    "time_diff_rel": (
                        (entry["b"]["travel_time_ns"]
                        - entry["a"]["travel_time_ns"])*100.0/entry["a"]["travel_time_ns"]
                    ),
                    "path_diff": (
                        entry["b"]["path_length_m"]
                        - entry["a"]["path_length_m"]
                    ),
                    "angle_diff": (
                        angle_between(
                            entry["a"]["receive_vector"],
                            entry["b"]["receive_vector"],
                        ) / units.deg
                    ),
                    "solving_time_ratio": safe_get(
                        lambda: (
                            entry["a"]["solving_time_ms"]
                            / entry["b"]["solving_time_ms"]
                        ),
                        warning=f"solving time ratio failed (group {group})",
                    ),
                    "attenuation_diff": safe_get(
                        lambda : (entry["b"]["attenuation"]-entry["a"]["attenuation"]),
                        default=None,
                        warning=f"attenuation diff failed (group {group})",
                    ),
                    "focusing_diff": safe_get(
                        lambda : (entry["b"]["focusing"]-entry["a"]["focusing"]),
                        default=None,
                        warning=f"focusing diff failed (group {group})",
                    )

                }

            out["solutions"][group] = entry

    except Exception as e:

        out["exception"] = str(e)

    return out


def run_batch_comparison_full(
    tracer_a,
    tracer_b,
    grid,
    end_point,
):

    results = []

    total = len(grid)

    for i, g in enumerate(grid):

        if i % 2000 == 0:
            logger.info(f"{i}/{total}")

        res = {
            "ix": g["ix"],
            "iz": g["iz"],
            "x": g["x"],
            "z": g["z"],
        }

        comp = compare_point_full(
            tracer_a,
            tracer_b,
            g["start"],
            end_point,
        )

        res.update(comp)

        results.append(res)

    return results


# ============================================================
# FLATTEN
# ============================================================
def flatten_full(results):

    rows = []

    for r in results:

        base = {
            "ix": r["ix"],
            "iz": r["iz"],
            "x": r["x"],
            "z": r["z"],
            "n_a": r.get("n_a"),
            "n_b": r.get("n_b"),
        }

        #for stype, sol in r.get("solutions", {}).items():
        for group, sol in r.get("solutions", {}).items():

            row = base.copy()
            #row["solution_group"] = stype
            row["solution_group"] = group

            # -------------------------
            # module A
            # -------------------------
            if sol["a"] is not None:
                row["time_a"] = sol["a"]["travel_time_ns"]
                row["path_a"] = sol["a"]["path_length_m"]
                row["solving_time_a"] = sol["a"]["solving_time_ms"]
                row["receive_angle_a"] = sol["a"]["receive_angle"]
                row["attenuation_a"] = sol["a"]["attenuation"]
                row["focusing_a"] = sol["a"]["focusing"]

            # -------------------------
            # module B
            # -------------------------
            if sol["b"] is not None:
                row["time_b"] = sol["b"]["travel_time_ns"]
                row["path_b"] = sol["b"]["path_length_m"]
                row["solving_time_b"] = sol["b"]["solving_time_ms"]
                row["receive_angle_b"] = sol["b"]["receive_angle"]
                row["attenuation_b"] = sol["b"]["attenuation"]
                row["focusing_b"] = sol["b"]["focusing"]

            # -------------------------
            # differences
            # -------------------------
            if sol["diff"] is not None:
                row["time_diff"] = sol["diff"]["time_diff"]
                row["time_diff_rel"] = sol["diff"]["time_diff_rel"]
                row["path_diff"] = sol["diff"]["path_diff"]
                row["angle_diff"] = sol["diff"]["angle_diff"]
                row["solving_time_ratio"] = sol["diff"]["solving_time_ratio"]
                row["attenuation_diff"] = sol["diff"]["attenuation_diff"]
                row["focusing_diff"] = sol["diff"]["focusing_diff"]

            # NEW: solution existence flags
            row["has_a"] = sol["a"] is not None
            row["has_b"] = sol["b"] is not None

            rows.append(row)

    return rows