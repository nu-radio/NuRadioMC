"""Generate per-channel R-Z travel time lookup tables (in-ice only).

Supports three output modes:
- multiray: per-ray-type tables (direct, refracted, reflected)
- combined: single table with min travel time across all ray types
- all: both multiray and combined

Uses NuRadioMC analytic raytracing to classify solutions by ray type.
"""

import numpy as np
import os
import argparse
from multiprocessing import Pool

from NuRadioMC.utilities.medium import greenland_simple
from NuRadioMC.SignalProp import analyticraytracing
import NuRadioReco.detector.detector
from astropy.time import Time

RAY_TYPE_NAMES = {1: 'direct', 2: 'refracted', 3: 'reflected'}
RAY_TYPES = list(RAY_TYPE_NAMES.keys())


def get_antenna_position(station, ch, det_date="2022-10-01",
                         detector_file=None):
    """Get antenna absolute Z for R-Z table creation.

    Parameters
    ----------
    station : int
        Station number.
    ch : int
        Channel number.
    det_date : str
        Detector date string.
    detector_file : str or None
        Path to exported detector JSON.xz file. If None, queries MongoDB.

    Returns
    -------
    np.ndarray
        Antenna position as [0, 0, z_absolute].
    """
    if detector_file:
        from NuRadioReco.detector import RNO_G
        det = RNO_G.rnog_detector.Detector(
            detector_file=detector_file,
            select_stations=station,
        )
    else:
        det = NuRadioReco.detector.detector.Detector(source="rnog_mongo")
    det.update(Time(det_date))
    ant_rel_pos = det.get_relative_position(station, ch)
    station_abs_pos = det.get_absolute_position(station)
    ant_absolute_z = ant_rel_pos[2] + station_abs_pos[2]
    return np.array([0.0, 0.0, ant_absolute_z])


def travel_time_gridpoint(args):
    """Compute travel times for all ray types at a single in-ice grid point.

    Returns
    -------
    tuple
        (i, j, dict mapping ray_type_int -> travel_time)
    """
    i, j, r_vals, z_vals, ice_model, ant_pos = args
    r = r_vals[i]
    z = z_vals[j]

    src = ant_pos.copy()
    src[0] += r
    src[2] = z

    if z >= 0:
        return i, j, {}

    rt = analyticraytracing.ray_tracing(ice_model)
    rt.set_start_and_end_point(list(src), list(ant_pos))
    rt.find_solutions()

    times_by_type = {}
    for i_sol in range(rt.get_number_of_solutions()):
        sol_type = rt.get_solution_type(i_sol)
        tt = rt.get_travel_time(i_sol)
        if np.isfinite(tt):
            if sol_type not in times_by_type or tt < times_by_type[sol_type]:
                times_by_type[sol_type] = float(tt)

    return i, j, times_by_type


def generate_tables(station, ch, mode="multiray", rz_res=1, r_max=1600,
                    z_min=-1600, z_max=0, num_threads=1, output_dir=None,
                    name_suffix="", det_date="2022-10-01",
                    detector_file=None):
    """Generate R-Z lookup tables for one channel (in-ice only).

    Parameters
    ----------
    station : int
        Station number.
    ch : int
        Channel number.
    mode : str
        "multiray" (3 per-ray-type files), "combined" (1 min-time file),
        or "all" (both).
    rz_res : float
        Grid resolution in meters.
    r_max : float
        Maximum R value in meters.
    z_min : float
        Minimum Z (most negative depth).
    z_max : float
        Maximum Z.
    num_threads : int
        Number of parallel processes.
    output_dir : str or None
        Output directory.
    name_suffix : str
        Suffix appended to filenames.
    det_date : str
        Detector date string.
    detector_file : str or None
        Path to exported detector JSON.xz file.
    """
    if output_dir is None:
        output_dir = os.path.join(".", f"station{station}")
    os.makedirs(output_dir, exist_ok=True)

    R_RANGE = np.arange(0, r_max + rz_res, rz_res)
    Z_RANGE = np.arange(z_min, z_max + rz_res, rz_res)

    ice_model = greenland_simple()
    ant_pos = get_antenna_position(station, ch, det_date=det_date,
                                   detector_file=detector_file)

    print(f"Generating tables for ch{ch} (station {station}), mode={mode}")
    print(f"  Antenna position: {ant_pos}")
    print(f"  R: [0, {r_max}] m, {len(R_RANGE)} points")
    print(f"  Z: [{z_min}, {z_max}] m, {len(Z_RANGE)} points")
    print(f"  Grid points: {len(R_RANGE) * len(Z_RANGE)}")
    print(f"  Using {num_threads} threads")

    work_items = [
        (i, j, R_RANGE, Z_RANGE, ice_model, ant_pos)
        for i in range(len(R_RANGE))
        for j in range(len(Z_RANGE))
    ]

    tables = {
        1: np.full((len(R_RANGE), len(Z_RANGE)), np.nan),
        2: np.full((len(R_RANGE), len(Z_RANGE)), np.nan),
        3: np.full((len(R_RANGE), len(Z_RANGE)), np.nan),
    }

    with Pool(num_threads) as pool:
        results = pool.map(travel_time_gridpoint, work_items)

    for i, j, times_by_type in results:
        for sol_type, tt in times_by_type.items():
            if sol_type in tables:
                tables[sol_type][i, j] = tt

    sfx = f"_{name_suffix}" if name_suffix else ""
    write_multiray = mode in ("multiray", "all")
    write_combined = mode in ("combined", "all")

    if write_multiray:
        for sol_type, name in RAY_TYPE_NAMES.items():
            data = tables[sol_type]
            outpath = os.path.join(
                output_dir,
                f"st{station}_ch{ch}_rz_table_{name}{sfx}.npz"
            )
            np.savez_compressed(
                outpath,
                r_range_vals=R_RANGE,
                z_range_vals=Z_RANGE,
                data=data,
            )
            n_valid = np.sum(~np.isnan(data))
            n_total = data.size
            print(f"  {name}: {n_valid}/{n_total} valid "
                  f"({100*n_valid/n_total:.1f}%)")
            if n_valid > 0:
                valid = data[~np.isnan(data)]
                print(f"    time range: [{valid.min():.2f}, "
                      f"{valid.max():.2f}] ns")
            print(f"    Saved: {outpath}")

    if write_combined:
        combined = np.full((len(R_RANGE), len(Z_RANGE)), np.nan)
        for sol_type in RAY_TYPES:
            mask = ~np.isnan(tables[sol_type])
            both_valid = mask & ~np.isnan(combined)
            combined[both_valid] = np.minimum(
                combined[both_valid], tables[sol_type][both_valid]
            )
            new_only = mask & np.isnan(combined)
            combined[new_only] = tables[sol_type][new_only]

        outpath = os.path.join(
            output_dir, f"st{station}_ch{ch}_rz_table{sfx}.npz"
        )
        np.savez_compressed(
            outpath,
            r_range_vals=R_RANGE,
            z_range_vals=Z_RANGE,
            data=combined,
        )
        n_valid = np.sum(~np.isnan(combined))
        n_total = combined.size
        print(f"  combined: {n_valid}/{n_total} valid "
              f"({100*n_valid/n_total:.1f}%)")
        if n_valid > 0:
            valid = combined[~np.isnan(combined)]
            print(f"    time range: [{valid.min():.2f}, "
                  f"{valid.max():.2f}] ns")
        print(f"    Saved: {outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate R-Z travel time lookup tables (in-ice only)"
    )
    parser.add_argument("--station", type=int, required=True)
    parser.add_argument("--channel", type=int, required=True)
    parser.add_argument("--mode", type=str, default="multiray",
                        choices=["multiray", "combined", "all"],
                        help="multiray (3 ray-type files), combined "
                             "(1 min-time file), or all (default: multiray)")
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--rz-res", type=int, default=1)
    parser.add_argument("--r-max", type=int, default=1600)
    parser.add_argument("--z-min", type=int, default=-1600)
    parser.add_argument("--z-max", type=int, default=0)
    parser.add_argument("--name-suffix", type=str, default="")
    parser.add_argument("--det-date", type=str, default="2022-10-01",
                        help="Detector date for antenna positions "
                             "(default: 2022-10-01)")
    parser.add_argument("--detector-file", type=str, default=None,
                        help="Path to exported detector JSON.xz file. "
                             "Use for batch jobs where MongoDB is "
                             "unreachable.")
    args = parser.parse_args()

    generate_tables(
        args.station, args.channel,
        mode=args.mode,
        rz_res=args.rz_res,
        r_max=args.r_max,
        z_min=args.z_min,
        z_max=args.z_max,
        num_threads=args.num_threads,
        output_dir=args.output_dir,
        name_suffix=args.name_suffix,
        det_date=args.det_date,
        detector_file=args.detector_file,
    )
