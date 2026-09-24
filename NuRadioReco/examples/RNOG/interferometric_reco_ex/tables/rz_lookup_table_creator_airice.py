"""Generate per-channel R-Z travel-time lookup tables that extend above the ice surface.

Below the surface every grid point is traced exactly as `rz_lookup_table_creator_inice.py`
does (NuRadioMC analytic ray tracer, solutions classified by ray type), so the in-ice rows
reproduce the tables written by `rz_lookup_table_creator_inice.py` bit for bit. The surface
row (z = 0) takes the limit from below (a source 1 cm under the surface). Above the surface
each point is traced with the air-ice ray tracer adapted from the RNO-G antenna-positioning
package (straight air leg, Snell refraction at a flat surface, analytic in-ice leg) and
stored as the direct ray type; refracted and reflected types have no above-surface solution.

Output NPZ files keep the layout of `rz_lookup_table_creator_inice.py` (`r_range_vals`,
`z_range_vals`, `data`) and add provenance keys (antenna z, ice model, detector source and date, generator commit) that the
reconstruction loader ignores.
"""

import argparse
import datetime
import logging
import os
import subprocess
import time
from multiprocessing import Pool

import numpy as np
from astropy.time import Time

import NuRadioReco.detector.detector
from NuRadioMC.SignalProp import analyticraytracing
from NuRadioMC.SignalProp.air_ice_raytracer import AirIceRaytracer
from NuRadioMC.utilities.medium import greenland_simple
from NuRadioReco.utilities import units

RAY_TYPE_NAMES = {1: 'direct', 2: 'refracted', 3: 'reflected'}
RAY_TYPES = list(RAY_TYPE_NAMES.keys())
SURFACE_EPS = 0.01
R_EPS = 0.01
ICE_MODEL_NAME = 'greenland_simple'

_ice_model = None
_ant_pos = None
_tracer = None


def get_antenna_position(station, ch, det_date, detector_file=None):
    """Return the antenna position as [0, 0, z_absolute] from MongoDB or a snapshot file.

    Args:
        station: Station number.
        ch: Channel number.
        det_date: Detector epoch as an ISO date string.
        detector_file: Exported detector snapshot; None queries the live database.
    """
    if detector_file:
        from NuRadioReco.detector import RNO_G
        det = RNO_G.rnog_detector.Detector(detector_file=detector_file, select_stations=station)
    else:
        det = NuRadioReco.detector.detector.Detector(source="rnog_mongo")
    det.update(Time(det_date))
    z = det.get_relative_position(station, ch)[2] + det.get_absolute_position(station)[2]
    return np.array([0.0, 0.0, z])


def _init_worker(ant_pos):
    """Give each worker process the ice model, antenna position and one reusable tracer."""
    global _ice_model, _ant_pos, _tracer
    logging.getLogger('NuRadioMC').setLevel(logging.ERROR)
    _ice_model = greenland_simple()
    _ant_pos = ant_pos
    _tracer = analyticraytracing.ray_tracing(_ice_model)


def _inice_times(src):
    """Fastest travel time per ray type from an in-ice source (same code path as `rz_lookup_table_creator_inice.py`)."""
    _tracer.set_start_and_end_point(list(src), list(_ant_pos))
    _tracer.find_solutions()
    times = {}
    for i_sol in range(_tracer.get_number_of_solutions()):
        sol_type = _tracer.get_solution_type(i_sol)
        tt = _tracer.get_travel_time(i_sol)
        if np.isfinite(tt) and (sol_type not in times or tt < times[sol_type]):
            times[sol_type] = float(tt)
    return times


def _air_time(src):
    """Total travel time of the refracted air-to-ice path, or an empty dict when none is found."""
    try:
        tracer = AirIceRaytracer(ice_model=_ice_model, start_point=src, end_point=_ant_pos,
                                 precision=0.001 * units.deg, max_iterations=100)
        tracer.run()
        tt = tracer.get_signal_travel_time(path='total')
    except Exception:
        return {}
    return {1: float(tt)} if np.isfinite(tt) else {}


def travel_time_row(args):
    """Compute the travel times of one z row for every R value.

    Returns:
        (j, list of dicts mapping ray type -> travel time, one per R value)
    """
    j, r_vals, z = args
    out = []
    for r in r_vals:
        src = _ant_pos.copy()
        src[0] += r
        if z < 0:
            src[2] = z
            out.append(_inice_times(src))
        elif z == 0:
            src[2] = -SURFACE_EPS
            out.append(_inice_times(src))
        else:
            src[0] = _ant_pos[0] + max(r, R_EPS)
            src[2] = z
            out.append(_air_time(src))
    return j, out


def generator_commit():
    """Short git commit of the checkout this script runs from, or 'unknown'."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        return subprocess.check_output(['git', '-C', here, 'rev-parse', '--short', 'HEAD'],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


def generate_tables(station, ch, mode='all', rz_res=1.0, r_max=1600.0, z_min=-1600.0,
                    z_max=300.0, num_threads=1, output_dir=None, name_suffix='',
                    det_date='2022-10-01', detector_file=None, progress_every=5):
    """Generate the R-Z tables of one channel and write them as NPZ files.

    Args:
        station: Station number.
        ch: Channel number.
        mode: 'multiray' (three ray-type files), 'combined' (one min-time file),
            'solution_ordered' (two arrival-ordered files) or 'all' (every file).
        rz_res: Grid step in metres for both axes.
        r_max: Largest horizontal distance in metres.
        z_min: Lowest z in metres (negative).
        z_max: Highest z in metres; positive values extend the table above the surface.
        num_threads: Worker processes.
        output_dir: Output directory (default `./station{N}`).
        name_suffix: Suffix inserted before `.npz`.
        det_date: Detector epoch.
        detector_file: Snapshot file for nodes without database access.
        progress_every: Print a progress line every this many z rows.

    Side effects:
        Writes NPZ files into `output_dir`.
    """
    output_dir = output_dir or os.path.join('.', f'station{station}')
    os.makedirs(output_dir, exist_ok=True)
    n_r = int(round(r_max / rz_res)) + 1
    n_z = int(round((z_max - z_min) / rz_res)) + 1
    r_range = np.round(np.arange(n_r) * rz_res, 6)
    z_range = np.round(z_min + np.arange(n_z) * rz_res, 6)
    ant_pos = get_antenna_position(station, ch, det_date, detector_file)
    n_above = int(np.sum(z_range > 0))
    print(f"Generating tables for st{station} ch{ch}, mode={mode}")
    print(f"  antenna position: {ant_pos}")
    print(f"  R: [0, {r_max}] m, {n_r} points; Z: [{z_min}, {z_max}] m, {n_z} points "
          f"({n_above} rows above the surface); {num_threads} threads", flush=True)

    tables = {t: np.full((n_r, n_z), np.nan) for t in RAY_TYPES}
    work = [(j, r_range, float(z_range[j])) for j in range(n_z)]
    t_start = time.time()
    with Pool(num_threads, initializer=_init_worker, initargs=(ant_pos,)) as pool:
        for n_done, (j, row) in enumerate(pool.imap_unordered(travel_time_row, work), 1):
            for i, times in enumerate(row):
                for sol_type, tt in times.items():
                    if sol_type in tables:
                        tables[sol_type][i, j] = tt
            if n_done % progress_every == 0 or n_done == n_z:
                elapsed = time.time() - t_start
                print(f"  rows {n_done}/{n_z} done, {elapsed:.0f} s elapsed, "
                      f"eta {elapsed / n_done * (n_z - n_done):.0f} s", flush=True)

    meta = dict(
        station_id=station, channel_id=ch, antenna_z_abs=float(ant_pos[2]),
        ice_model=ICE_MODEL_NAME, det_date=det_date,
        det_source='rnog_file' if detector_file else 'rnog_mongo',
        det_file=os.path.basename(detector_file) if detector_file else '',
        surface_method='air_ice_raytracer', surface_eps_m=SURFACE_EPS,
        generator=os.path.basename(__file__), generator_commit=generator_commit(),
        created=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
    )
    sfx = f"_{name_suffix}" if name_suffix else ""

    def save(name, data):
        path = os.path.join(output_dir, f"st{station}_ch{ch}_rz_table{name}{sfx}.npz")
        np.savez_compressed(path, r_range_vals=r_range, z_range_vals=z_range, data=data, **meta)
        n_valid = int(np.sum(np.isfinite(data)))
        print(f"  {name or 'combined'}: {n_valid}/{data.size} valid "
              f"({100 * n_valid / data.size:.1f}%), saved {path}", flush=True)

    if mode in ('multiray', 'all'):
        for sol_type, name in RAY_TYPE_NAMES.items():
            save(f"_{name}", tables[sol_type])
    stack = np.stack([tables[t] for t in RAY_TYPES])
    with np.errstate(all='ignore'):
        ordered = np.sort(np.where(np.isnan(stack), np.inf, stack), axis=0)
    ordered[np.isinf(ordered)] = np.nan
    if mode in ('combined', 'all'):
        save('', ordered[0])
    if mode in ('solution_ordered', 'all'):
        save('_solution_0', ordered[0])
        save('_solution_1', ordered[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--station', type=int, required=True)
    parser.add_argument('--channel', type=int, required=True)
    parser.add_argument('--mode', default='all',
                        choices=['multiray', 'combined', 'solution_ordered', 'all'])
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--rz-res', type=float, default=1.0)
    parser.add_argument('--r-max', type=float, default=1600.0)
    parser.add_argument('--z-min', type=float, default=-1600.0)
    parser.add_argument('--z-max', type=float, default=300.0)
    parser.add_argument('--name-suffix', default='')
    parser.add_argument('--det-date', default='2022-10-01')
    parser.add_argument('--detector-file', default=None,
                        help='exported detector snapshot; only for batch nodes without '
                             'database access')
    parser.add_argument('--progress-every', type=int, default=5)
    args = parser.parse_args()
    generate_tables(args.station, args.channel, mode=args.mode, rz_res=args.rz_res,
                    r_max=args.r_max, z_min=args.z_min, z_max=args.z_max,
                    num_threads=args.num_threads, output_dir=args.output_dir,
                    name_suffix=args.name_suffix, det_date=args.det_date,
                    detector_file=args.detector_file, progress_every=args.progress_every)
