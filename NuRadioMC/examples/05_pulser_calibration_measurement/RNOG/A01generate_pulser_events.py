"""
Generate RNO-G calibration pulser event input for NuRadioMC simulation.

Creates HDF5 event files with emitter at specified positions relative to
a station's phased array center. Uses measured RNO-G calibration pulser
waveform templates (rno_cal5C_*dB).

Supports single positions or grids over (azimuth, zenith, distance).
For grids, each position produces a separate events file in the output
directory.

Usage:
    # Single position
    python A01generate_pulser_events.py --station 23 --az 45 --zen 110 --r 200

    # Grid of positions (start stop step for each axis)
    python A01generate_pulser_events.py \\
        --station 23 \\
        --az 0 360 30 --zen 80 160 10 --r 50 300 50 \\
        --n-events 50 --output-dir data
"""

import argparse
import os

import numpy as np
from datetime import datetime
from itertools import product

from NuRadioReco.utilities import units
from NuRadioMC.EvtGen.generator import write_events_to_hdf5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_pa_center(det, station_id):
    """Get PA center as midpoint of channels 1 and 2."""
    station_abs = np.array(det.get_absolute_position(station_id))
    ch1_rel = np.array(det.get_relative_position(station_id, 1))
    ch2_rel = np.array(det.get_relative_position(station_id, 2))
    return station_abs + (ch1_rel + ch2_rel) / 2


def compute_emitter_position(det, station_id, az_deg, zen_deg, r_m):
    """Compute absolute emitter position from spherical coords relative to PA center."""
    pa_center = get_pa_center(det, station_id)
    az = np.radians(az_deg)
    zen = np.radians(zen_deg)
    emitter_pos = pa_center + r_m * np.array([
        np.sin(zen) * np.cos(az),
        np.sin(zen) * np.sin(az),
        np.cos(zen)
    ])
    return emitter_pos, pa_center


def generate_events(filename, n_events, emitter_pos, amplitude=1.0, attenuation=10):
    """Write emitter events to an HDF5 file.

    Parameters
    ----------
    filename : str
        Output HDF5 file path.
    n_events : int
        Number of events to generate.
    emitter_pos : array-like
        Absolute (x, y, z) position of emitter in meters.
    amplitude : float
        Amplitude scaling in volts (1.0 = lab measurement level).
    attenuation : int
        Cal pulser attenuation in dB (0, 5, 10, 15, or 20).
    """
    model = f"rno_cal5C_{attenuation}dB"

    attributes = {
        'simulation_mode': 'emitter',
        'n_events': n_events,
        'start_event_id': 0,
    }

    data_sets = {
        'event_group_ids': np.arange(n_events),
        'shower_ids': np.arange(n_events),

        'emitter_model': [model] * n_events,
        'emitter_amplitudes': np.ones(n_events) * amplitude * units.V,
        'emitter_antenna_type': ['RNOG_vpol_v3_5inch_center_n1.74'] * n_events,

        'xx': np.ones(n_events) * emitter_pos[0],
        'yy': np.ones(n_events) * emitter_pos[1],
        'zz': np.ones(n_events) * emitter_pos[2],

        'emitter_orientation_phi': np.zeros(n_events),
        'emitter_orientation_theta': np.zeros(n_events),
        'emitter_rotation_phi': np.zeros(n_events),
        'emitter_rotation_theta': np.ones(n_events) * 90 * units.deg,

        'emitter_polarization': np.zeros(n_events),
    }

    write_events_to_hdf5(filename, data_sets, attributes)
    print(f"Generated {n_events} events with model '{model}' -> {filename}")


def parse_axis(values):
    """Parse a CLI axis specification into a list of values.

    Parameters
    ----------
    values : list of float
        Either a single value (fixed) or three values (start, stop, step).

    Returns
    -------
    list of float
    """
    if len(values) == 1:
        return [values[0]]
    elif len(values) == 3:
        return np.arange(*values).tolist()
    else:
        raise SystemExit(f"Expected 1 or 3 values, got {len(values)}: {values}")


def build_grid(azimuths, zeniths, distances):
    """Build list of (r, zen, az) tuples from axis values.

    Returns
    -------
    list of (float, float, float)
        All (r, zen, az) combinations.
    """
    return list(product(distances, zeniths, azimuths))


def pos_label(r, zen, az):
    """File label for a single grid point."""
    return f"r{r}_zen{zen}_az{az}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate RNO-G calibration pulser events")
    parser.add_argument('--station', type=int, default=23)
    parser.add_argument('--n-events', type=int, default=50)
    parser.add_argument('--amplitude', type=float, default=1.0,
                        help='Amplitude in V (1.0 = lab measurement level)')
    parser.add_argument('--attenuation', type=int, default=10,
                        choices=[0, 5, 10, 15, 20])

    parser.add_argument('--az', type=float, nargs='+', required=True,
                        metavar='DEG',
                        help='Azimuth: single value or start stop step')
    parser.add_argument('--zen', type=float, nargs='+', required=True,
                        metavar='DEG',
                        help='Zenith: single value or start stop step')
    parser.add_argument('--r', type=float, nargs='+', required=True,
                        metavar='M',
                        help='Distance: single value or start stop step')

    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(SCRIPT_DIR, 'data'))
    args = parser.parse_args()

    azimuths = parse_axis(args.az)
    zeniths = parse_axis(args.zen)
    distances = parse_axis(args.r)
    grid = build_grid(azimuths, zeniths, distances)

    print(f"Grid: {len(distances)} distances x "
          f"{len(zeniths)} zeniths x {len(azimuths)} azimuths "
          f"= {len(grid)} positions")
    print(f"  r:   {distances}")
    print(f"  zen: {zeniths}")
    print(f"  az:  {azimuths}")

    from NuRadioReco.detector.RNO_G.rnog_detector import Detector
    det = Detector()
    det.update(datetime(2022, 8, 1))

    outdir = os.path.abspath(args.output_dir)
    os.makedirs(outdir, exist_ok=True)

    manifest = []
    for r, zen, az in grid:
        label = pos_label(r, zen, az)

        emitter_pos, pa_center = compute_emitter_position(
            det, args.station, az, zen, r)

        events_file = os.path.join(outdir, f'events_{label}.hdf5')
        generate_events(events_file, args.n_events, emitter_pos,
                        args.amplitude, args.attenuation)

    print(f"\nGenerated {len(grid)} positions in {outdir}/")
