"""Evaluate 3D reconstruction results against simulation truth.

Computes angular separation between reconstructed and true source directions
for each event, then prints summary statistics. Works with any NuRadioMC
simulation; reference values from the shipped validation datasets are printed
for comparison but will differ for other simulation sets, stations, or configs.

Supports two dataset types:
- neutrino: truth from paired HDF5 files (vertex positions), or directly
  from NUR files if no HDF5 files are found in --sim-dir
- pulser: truth from NUR filenames (output_r{R}_zen{ZEN}_az{AZ}.nur)

Usage:
    # Neutrino GZK dataset (paired HDF5 truth files)
    python evaluate_reco_results.py \
        --reco-file merged_reco_results.h5 \
        --dataset neutrino \
        --sim-dir /path/to/gzk_hdf5_files/

    # Neutrino dataset (NUR files only, no paired HDF5)
    python evaluate_reco_results.py \
        --reco-file merged_reco_results.h5 \
        --dataset neutrino \
        --sim-dir /path/to/nur_files/

    # Pulser sim dataset
    python evaluate_reco_results.py \
        --reco-file merged_reco_results.h5 \
        --dataset pulser
"""

import argparse
import glob
import os
import re

import h5py
import numpy as np

from NuRadioReco.detector.RNO_G import rnog_detector
from astropy.time import Time


def get_pa_center(station, det_date, detector_file=None):
    """Get phased array center absolute position.

    Parameters
    ----------
    station : int
        Station ID.
    det_date : str
        Detector date string.
    detector_file : str or None
        Path to exported detector JSON.xz file. If None, queries MongoDB.

    Returns
    -------
    np.ndarray
        PA center absolute position [x, y, z].
    """
    if detector_file:
        det = rnog_detector.Detector(
            detector_file=detector_file,
            select_stations=station,
        )
    else:
        from NuRadioReco.detector import detector as det_module
        det = det_module.Detector(source="rnog_mongo")
    det.update(Time(det_date))
    stn_pos = np.array(det.get_absolute_position(station))
    ch1_rel = np.array(det.get_relative_position(station, 1))
    ch2_rel = np.array(det.get_relative_position(station, 2))
    return stn_pos + 0.5 * (ch1_rel + ch2_rel)


def load_neutrino_truth(sim_dir, pa_abs, station=23):
    """Build truth lookup from GZK simulation HDF5 files.

    Parameters
    ----------
    sim_dir : str
        Directory containing paired HDF5 truth files.
    pa_abs : np.ndarray
        Phased array center absolute position [x, y, z].
    station : int
        Station ID.

    Returns
    -------
    dict
        Maps (nur_basename, event_group_id) to (rho, phi_deg, z_abs).
    """
    hdf5_files = sorted(glob.glob(os.path.join(sim_dir, "*.hdf5")))
    truth = {}
    station_key = f'station_{station}'

    for hf in hdf5_files:
        with h5py.File(hf, 'r') as h:
            if station_key not in h:
                continue
            egids_mc = h['event_group_ids'][:]
            xx, yy, zz = h['xx'][:], h['yy'][:], h['zz'][:]
            egids_trig = h[f'{station_key}/event_group_ids'][:]
            trig_set = set(egids_trig.tolist())
            basename = os.path.basename(hf).replace('.hdf5', '.nur')

            for i, egid in enumerate(egids_mc):
                if egid in trig_set:
                    dx = xx[i] - pa_abs[0]
                    dy = yy[i] - pa_abs[1]
                    rho = np.sqrt(dx**2 + dy**2)
                    phi = np.degrees(np.arctan2(dy, dx)) % 360.0
                    truth[(basename, int(egid))] = (rho, phi, zz[i])

    return truth


def load_neutrino_truth_nur(sim_dir, pa_abs, station=23):
    """Build truth lookup from NUR simulation files.

    Reads the primary interaction vertex from each event in the NUR files.
    Use this when paired HDF5 truth files are not available.

    Parameters
    ----------
    sim_dir : str
        Directory containing NUR simulation files.
    pa_abs : np.ndarray
        Phased array center absolute position [x, y, z].
    station : int
        Station ID.

    Returns
    -------
    dict
        Maps (nur_basename, run_number) to (rho, phi_deg, z_abs).
    """
    from NuRadioReco.modules.io.NuRadioRecoio import NuRadioRecoio
    import NuRadioReco.framework.parameters as parameters

    nur_files = sorted(glob.glob(os.path.join(sim_dir, "*.nur")))
    truth = {}

    for nf in nur_files:
        basename = os.path.basename(nf)
        fin = NuRadioRecoio([nf])
        for evt in fin.get_events():
            stn = evt.get_station(station)
            if stn is None:
                continue
            sim_shower = evt.get_first_sim_shower()
            if sim_shower is None:
                continue
            vertex = sim_shower.get_parameter(parameters.showerParameters.vertex)
            run_number = evt.get_run_number()

            dx = vertex[0] - pa_abs[0]
            dy = vertex[1] - pa_abs[1]
            rho = np.sqrt(dx**2 + dy**2)
            phi = np.degrees(np.arctan2(dy, dx)) % 360.0
            truth[(basename, int(run_number))] = (rho, phi, vertex[2])

    return truth


def load_pulser_truth(reco_file, pa_abs):
    """Build truth lookup from pulser scan filenames.

    Filenames follow the pattern output_r{R}_zen{ZEN}_az{AZ}.nur, where
    R is distance from PA center, ZEN is zenith angle, AZ is azimuth angle.

    Parameters
    ----------
    reco_file : str
        Reco results HDF5 file (to extract source_file entries).
    pa_abs : np.ndarray
        Phased array center absolute position [x, y, z].

    Returns
    -------
    dict
        Maps nur_basename to (rho, phi_deg, z_abs).
    """
    truth = {}

    with h5py.File(reco_file, 'r') as f:
        src_files = f['results']['source_file'][:]

    seen = set()
    for sf in src_files:
        if isinstance(sf, bytes):
            sf = sf.decode()
        if sf in seen:
            continue
        seen.add(sf)

        m = re.search(r'output_r([\d.]+)_zen([\d.]+)_az([\d.]+)\.', sf)
        if not m:
            continue
        r = float(m.group(1))
        zen = np.radians(float(m.group(2)))
        az = np.radians(float(m.group(3)))

        pos = pa_abs + r * np.array([
            np.sin(zen) * np.cos(az),
            np.sin(zen) * np.sin(az),
            np.cos(zen)
        ])
        dx = pos[0] - pa_abs[0]
        dy = pos[1] - pa_abs[1]
        rho = np.sqrt(dx**2 + dy**2)
        phi = np.degrees(np.arctan2(dy, dx)) % 360.0
        truth[sf] = (rho, phi, pos[2])

    return truth


def angular_separation(reco_rho, reco_phi, reco_z, truth_rho, truth_phi,
                       truth_z, pa_z):
    """Compute angular separation between reco and truth directions.

    Parameters
    ----------
    reco_rho, reco_phi, reco_z : float
        Reconstructed position (rho in m, phi in deg, z in m absolute).
    truth_rho, truth_phi, truth_z : float
        True position (same units).
    pa_z : float
        PA center absolute z coordinate.

    Returns
    -------
    float
        Angular separation in degrees.
    """
    dz_r = reco_z - pa_z
    dz_t = truth_z - pa_z
    vr = np.array([reco_rho * np.cos(np.radians(reco_phi)),
                    reco_rho * np.sin(np.radians(reco_phi)), dz_r])
    vt = np.array([truth_rho * np.cos(np.radians(truth_phi)),
                    truth_rho * np.sin(np.radians(truth_phi)), dz_t])
    nr, nt = np.linalg.norm(vr), np.linalg.norm(vt)
    if nr == 0 or nt == 0:
        return np.nan
    cos_ang = np.clip(np.dot(vr / nr, vt / nt), -1, 1)
    return np.degrees(np.arccos(cos_ang))


def print_metrics(ang_seps, label):
    """Print angular separation summary statistics.

    Parameters
    ----------
    ang_seps : np.ndarray
        Angular separations in degrees.
    label : str
        Label for the output header.
    """
    valid = ang_seps[~np.isnan(ang_seps)]
    print(f"\n{label}")
    print(f"  Events:           {len(valid)}")
    print(f"  Median:           {np.median(valid):.2f} deg")
    print(f"  68th percentile:  {np.percentile(valid, 68):.2f} deg")
    print(f"  90th percentile:  {np.percentile(valid, 90):.2f} deg")
    print(f"  Fraction < 1 deg: {100 * np.mean(valid < 1):.0f}%")
    print(f"  Fraction < 2 deg: {100 * np.mean(valid < 2):.0f}%")


EXPECTED = {
    'neutrino': (
        "Reference values (from shipped GZK validation dataset, station 23,\n"
        "  reco3d_neutrino_gzk.yaml, hw mode, 12,916 events):\n"
        "  Median: 1.04 deg, 68th: 2.04 deg, 90th: 14.09 deg\n"
        "  <1 deg: 49%, <2 deg: 68%\n"
        "  Note: your results will differ if using a different simulation set,\n"
        "  station, or config. These are provided as a ballpark reference."
    ),
    'pulser': (
        "Reference values (from shipped pulser validation dataset, station 23,\n"
        "  reco3d_pulser_sim.yaml, rxtx mode, 27 stratified events):\n"
        "  Median: 0.27 deg, 68th: 1.15 deg\n"
        "  <1 deg: 67%, <2 deg: 74%\n"
        "  Note: your results will differ if using a different simulation set,\n"
        "  station, or config. These are provided as a ballpark reference."
    ),
}


def main():
    """Load reco results and truth, compute and print angular separation metrics."""
    parser = argparse.ArgumentParser(
        description="Evaluate 3D reco results against simulation truth"
    )
    parser.add_argument("--reco-file", type=str, required=True,
                        help="Merged reco results HDF5 file")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["neutrino", "pulser"],
                        help="Dataset type: neutrino (HDF5 truth) or "
                             "pulser (filename truth)")
    parser.add_argument("--sim-dir", type=str, default=None,
                        help="Directory with paired .hdf5 truth files "
                             "(neutrino only)")
    parser.add_argument("--detector-file", type=str, default=None,
                        help="Exported detector JSON.xz file "
                             "(default: query MongoDB)")
    parser.add_argument("--station", type=int, default=23)
    parser.add_argument("--det-date", type=str, default="2022-10-01")
    args = parser.parse_args()

    if args.dataset == "neutrino" and args.sim_dir is None:
        parser.error("--sim-dir is required for neutrino dataset")

    pa_abs = get_pa_center(args.station, args.det_date,
                           detector_file=args.detector_file)

    if args.dataset == "neutrino":
        hdf5_files = glob.glob(os.path.join(args.sim_dir, "*.hdf5"))
        if hdf5_files:
            truth = load_neutrino_truth(args.sim_dir, pa_abs, args.station)
            print(f"Loaded {len(truth)} truth entries from HDF5 files in {args.sim_dir}")
        else:
            nur_files = glob.glob(os.path.join(args.sim_dir, "*.nur"))
            if not nur_files:
                print(f"No .hdf5 or .nur files found in {args.sim_dir}")
                return
            truth = load_neutrino_truth_nur(args.sim_dir, pa_abs, args.station)
            print(f"Loaded {len(truth)} truth entries from NUR files in {args.sim_dir}")
    else:
        truth = load_pulser_truth(args.reco_file, pa_abs)
        print(f"Loaded {len(truth)} unique pulser positions from filenames")

    with h5py.File(args.reco_file, 'r') as f:
        reco_rho = f['results']['rho'][:]
        reco_phi = f['results']['phi'][:]
        reco_z = f['results']['z'][:]
        run_num = f['results']['run_number'][:]
        src_file = f['results']['source_file'][:]

    ang_seps = []
    matched = 0
    for i in range(len(reco_rho)):
        sf = src_file[i]
        if isinstance(sf, bytes):
            sf = sf.decode()

        if args.dataset == "neutrino":
            key = (sf, int(run_num[i]))
        else:
            key = sf

        if key not in truth:
            continue
        matched += 1
        t_rho, t_phi, t_z = truth[key]
        ang_seps.append(angular_separation(
            reco_rho[i], reco_phi[i], reco_z[i],
            t_rho, t_phi, t_z, pa_abs[2]
        ))

    ang_seps = np.array(ang_seps)
    print(f"Matched {matched} events")

    print_metrics(ang_seps, "Angular separation metrics:")

    if args.dataset in EXPECTED:
        print()
        print(EXPECTED[args.dataset])


if __name__ == "__main__":
    main()
