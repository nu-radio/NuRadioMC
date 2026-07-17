#!/usr/bin/env python3
"""Measure FT-noise trigger-channel Vrms over the full FT pool for a station.

Processes every forced-trigger event in every ROOT file of the station's FT pool, so the
per-channel Vrms is backed by the whole dataset and its run-to-run spread is visible. This
is the full-pool generalization of measure_trigger_vrms.py (which sampled 200 realizations).

Method, per FT event and per PA channel (0-3):
  1. take the forced-trigger readout trace (3.2 GHz, median-baseline-corrected),
  2. upsample to the 5 GHz internal rate (simulate.upsample_trace),
  3. apply the readout->trigger transfer (trigger_filter / readout_filter) to build the
     trigger-channel copy, the same transform the simulation injects FT noise through,
  4. measure the trigger-copy Vrms.

The recommended fixed value per channel is the median of the per-event trigger-copy Vrms
across the pool. Re-measurement depends on the FT pool, the clean mask, and the detector
response vintage, so it does not reproduce the historical hardcoded dicts to the digit; the
shipped trigger_vrms_station{NN}.yaml pin what production actually used.

Usage: python measure_trigger_vrms_full.py --station 23 --ft_noise_dir <dir> [--clean_mask <npz>]
"""
import argparse
import os
import sys
import glob
import logging
import numpy as np
from joblib import Parallel, delayed

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, '..', '..'))

from astropy.time import Time
from NuRadioReco.utilities import units
from NuRadioReco.detector.RNO_G import rnog_detector
from NuRadioReco.modules.RNO_G import hardwareResponseIncorporator
from simulate import upsample_trace

TRIGGER_CHS = [0, 1, 2, 3]
N_UP = int(round(2048 * (5.0 / 3.2)))  # FT event length at 5 GHz


def load_flagged(mask_path):
    """Return the set of (run, event) pairs flagged as contaminated in the clean mask."""
    flagged = set()
    if mask_path and os.path.exists(mask_path):
        m = np.load(mask_path)
        for r, e, c in zip(m['runNum'], m['eventNum'], m['is_clean']):
            if c == 0:
                flagged.add((int(r), int(e)))
    return flagged


def process_file(path, station_id, detector_file, event_time, flagged, mode):
    """Measure trigger-copy Vrms for every selected event in one FT/burn file.

    `mode` is "ft" (forced triggers only, contamination mask applied) or "burn" (all
    triggers; the per-event RMS median is the noise floor). Returns
    {run, n_evt, vrms:{ch:[...]}} or None on a read error.
    """
    from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData

    det = rnog_detector.Detector(detector_file=detector_file, log_level=logging.ERROR,
                                 select_stations=station_id)
    det.update(Time(event_time))
    hw_resp = hardwareResponseIncorporator.hardwareResponseIncorporator()
    hw_resp.begin(trigger_channels=np.array(TRIGGER_CHS))

    ff = np.fft.rfftfreq(N_UP, d=1.0 / (5.0 * units.GHz))
    transfer = {}
    for ch_id in TRIGGER_CHS:
        readout = hw_resp.get_filter(ff, station_id, ch_id, det, sim_to_data=True, is_trigger=False)
        trigger = hw_resp.get_filter(ff, station_id, ch_id, det, sim_to_data=True, is_trigger=True)
        rabs = np.abs(readout)
        safe = np.where(rabs > 1e-3 * np.max(rabs), readout, np.max(rabs))
        transfer[ch_id] = trigger / safe

    out = {ch: [] for ch in TRIGGER_CHS}
    selectors = [lambda einfo: einfo.triggerType == "FORCE"] if mode == "ft" else None
    try:
        reader = readRNOGData()
        reader.begin([path], convert_to_voltage=True, apply_baseline_correction="median",
                     selectors=selectors, select_runs=False, read_calibrated_data=False)
    except Exception:
        return None

    n_evt = 0
    try:
        for evt_in in reader.run():
            st_in = evt_in.get_station(station_id)
            if st_in is None:
                continue
            if (evt_in.get_run_number(), evt_in.get_id()) in flagged:
                continue
            vals, ok = {}, True
            for ch_id in TRIGGER_CHS:
                ch_in = st_in.get_channel(ch_id) if st_in.has_channel(ch_id) else None
                if ch_in is None:
                    ok = False
                    break
                up = upsample_trace(ch_in.get_trace(), N_UP)
                trig = np.fft.irfft(np.fft.rfft(up) * transfer[ch_id], n=N_UP)
                vals[ch_id] = float(np.std(trig))
            if not ok:
                continue
            for ch_id in TRIGGER_CHS:
                out[ch_id].append(vals[ch_id])
            n_evt += 1
    except Exception:
        pass
    finally:
        try:
            reader.end()
        except Exception:
            pass

    base = os.path.basename(path.rstrip("/"))
    run = int(base.split("_run")[1].split(".")[0] if "_run" in base else base.replace("run", ""))
    return {"run": run, "n_evt": n_evt, "vrms": out}


def main():
    """Process the full pool for one station and report per-channel trigger-path Vrms."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", type=int, required=True)
    parser.add_argument("--mode", choices=["ft", "burn"], default="ft")
    parser.add_argument("--ft_noise_dir", default=None,
                        help="station FT ROOT directory (mode ft); files station{NN}_run*.root")
    parser.add_argument("--burn_root", default=None,
                        help="station burn-sample run directory (mode burn)")
    parser.add_argument("--clean_mask", default=None, help="clean-mask npz (mode ft)")
    parser.add_argument("--detector_file", default=None,
                        help="detector description file; omit to query MongoDB at --event_time")
    parser.add_argument("--event_time", default="2022-10-01")
    parser.add_argument("--outdir", default=BASE_DIR)
    parser.add_argument("--n_jobs", type=int,
                        default=int(os.environ.get("SLURM_CPUS_PER_TASK", 20)))
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    sid = args.station
    if args.mode == "ft":
        if not args.ft_noise_dir:
            parser.error("--ft_noise_dir is required in mode ft")
        paths = sorted(glob.glob(os.path.join(args.ft_noise_dir, f"station{sid}_run*.root")))
        flagged = load_flagged(args.clean_mask)
    else:
        if not args.burn_root:
            parser.error("--burn_root is required in mode burn")
        paths = sorted(glob.glob(os.path.join(args.burn_root, "run*")))
        flagged = set()

    print(f"Station {sid} ({args.mode}): {len(paths)} inputs, {len(flagged)} flagged, "
          f"{args.n_jobs} workers")
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(process_file)(p, sid, args.detector_file, args.event_time, flagged, args.mode)
        for p in paths)
    results = [r for r in results if r is not None and r["n_evt"] > 0]
    total_evt = sum(r["n_evt"] for r in results)
    print(f"Read {len(results)} runs, {total_evt:,} events\n")

    per_ch = {ch: np.concatenate([np.array(r["vrms"][ch]) for r in results]) for ch in TRIGGER_CHS}
    per_run = {ch: np.array([np.mean(r["vrms"][ch]) for r in results]) for ch in TRIGGER_CHS}

    print(f"STATION {sid} ({args.mode.upper()}) TRIGGER-PATH Vrms "
          f"({total_evt:,} events, {len(results)} runs)")
    print(f"{'ch':>4} {'median':>10} {'mean':>10} {'std':>9} {'p5':>10} {'p95':>10} {'run-std':>9}")
    recommended = {}
    for ch in TRIGGER_CHS:
        v = per_ch[ch] / units.mV
        recommended[ch] = float(np.median(per_ch[ch]))
        print(f"{ch:>4} {np.median(v):>9.4f}m {np.mean(v):>9.4f}m {np.std(v):>8.4f}m "
              f"{np.percentile(v,5):>9.4f}m {np.percentile(v,95):>9.4f}m "
              f"{np.std(per_run[ch]/units.mV):>8.4f}m")

    print("\nTRIGGER_VRMS_FT = {")
    for ch in TRIGGER_CHS:
        print(f"    {ch}: {recommended[ch]:.6e},")
    print("}")

    np.savez(os.path.join(args.outdir, f"trigger_vrms_station{sid}_{args.mode}.npz"),
             **{f"vrms_ch{ch}": per_ch[ch] for ch in TRIGGER_CHS},
             **{f"runmean_ch{ch}": per_run[ch] for ch in TRIGGER_CHS},
             runs=np.array([r["run"] for r in results]))


if __name__ == "__main__":
    main()
