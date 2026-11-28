#!/usr/bin/env python3
"""
Validate ``speedup: reuse_efields_for_identical_positions``.

A detector with channels at identical positions (noiseless copies, a copy with a different antenna
orientation, and a copy with a different readout length; trigger and non-trigger channels) is
simulated twice with the same seed: once with the reuse of the electric fields switched on and once
switched off. Both runs have to produce bit-identical HDF5 and ``.nur`` output and the same
candidate flags per channel, and the run with reuse has to call ``calculate_sim_efield`` fewer
times. Each run is a separate process because signal-generation
modules keep their random state in singletons.

Exits with a non-zero code on failure. All files are written to a temporary directory that is
removed at the end.
"""
import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile

import h5py
import numpy as np
import yaml

DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(DIR, "1e18_output_reference.hdf5")
DETECTOR_FILE = os.path.join(DIR, "surface_station_1GHz.json")

# channels 0 - 7 are the original detector, 8 - 12 are added at the positions of existing channels
TRIGGER_CHANNELS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
COPIES = {
    8: dict(original=0, changes={"noiseless": True}),  # trigger channel, reuses 0 (same loop)
    9: dict(original=4, changes={"noiseless": True}),  # non-trigger channel, reuses 4 (other loop)
    10: dict(original=1, changes={"ant_rotation_phi": 0.0}),  # other orientation, reuses 1
    11: dict(original=2, changes={"adc_n_samples": 512}),  # other trace length, must not reuse 2
    12: dict(original=0, changes={"noiseless": True}),  # second reuse of 0 (catches shared objects)
}


def write_detector(filename):
    with open(DETECTOR_FILE, "r") as fin:
        det = json.load(fin)

    channels = det["channels"]
    by_id = {ch["channel_id"]: ch for ch in channels.values()}
    for channel_id, spec in COPIES.items():
        new_channel = copy.deepcopy(by_id[spec["original"]])
        new_channel.update(spec["changes"])
        new_channel["channel_id"] = channel_id
        channels[str(len(channels) + 1)] = new_channel

    with open(filename, "w") as fout:
        json.dump(det, fout, indent=4)


def write_config(filename, reuse):
    config = {
        "noise": True,
        "seed": 1234,
        "sampling_rate": 5.,
        "speedup": {
            "min_efield_amplitude": 2,
            "redo_raytracing": True,
            "reuse_efields_for_identical_positions": reuse,
        },
        "propagation": {"ice_model": "ARAsim_southpole", "focusing": False},
        # Alvarez2009 draws a random shower parameter (k_L), which has to be reused consistently
        "signal": {"model": "Alvarez2009"},
        "trigger": {"noise_temperature": 300},
        "weights": {"weight_mode": "core_mantle_crust_simple"},
    }
    with open(filename, "w") as fout:
        yaml.dump(config, fout)


def run_simulation(detector_file, config_file, output_hdf5, output_nur):
    """ Run the simulation and return the number of ``calculate_sim_efield`` calls """
    import NuRadioReco.modules.channelBandPassFilter
    import NuRadioReco.modules.trigger.simpleThreshold
    from NuRadioReco.utilities import units
    from NuRadioReco.utilities.logging import LOGGING_STATUS
    from NuRadioMC.simulation import simulation

    channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
    triggerSimulator = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()

    n_calls = [0]
    calculate_sim_efield = simulation.calculate_sim_efield

    def counting_calculate_sim_efield(*args, **kwargs):
        n_calls[0] += 1
        return calculate_sim_efield(*args, **kwargs)

    simulation.calculate_sim_efield = counting_calculate_sim_efield

    # The candidate flag of each channel only matters if it is False for all channels of a station,
    # hence, it is recorded per call rather than validated via the output files.
    candidates = []
    calculate_sim_efield_cached = simulation._calculate_sim_efield_cached

    def recording_calculate_sim_efield_cached(*args, **kwargs):
        sim_station = calculate_sim_efield_cached(*args, **kwargs)
        candidates.append([kwargs["channel_id"], sim_station.is_candidate(),
                           len(sim_station.get_electric_fields())])
        return sim_station

    simulation._calculate_sim_efield_cached = recording_calculate_sim_efield_cached

    class mySimulation(simulation.simulation):

        def _detector_simulation_filter_amp(self, evt, station, det):
            channelBandPassFilter.run(evt, station, det, passband=[80 * units.MHz, 500 * units.MHz],
                                      filter_type='butter', order=10)

        def _detector_simulation_trigger(self, evt, station, det):
            triggerSimulator.run(evt, station, det, threshold=3 * self._Vrms,
                                 triggered_channels=TRIGGER_CHANNELS, number_concidences=1,
                                 trigger_name='simple_threshold')

    sim = mySimulation(
        inputfilename=INPUT_FILE, outputfilename=output_hdf5, detectorfile=detector_file,
        outputfilenameNuRadioReco=output_nur, config_file=config_file, default_detector_station=101,
        trigger_channels=TRIGGER_CHANNELS, file_overwrite=True, log_level=LOGGING_STATUS)
    n_triggered = sim.run()
    return n_calls[0], n_triggered, candidates


def compare_hdf5(file1, file2):
    """ Return a list of all datasets and attributes which are not identical """
    mismatches = []
    with h5py.File(file1, "r") as f1, h5py.File(file2, "r") as f2:
        for key in set(f1.attrs) | set(f2.attrs):
            if key == "config":  # differs by the switch which is tested here
                continue
            if key not in f1.attrs or key not in f2.attrs or not np.array_equal(f1.attrs[key], f2.attrs[key]):
                mismatches.append(f"attribute {key}")

        def visit(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            if name not in f2:
                mismatches.append(f"dataset {name} missing")
            elif not np.array_equal(np.array(obj), np.array(f2[name]), equal_nan=obj.dtype.kind == "f"):
                mismatches.append(f"dataset {name}")

        f1.visititems(visit)
        f2.visititems(lambda name, obj: mismatches.append(f"dataset {name} missing")
                      if isinstance(obj, h5py.Dataset) and name not in f1 else None)
    return mismatches


def _traces(base_station, kind):
    if kind == "efields":
        objects = base_station.get_electric_fields()
    else:
        objects = base_station.iter_channels()

    traces = {}
    for obj in objects:
        key = obj.get_id() if kind == "channels" else obj.get_unique_identifier()
        traces[key] = (obj.get_trace(), obj.get_trace_start_time())
    return traces


def compare_nur(file1, file2):
    """ Return a list of all traces and triggers which are not identical """
    import NuRadioReco.modules.io.eventReader

    events = []
    for filename in (file1, file2):
        reader = NuRadioReco.modules.io.eventReader.eventReader()
        reader.begin(filename)
        events.append(list(reader.run()))

    if len(events[0]) != len(events[1]):
        return [f"number of events {len(events[0])} != {len(events[1])}"]

    mismatches = []
    for evt1, evt2 in zip(*events):
        evt_id = (evt1.get_run_number(), evt1.get_id())
        if evt_id != (evt2.get_run_number(), evt2.get_id()):
            mismatches.append(f"event id {evt_id} != {(evt2.get_run_number(), evt2.get_id())}")
            continue

        for st1, st2 in zip(evt1.get_stations(), evt2.get_stations()):
            pairs = [(st1, st2, "channels"), (st1, st2, "efields"),
                     (st1.get_sim_station(), st2.get_sim_station(), "sim_channels"),
                     (st1.get_sim_station(), st2.get_sim_station(), "efields")]
            for obj1, obj2, kind in pairs:
                traces1, traces2 = _traces(obj1, kind), _traces(obj2, kind)
                if traces1.keys() != traces2.keys():
                    mismatches.append(f"event {evt_id}: different {kind}")
                    continue
                for key in traces1:
                    if not (np.array_equal(traces1[key][0], traces2[key][0])
                            and traces1[key][1] == traces2[key][1]):
                        mismatches.append(f"event {evt_id}: {kind} {key}")

            for trigger1 in st1.get_triggers().values():
                trigger2 = st2.get_trigger(trigger1.get_name())
                if trigger1.has_triggered() != trigger2.has_triggered() or not np.array_equal(
                        trigger1.get_trigger_time(), trigger2.get_trigger_time(), equal_nan=True):
                    mismatches.append(f"event {evt_id}: trigger {trigger1.get_name()}")

    return mismatches


def main():
    tmp_dir = tempfile.mkdtemp(prefix="nuradiomc_efield_reuse_")
    try:
        detector_file = os.path.join(tmp_dir, "detector.json")
        write_detector(detector_file)

        outputs = {}
        for reuse in (True, False):
            config_file = os.path.join(tmp_dir, f"config_{reuse}.yaml")
            write_config(config_file, reuse)
            output_hdf5 = os.path.join(tmp_dir, f"output_{reuse}.hdf5")
            output_nur = os.path.join(tmp_dir, f"output_{reuse}.nur")
            result = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--run", detector_file, config_file,
                 output_hdf5, output_nur], check=True, stdout=subprocess.PIPE, text=True)
            n_calls, n_triggered, candidates = json.loads(result.stdout.strip().splitlines()[-1])
            outputs[reuse] = dict(hdf5=output_hdf5, nur=output_nur, n_calls=n_calls, n_triggered=n_triggered,
                                  candidates=candidates)
            print(f"reuse = {reuse}: {n_calls} calls of calculate_sim_efield, {n_triggered} triggered events")

        errors = []
        if outputs[True]["n_triggered"] == 0:
            errors.append("no event triggered, the comparison is meaningless")
        if outputs[True]["n_calls"] >= outputs[False]["n_calls"]:
            errors.append("the electric fields were not reused (calculate_sim_efield was not called less often)")
        if outputs[True]["candidates"] != outputs[False]["candidates"]:
            errors.append("different candidate flags or numbers of electric fields per channel")

        errors += compare_hdf5(outputs[True]["hdf5"], outputs[False]["hdf5"])
        errors += compare_nur(outputs[True]["nur"], outputs[False]["nur"])
    finally:
        shutil.rmtree(tmp_dir)

    if errors:
        print("Output with and without reusing the electric fields differs:\n\t" + "\n\t".join(errors))
        sys.exit(1)
    print("Output with and without reusing the electric fields is identical.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", nargs=4, metavar=("DETECTOR", "CONFIG", "HDF5", "NUR"),
                        help="internal: run one simulation and print the number of calculate_sim_efield calls, "
                        "of triggered events, and the candidate flags")
    args = parser.parse_args()
    if args.run:
        print(json.dumps(run_simulation(*args.run)))
    else:
        main()
