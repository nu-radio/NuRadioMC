#!/usr/bin/env python3
"""
Regression test: a simulation with two stations that trigger in the same event group must
write every shower and event once (to the HDF5 and the .nur output).

Copies the detector description, moves station 102 to the position of station 101 (both
stations then see the same showers), runs T02RunSimulation.py and checks for duplicates.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter

import h5py

from NuRadioReco.modules.io.eventReader import eventReader

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('inputfilename', help="NuRadioMC input event list")
parser.add_argument('detectordescription', help="detector description with stations 101 and 102")
parser.add_argument('config', help="NuRadioMC yaml config file")
args = parser.parse_args()


def duplicates(values):
    return {k: n for k, n in Counter(values).items() if n > 1}


with tempfile.TemporaryDirectory() as tmpdir:
    with open(args.detectordescription) as f:
        det = json.load(f)
    station_101 = [s for s in det['stations'].values() if s['station_id'] == 101][0]
    for station in det['stations'].values():
        if station['station_id'] == 102:
            station['pos_easting'] = station_101['pos_easting']
            station['pos_northing'] = station_101['pos_northing']
    detector_file = os.path.join(tmpdir, 'two_stations.json')
    with open(detector_file, 'w') as f:
        json.dump(det, f)

    output_hdf5 = os.path.join(tmpdir, 'output.hdf5')
    output_nur = os.path.join(tmpdir, 'output.nur')
    subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'T02RunSimulation.py'),
         args.inputfilename, detector_file, args.config, output_hdf5, output_nur],
        check=True)

    errors = []
    with h5py.File(output_hdf5, 'r') as fin:
        dup = duplicates(fin['shower_ids'][()].tolist())
        if dup:
            errors.append(f"duplicated top-level shower_ids (id: count): {dup}")
        for station_id in [101, 102]:
            group = fin[f'station_{station_id}']
            if 'shower_id' not in group:
                errors.append(f"station {station_id} did not trigger; the test needs both stations to trigger")
                continue
            dup = duplicates(group['shower_id'][()].tolist())
            if dup:
                errors.append(f"station {station_id}: duplicated shower_id (id: count): {dup}")
            dup = duplicates(zip(group['event_group_ids'][()].tolist(), group['event_ids'][()].tolist()))
            if dup:
                errors.append(f"station {station_id}: duplicated (event_group_id, event_id): {dup}")

    reader = eventReader()
    reader.begin(output_nur)
    dup = duplicates((evt.get_run_number(), evt.get_id(), tuple(evt.get_station_ids())) for evt in reader.run())
    reader.end()
    if dup:
        errors.append(f"duplicated events in .nur file (run, id, stations): {dup}")

if errors:
    print("\n".join(errors))
    sys.exit(1)
print("multi-station simulation: no duplicated showers or events")
