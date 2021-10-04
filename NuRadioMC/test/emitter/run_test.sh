#!/bin/bash
set -e

python T01generate_event_list.py
python T02RunSimulation.py emitter_event_list.hdf5  dipole_100m.json config.yaml output.hdf5
