#!/bin/bash
set -e
cd NuRadioMC/test/emitter
python T01generate_events.py
python T02RunSimulation.py emitter_event_list.hdf5  dipole_100m.json config.yaml output.hdf5
