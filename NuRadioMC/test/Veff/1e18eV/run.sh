#!/bin/bash

python T01generate_event_list.py
python T02RunSimulation.py 1e18_full.hdf5  ../dipole_100m.json ../config.yaml output.hdf5
python T03check_output.py