#!/bin/bash

python T02RunSimulation.py MB_1e18_reference.hdf5 surface_station_1GHz.json config_MB.yaml MB_1e18_output.hdf5

python T03validate.py MB_1e18_output.hdf5 MB_1e18_reference.hdf5