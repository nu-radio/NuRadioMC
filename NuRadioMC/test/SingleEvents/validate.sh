#!/bin/bash

python3 T02RunSimulation.py 1e18_output_reference.hdf5 surface_station_1GHz.json config.yaml 1e18_output.hdf5 1e18_output.nur

python3 T03validate.py 1e18_output.hdf5 1e18_output_reference.hdf5

python3 T05validate_nur_file.py 1e18_output.nur 1e18_output_reference.nur

# cleanup 
rm -v 1e18_output.hdf5
rm -v 1e18_output_reference.hdf5
