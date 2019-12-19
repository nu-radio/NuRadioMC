#!/bin/bash

python T02RunSimulation.py 1e18_output_reference.hdf5 surface_station_1GHz.json config.yaml 1e18_output.hdf5 1e18_output.nur

python T03validate.py 1e18_output.hdf5 1e18_output_reference.hdf5
