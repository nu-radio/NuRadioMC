#!/bin/bash
set -e
NuRadioMC/test/SingleEvents/T02RunSimulation.py NuRadioMC/test/SingleEvents/1e18_output_ARZ_reference.hdf5 NuRadioMC/test/SingleEvents/surface_station_1GHz.json NuRadioMC/test/SingleEvents/config_ARZ.yaml NuRadioMC/test/SingleEvents/1e18_output_ARZ.hdf5
NuRadioMC/test/SingleEvents/T04validate_allmost_equal.py NuRadioMC/test/SingleEvents/1e18_output_ARZ.hdf5 NuRadioMC/test/SingleEvents/1e18_output_ARZ_reference.hdf5