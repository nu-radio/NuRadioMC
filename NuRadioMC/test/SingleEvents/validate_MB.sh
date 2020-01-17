#!/bin/bash
set -e

NuRadioMC/test/SingleEvents/T02RunSimulation.py NuRadioMC/test/SingleEvents/MB_1e18_reference.hdf5 NuRadioMC/test/SingleEvents/surface_station_1GHz.json NuRadioMC/test/SingleEvents/config_MB.yaml NuRadioMC/test/SingleEvents/MB_1e18_output.hdf5

NuRadioMC/test/SingleEvents/T04validate_allmost_equal.py NuRadioMC/test/SingleEvents/MB_1e18_output.hdf5 NuRadioMC/test/SingleEvents/MB_1e18_reference.hdf5