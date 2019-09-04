#!/bin/bash

set -e
NuRadioMC/test/SingleEvents/T02RunSimulation.py NuRadioMC/test/SingleEvents/1e18_output_reference.hdf5 NuRadioMC/test/SingleEvents/surface_station_1GHz.json NuRadioMC/test/SingleEvents/config.yaml NuRadioMC/test/SingleEvents/1e18_output.hdf5

NuRadioMC/test/SingleEvents/T03validate.py NuRadioMC/test/SingleEvents/1e18_output.hdf5 NuRadioMC/test/SingleEvents/1e18_output_reference.hdf5
