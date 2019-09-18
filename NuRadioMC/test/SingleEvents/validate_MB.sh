#!/bin/bash

NuRadioMC/test/SingleEvents/T02RunSimulation.py NuRadioMC/test/SingleEvents/MB_1e18_reference.hdf5 NuRadioMC/test/SingleEvents/surface_station_1GHz.json NuRadioMC/test/SingleEvents/config_MB.yaml NuRadioMC/test/SingleEvents/MB_1e18_output.hdf5

NuRadioMC/test/SingleEvents/T03validate.py NuRadioMC/test/SingleEvents/MB_1e18_output.hdf5 NuRadioMC/test/SingleEvents/MB_1e18_reference.hdf5