#!/bin/bash

set -e

NuRadioMC/test/atmospheric_Aeff/1e18eV/T01generate_event_list.py

NuRadioMC/test/atmospheric_Aeff/1e18eV/T02RunSimulation.py 1e18_full.hdf5  ../dipole_100m.json ../config.yaml output.hdf5 output.nur

NuRadioMC/test/atmospheric_Aeff/1e18eV/T03check_output.py NuRadioMC/test/atmospheric_Aeff/1e18eV/output.hdf5

# tables/dummy is tracked
rm -r NuRadioMC/test/atmospheric_Aeff/1e18eV/tables/Greenland
rm NuRadioMC/test/atmospheric_Aeff/1e18eV/{output*,1e18_full.hdf5}
