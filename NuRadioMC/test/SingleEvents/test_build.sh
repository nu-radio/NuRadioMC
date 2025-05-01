#!/bin/bash
set -e

python3 NuRadioMC/test/SingleEvents/T02RunSimulation.py NuRadioMC/test/SingleEvents/1e18_output_reference.hdf5 NuRadioMC/test/SingleEvents/surface_station_1GHz.json NuRadioMC/test/SingleEvents/config.yaml NuRadioMC/test/SingleEvents/1e18_output.hdf5 NuRadioMC/test/SingleEvents/1e18_output.nur
python3 NuRadioMC/test/SingleEvents/T04validate_allmost_equal.py NuRadioMC/test/SingleEvents/1e18_output.hdf5 NuRadioMC/test/SingleEvents/1e18_output_reference.hdf5
python3 NuRadioMC/test/SingleEvents/T05validate_nur_file.py NuRadioMC/test/SingleEvents/1e18_output.nur NuRadioMC/test/SingleEvents/1e18_output_reference.nur
python3 NuRadioMC/test/SingleEvents/T02RunSimulation.py NuRadioMC/test/SingleEvents/1e18_output_reference.hdf5 NuRadioMC/test/SingleEvents/surface_station_1GHz.json NuRadioMC/test/SingleEvents/config_noise.yaml NuRadioMC/test/SingleEvents/1e18_output_noise.hdf5
python3 NuRadioMC/test/SingleEvents/T04validate_allmost_equal.py NuRadioMC/test/SingleEvents/1e18_output_noise.hdf5 NuRadioMC/test/SingleEvents/1e18_output_noise_reference.hdf5

# cleanup 
rm -v NuRadioMC/test/SingleEvents/{1e18_output_noise.hdf5,1e18_output.hdf5,1e18_output.nur}