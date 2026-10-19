#!/bin/bash
set -e

python3 NuRadioMC/test/SingleEvents/T02RunSimulation.py NuRadioMC/test/SingleEvents/1e18_output_reference.hdf5 NuRadioMC/test/SingleEvents/surface_station_1GHz.json NuRadioMC/test/SingleEvents/config.yaml NuRadioMC/test/SingleEvents/1e18_output.hdf5 NuRadioMC/test/SingleEvents/1e18_output.nur
python3 NuRadioMC/test/SingleEvents/T04validate_allmost_equal.py NuRadioMC/test/SingleEvents/1e18_output.hdf5 NuRadioMC/test/SingleEvents/1e18_output_reference.hdf5
# numerical focusing factor: it depends on the precision of the ray tracing solution and varies by up to ~1e-3
# between platforms, hence the channel/station parameters are compared with a looser tolerance
python3 NuRadioMC/test/SingleEvents/T05validate_nur_file.py NuRadioMC/test/SingleEvents/1e18_output.nur NuRadioMC/test/SingleEvents/1e18_output_reference.nur 7 1e-2
# analytic focusing factor: deterministic, hence the parameters are compared with the default (tight) tolerance
python3 NuRadioMC/test/SingleEvents/T02RunSimulation.py NuRadioMC/test/SingleEvents/1e18_output_reference.hdf5 NuRadioMC/test/SingleEvents/surface_station_1GHz.json NuRadioMC/test/SingleEvents/config_analytic_focusing.yaml NuRadioMC/test/SingleEvents/1e18_output_analytic_focusing.hdf5 NuRadioMC/test/SingleEvents/1e18_output_analytic_focusing.nur
python3 NuRadioMC/test/SingleEvents/T05validate_nur_file.py NuRadioMC/test/SingleEvents/1e18_output_analytic_focusing.nur NuRadioMC/test/SingleEvents/1e18_output_analytic_focusing_reference.nur
python3 NuRadioMC/test/SingleEvents/T02RunSimulation.py NuRadioMC/test/SingleEvents/1e18_output_reference.hdf5 NuRadioMC/test/SingleEvents/surface_station_1GHz.json NuRadioMC/test/SingleEvents/config_noise.yaml NuRadioMC/test/SingleEvents/1e18_output_noise.hdf5
python3 NuRadioMC/test/SingleEvents/T04validate_allmost_equal.py NuRadioMC/test/SingleEvents/1e18_output_noise.hdf5 NuRadioMC/test/SingleEvents/1e18_output_noise_reference.hdf5

# cleanup 
rm -v NuRadioMC/test/SingleEvents/{1e18_output_noise.hdf5,1e18_output.hdf5,1e18_output.nur,1e18_output_analytic_focusing.hdf5,1e18_output_analytic_focusing.nur}