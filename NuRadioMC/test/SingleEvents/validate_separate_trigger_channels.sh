#!/bin/bash
set -e

python3 NuRadioMC/examples/01_Veff_simulation_B/T02RunSimulation.py NuRadioMC/examples/01_Veff_simulation_B/1e19_n1e3.hdf5 NuRadioMC/examples/01_Veff_simulation_B/surface_station_1GHz.json NuRadioMC/examples/01_Veff_simulation_B/config_no_noise.yaml NuRadioMC/examples/01_Veff_simulation_B/1e19_output.hdf5 NuRadioMC/examples/01_Veff_simulation_B/1e19_output.nur

python3 NuRadioMC/examples/01_Veff_simulation_B/T02RunSimulation_trigger_channels.py NuRadioMC/examples/01_Veff_simulation_B/1e19_n1e3.hdf5 NuRadioMC/examples/01_Veff_simulation_B/surface_station_1GHz.json NuRadioMC/examples/01_Veff_simulation_B/config_no_noise.yaml NuRadioMC/examples/01_Veff_simulation_B/1e19_output_trigger_channels.hdf5 NuRadioMC/examples/01_Veff_simulation_B/1e19_output_trigger_channels.nur
# larger tolerance: for the non-trigger channels of the second simulation, the detector response is applied
# per efield and the traces are summed afterwards (instead of applying it to the summed efields), which changes
# the (envelope) maximum amplitudes by up to ~7.5e-4
python3 NuRadioMC/test/SingleEvents/T04validate_allmost_equal.py NuRadioMC/examples/01_Veff_simulation_B/1e19_output.hdf5 NuRadioMC/examples/01_Veff_simulation_B/1e19_output_trigger_channels.hdf5 --rtol 1e-3
python3 NuRadioMC/test/SingleEvents/T05validate_nur_file.py NuRadioMC/examples/01_Veff_simulation_B/1e19_output.nur NuRadioMC/examples/01_Veff_simulation_B/1e19_output_trigger_channels.nur 6


# cleanup 
rm -v NuRadioMC/examples/01_Veff_simulation_B/{1e19_output.hdf5,1e19_output.nur,1e19_output_trigger_channels.hdf5,1e19_output_trigger_channels.nur}