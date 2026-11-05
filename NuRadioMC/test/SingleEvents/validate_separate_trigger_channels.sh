#!/bin/bash
set -e

dir=NuRadioMC/test/SingleEvents
example=NuRadioMC/examples/01_Veff_simulation_B
det="$example/surface_station_1GHz.json"
config="$example/config_no_noise.yaml"

python3 "$example/T02RunSimulation.py" "$example/1e19_n1e3.hdf5" "$det" "$config" "$example/1e19_output.hdf5" "$example/1e19_output.nur"

python3 "$example/T02RunSimulation_trigger_channels.py" "$example/1e19_n1e3.hdf5" "$det" "$config" "$example/1e19_output_trigger_channels.hdf5" "$example/1e19_output_trigger_channels.nur"
# larger tolerance: for the non-trigger channels of the second simulation, the detector response is applied
# per efield and the traces are summed afterwards (instead of applying it to the summed efields), which changes
# the (envelope) maximum amplitudes by up to ~7.5e-4
python3 "$dir/T04validate_allmost_equal.py" "$example/1e19_output.hdf5" "$example/1e19_output_trigger_channels.hdf5" --rtol 1e-3
python3 "$dir/T05validate_nur_file.py" "$example/1e19_output.nur" "$example/1e19_output_trigger_channels.nur" 6

# cleanup
rm -v "$example"/{1e19_output.hdf5,1e19_output.nur,1e19_output_trigger_channels.hdf5,1e19_output_trigger_channels.nur}
