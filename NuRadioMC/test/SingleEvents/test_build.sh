#!/bin/bash
set -e

dir=NuRadioMC/test/SingleEvents
det="$dir/surface_station_1GHz.json"

python3 "$dir/T02RunSimulation.py" "$dir/1e18_output_reference.hdf5" "$det" "$dir/config.yaml" "$dir/1e18_output.hdf5" "$dir/1e18_output.nur"
# larger tolerance: config.yaml uses focusing, whose numerical calculation amplifies platform-dependent
# ray-tracing differences (~1e-8) to up to ~5.2e-4 in the amplitudes (Linux CI vs. this reference)
python3 "$dir/T04validate_allmost_equal.py" "$dir/1e18_output.hdf5" "$dir/1e18_output_reference.hdf5" --rtol 1e-3
python3 "$dir/T05validate_nur_file.py" "$dir/1e18_output.nur" "$dir/1e18_output_reference.nur"
python3 "$dir/T02RunSimulation.py" "$dir/1e18_output_reference.hdf5" "$det" "$dir/config_noise.yaml" "$dir/1e18_output_noise.hdf5"
python3 "$dir/T04validate_allmost_equal.py" "$dir/1e18_output_noise.hdf5" "$dir/1e18_output_noise_reference.hdf5"

# cleanup
rm -v "$dir"/{1e18_output_noise.hdf5,1e18_output.hdf5,1e18_output.nur}
