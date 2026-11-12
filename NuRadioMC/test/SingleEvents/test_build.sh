#!/bin/bash
set -e

DIR=NuRadioMC/test/SingleEvents
DET="$DIR/surface_station_1GHz.json"

python3 "$DIR/T02RunSimulation.py" "$DIR/1e18_output_reference.hdf5" "$DET" "$DIR/config.yaml" "$DIR/1e18_output.hdf5" "$DIR/1e18_output.nur"
# larger tolerance: config.yaml uses focusing, whose numerical calculation amplifies platform-dependent
# ray-tracing differences (~1e-8) to up to ~5.2e-4 in the amplitudes (Linux CI vs. this reference)
python3 "$DIR/T04validate_allmost_equal.py" "$DIR/1e18_output.hdf5" "$DIR/1e18_output_reference.hdf5" --rtol 1e-3
python3 "$DIR/T05validate_nur_file.py" "$DIR/1e18_output.nur" "$DIR/1e18_output_reference.nur"
python3 "$DIR/T02RunSimulation.py" "$DIR/1e18_output_reference.hdf5" "$DET" "$DIR/config_noise.yaml" "$DIR/1e18_output_noise.hdf5"
python3 "$DIR/T04validate_allmost_equal.py" "$DIR/1e18_output_noise.hdf5" "$DIR/1e18_output_noise_reference.hdf5"

# cleanup
rm -v "$DIR"/{1e18_output_noise.hdf5,1e18_output.hdf5,1e18_output.nur}
