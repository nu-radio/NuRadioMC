#!/bin/bash
set -e

DIR=NuRadioMC/test/SingleEvents

"$DIR/T02RunSimulation.py" "$DIR/1e18_output_ARZ_reference.hdf5" "$DIR/surface_station_1GHz.json" "$DIR/config_ARZ.yaml" "$DIR/1e18_output_ARZ.hdf5"
"$DIR/T04validate_allmost_equal.py" "$DIR/1e18_output_ARZ.hdf5" "$DIR/1e18_output_ARZ_reference.hdf5"

# cleanup
rm -v "$DIR/1e18_output_ARZ.hdf5"
