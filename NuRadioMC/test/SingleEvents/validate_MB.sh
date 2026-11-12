#!/bin/bash
set -e

DIR=NuRadioMC/test/SingleEvents

"$DIR/T02RunSimulation.py" "$DIR/MB_1e18_reference.hdf5" "$DIR/surface_station_1GHz.json" "$DIR/config_MB.yaml" "$DIR/MB_1e18_output.hdf5"
"$DIR/T04validate_allmost_equal.py" "$DIR/MB_1e18_output.hdf5" "$DIR/MB_1e18_reference.hdf5"

# cleanup
rm -v "$DIR/MB_1e18_output.hdf5"
