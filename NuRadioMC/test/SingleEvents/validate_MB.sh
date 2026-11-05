#!/bin/bash
set -e

dir=NuRadioMC/test/SingleEvents

"$dir/T02RunSimulation.py" "$dir/MB_1e18_reference.hdf5" "$dir/surface_station_1GHz.json" "$dir/config_MB.yaml" "$dir/MB_1e18_output.hdf5"
"$dir/T04validate_allmost_equal.py" "$dir/MB_1e18_output.hdf5" "$dir/MB_1e18_reference.hdf5"

# cleanup
rm -v "$dir/MB_1e18_output.hdf5"
