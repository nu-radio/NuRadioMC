#!/bin/bash
set -e

dir=NuRadioMC/test/SingleEvents

"$dir/T02RunSimulation.py" "$dir/1e18_output_ARZ_reference.hdf5" "$dir/surface_station_1GHz.json" "$dir/config_ARZ.yaml" "$dir/1e18_output_ARZ.hdf5"
"$dir/T04validate_allmost_equal.py" "$dir/1e18_output_ARZ.hdf5" "$dir/1e18_output_ARZ_reference.hdf5"

# cleanup
rm -v "$dir/1e18_output_ARZ.hdf5"
