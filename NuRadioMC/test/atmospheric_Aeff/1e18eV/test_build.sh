#!/bin/bash

set -e

dir=NuRadioMC/test/atmospheric_Aeff/1e18eV

# writes 1e18_full.hdf5 and tables/ into $dir
"$dir/T01generate_event_list.py"

# resolves all paths relative to $dir
"$dir/T02RunSimulation.py" 1e18_full.hdf5  ../dipole_100m.json ../config.yaml output.hdf5 output.nur

# path relative to the working directory
"$dir/T03check_output.py" "$dir/output.hdf5"

# tables/dummy is tracked
rm -r "$dir/tables/Greenland"
rm "$dir"/{output*,1e18_full.hdf5}
