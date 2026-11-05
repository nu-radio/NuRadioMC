#!/bin/bash

set -e

dir=NuRadioMC/test/Veff/1e18eV

# writes 1e18_full.hdf5 into $dir
"$dir/T01generate_event_list.py"

# resolves all paths relative to $dir
"$dir/T02RunSimulation.py" 1e18_full.hdf5  ../dipole_100m.json ../config.yaml output.hdf5 output.nur

# resolves the path relative to $dir
"$dir/T03check_output.py" output.hdf5

# cleanup
rm -v "$dir"/{output.nur,output.hdf5,1e18_full.hdf5}
