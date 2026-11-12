#!/bin/bash

set -e

DIR=NuRadioMC/test/atmospheric_Aeff/1e18eV

# writes 1e18_full.hdf5 and tables/ into $DIR
"$DIR/T01generate_event_list.py"

# resolves all paths relative to $DIR
"$DIR/T02RunSimulation.py" 1e18_full.hdf5  ../dipole_100m.json ../config.yaml output.hdf5 output.nur

# path relative to the working directory
"$DIR/T03check_output.py" "$DIR/output.hdf5"

# tables/dummy is tracked
rm -r "$DIR/tables/Greenland"
rm "$DIR"/{output*,1e18_full.hdf5}
