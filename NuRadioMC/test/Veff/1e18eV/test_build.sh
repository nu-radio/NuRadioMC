#!/bin/bash

set -e

DIR=NuRadioMC/test/Veff/1e18eV

# writes 1e18_full.hdf5 into $DIR
"$DIR/T01generate_event_list.py"

# resolves all paths relative to $DIR
"$DIR/T02RunSimulation.py" 1e18_full.hdf5  ../dipole_100m.json ../config.yaml output.hdf5 output.parquet

# resolves the path relative to $DIR
"$DIR/T03check_output.py" output.hdf5

# cleanup
rm -v "$DIR"/{output.parquet,output_event.parquet,output.hdf5,1e18_full.hdf5}
