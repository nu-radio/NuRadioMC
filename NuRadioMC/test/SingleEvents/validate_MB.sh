#!/bin/bash
set -e

# usage: $0 [--create_reference]
# --create_reference: overwrite the reference file with the new output instead of validating against it
# (only after an intended physics change). T02 reads its input events from the reference hdf5 file, so the new output
# is moved over it only after the simulation has finished.
CREATE_REFERENCE=false
if [ "$1" == "--create_reference" ]; then
    CREATE_REFERENCE=true
elif [ -n "$1" ]; then
    echo "usage: $0 [--create_reference]" >&2
    exit 1
fi

DIR=NuRadioMC/test/SingleEvents

"$DIR/T02RunSimulation.py" "$DIR/MB_1e18_reference.hdf5" "$DIR/surface_station_1GHz.json" "$DIR/config_MB.yaml" "$DIR/MB_1e18_output.hdf5"
if [ "$CREATE_REFERENCE" == true ]; then
    mv -v "$DIR/MB_1e18_output.hdf5" "$DIR/MB_1e18_reference.hdf5"
    exit 0
fi
"$DIR/T04validate_allmost_equal.py" "$DIR/MB_1e18_output.hdf5" "$DIR/MB_1e18_reference.hdf5"

# cleanup
rm -v "$DIR/MB_1e18_output.hdf5"
