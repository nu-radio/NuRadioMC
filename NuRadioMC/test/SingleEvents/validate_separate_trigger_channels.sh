#!/bin/bash
set -e

DIR=NuRadioMC/test/SingleEvents
EXAMPLE=NuRadioMC/examples/01_Veff_simulation_B
DET="$EXAMPLE/surface_station_1GHz.json"
CONFIG="$EXAMPLE/config_no_noise.yaml"

python3 "$EXAMPLE/T02RunSimulation.py" "$EXAMPLE/1e19_n1e3.hdf5" "$DET" "$CONFIG" "$EXAMPLE/1e19_output.hdf5" "$EXAMPLE/1e19_output.nur"

python3 "$EXAMPLE/T02RunSimulation_trigger_channels.py" "$EXAMPLE/1e19_n1e3.hdf5" "$DET" "$CONFIG" "$EXAMPLE/1e19_output_trigger_channels.hdf5" "$EXAMPLE/1e19_output_trigger_channels.nur"
# larger tolerance: for the non-trigger channels of the second simulation, the detector response is applied
# per efield and the traces are summed afterwards (instead of applying it to the summed efields), which changes
# the (envelope) maximum amplitudes by up to ~7.5e-4
python3 "$DIR/T04validate_allmost_equal.py" "$EXAMPLE/1e19_output.hdf5" "$EXAMPLE/1e19_output_trigger_channels.hdf5" --rtol 1e-3
python3 "$DIR/T05validate_nur_file.py" "$EXAMPLE/1e19_output.nur" "$EXAMPLE/1e19_output_trigger_channels.nur" 6

# cleanup
rm -v "$EXAMPLE"/{1e19_output.hdf5,1e19_output.nur,1e19_output_trigger_channels.hdf5,1e19_output_trigger_channels.nur}
