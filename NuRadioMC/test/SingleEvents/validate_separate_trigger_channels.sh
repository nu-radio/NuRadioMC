#!/bin/bash
set -e

EX=NuRadioMC/examples/01_Veff_simulation_B
TEST=NuRadioMC/test/SingleEvents
INPUTS="$EX/1e19_n1e3.hdf5 $EX/surface_station_1GHz.json $EX/config_no_noise.yaml"
OUT=$EX/1e19_output
OUT_TRIG=$EX/1e19_output_trigger_channels

python3 $EX/T02RunSimulation.py $INPUTS $OUT.hdf5 $OUT.nur

python3 $EX/T02RunSimulation_trigger_channels.py $INPUTS $OUT_TRIG.hdf5 $OUT_TRIG.nur
python3 $TEST/T04validate_allmost_equal.py $OUT.hdf5 $OUT_TRIG.hdf5
python3 $TEST/T05validate_nur_file.py $OUT.nur $OUT_TRIG.nur 6


# cleanup
rm -v $OUT.hdf5 $OUT.nur $OUT_TRIG.hdf5 $OUT_TRIG.nur
