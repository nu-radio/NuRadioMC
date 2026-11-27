#!/bin/bash
set -e

EXAMPLE=NuRadioMC/examples/05_pulser_calibration_measurement/ARA

python "$EXAMPLE/A01generate_pulser_events.py"

python "$EXAMPLE/runARA02.py" \
    emitter_event_list.hdf5 \
    "$EXAMPLE/ARA02Alt.json" \
    "$EXAMPLE/config.yaml" \
    output.hdf5 output.nur

python NuRadioMC/test/examples/validate_radio_emitter_allmost_equal.py output.hdf5 \
    NuRadioMC/test/examples/radio_emitter_output_reference.hdf5

rm emitter_event_list.hdf5 output.hdf5 output.nur
