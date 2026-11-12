#!/bin/bash

set -e

DIR=NuRadioReco/test/trigger_tests

python3 "$DIR/generate_events.py"
python3 "$DIR/trigger_tests.py"
python3 "$DIR/compare_to_reference.py"

# clean up
rm -v "$DIR"/{trigger_test_input.hdf5,trigger_test_input.nur,trigger_test_output.nur}
