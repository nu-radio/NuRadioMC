#!/bin/bash

set -e

dir=NuRadioReco/test/trigger_tests

python3 "$dir/generate_events.py"
python3 "$dir/trigger_tests.py"
python3 "$dir/compare_to_reference.py"

# clean up
rm -v "$dir"/{trigger_test_input.hdf5,trigger_test_input.nur,trigger_test_output.nur}
