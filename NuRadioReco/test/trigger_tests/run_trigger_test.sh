#!/bin/bash

set -e
python3 NuRadioReco/test/trigger_tests/generate_events.py
python3 NuRadioReco/test/trigger_tests/trigger_tests.py
python3 NuRadioReco/test/trigger_tests/compare_to_reference.py

# clean up 
rm -v NuRadioReco/test/trigger_tests/trigger_test_input.hdf5
rm -v NuRadioReco/test/trigger_tests/trigger_test_input.nur
rm -v NuRadioReco/test/trigger_tests/trigger_test_output.nur