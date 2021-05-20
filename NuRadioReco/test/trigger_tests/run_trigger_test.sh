#!/bin/bash

set -e
python3 NuRadioReco/test/trigger_tests/generate_events.py
python3 NuRadioReco/test/trigger_tests/trigger_tests.py
python3 NuRadioReco/test/trigger_tests/compare_to_reference.py
