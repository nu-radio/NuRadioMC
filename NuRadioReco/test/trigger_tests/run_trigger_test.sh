#!/bin/bash

python NuRadioReco/test/trigger_tests/generate_events.py
python NuRadioReco/test/trigger_tests/trigger_tests.py
python NuRadioReco/test/trigger_tests/compare_to_reference.py
