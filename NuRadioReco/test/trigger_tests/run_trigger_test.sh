#!/bin/bash

python generate_events.py
python trigger_tests.py
python compare_to_reference.py
