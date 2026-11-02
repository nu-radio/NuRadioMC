#!/bin/bash

set -e

python3 NuRadioReco/test/LOFAR/test_lofar_data_pipeline.py
python3 NuRadioReco/test/LOFAR/test_lofar_simulation_pipeline.py

rm -rf NuRadioReco/test/LOFAR/test-data/