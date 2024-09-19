#!/bin/bash

set -e

wget https://rnog-data.zeuthen.desy.de/rnog_share/forced_triggers/station23_run325.root
mkdir -p tests/data/station23/run325
mv station23_run325.root tests/data/station23/run325/combined.root

python3 NuRadioReco/examples/RNO_data/read_data_example/read_rnog.py tests/data/station23/run325/combined.root test.nur