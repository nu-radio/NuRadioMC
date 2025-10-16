#!/bin/bash

set -e

if [ -f NuRadioReco/test/data/station23/run325/combined.root ]; then
    echo "Using existing data file at NuRadioReco/test/data/station23/run325/combined.root"
else
    wget https://rnog-data.zeuthen.desy.de/rnog_share/forced_triggers/station23_run325.root
    mkdir -p NuRadioReco/test/data/station23/run325
    mv station23_run325.root NuRadioReco/test/data/station23/run325/combined.root
fi

python3 NuRadioReco/examples/RNOG/data_analysis_example.py NuRadioReco/test/data/station23/run325/combined.root --outputfile NuRadioReco/examples/RNOG/processed.nur
