#!/bin/bash

set -e

DATA_DIR=NuRadioReco/test/data/station23/run325

if [ -f "$DATA_DIR/combined.root" ]; then
    echo "Using existing data file at $DATA_DIR/combined.root"
else
    wget https://rnog-data.zeuthen.desy.de/rnog_share/forced_triggers/station23_run325.root
    mkdir -p "$DATA_DIR"
    mv station23_run325.root "$DATA_DIR/combined.root"
fi

python3 NuRadioReco/examples/RNOG/data_analysis_example.py "$DATA_DIR/combined.root" --outputfile NuRadioReco/examples/RNOG/processed.nur

# cleanup
rm -v NuRadioReco/examples/RNOG/processed.nur
rm -fv channel_traces_*.png  # number of plots depends on the data
