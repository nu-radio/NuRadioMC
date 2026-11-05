#!/bin/bash

set -e

data_dir=NuRadioReco/test/data/station23/run325

if [ -f "$data_dir/combined.root" ]; then
    echo "Using existing data file at $data_dir/combined.root"
else
    wget https://rnog-data.zeuthen.desy.de/rnog_share/forced_triggers/station23_run325.root
    mkdir -p "$data_dir"
    mv station23_run325.root "$data_dir/combined.root"
fi

python3 NuRadioReco/examples/RNOG/data_analysis_example.py "$data_dir/combined.root" --outputfile NuRadioReco/examples/RNOG/processed.nur

# cleanup
rm -v NuRadioReco/examples/RNOG/processed.nur
rm -fv channel_traces_*.png  # number of plots depends on the data
