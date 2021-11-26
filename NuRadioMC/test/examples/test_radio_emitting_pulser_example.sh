#!/bin/bash
set -e
cd NuRadioMC/examples/05_pulser_calibration_measurement/radioEmitting_pulser_calibration
python A01generate_pulser_events.py
python runARA02.py emitter_event_list.hdf5 ARA02Alt.json config.yaml output.hdf5 output.nur
rm emitter_event_list.hdf5
rm output.hdf5
rm output.nur
