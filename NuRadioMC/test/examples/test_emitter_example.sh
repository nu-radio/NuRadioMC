set -e
cd NuRadioMC/examples/05_pulser_calibration_measurement/radioEmitting_pulser_calibration
python3 A01generate_pulser_events.py
python runARA02.py emitter_event_list.hdf5 ARA02Alt.json config.yaml output.hdf5 output.nur
rm output.nur
rm output.hdf5
