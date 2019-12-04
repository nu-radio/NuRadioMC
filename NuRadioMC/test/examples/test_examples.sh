set -e
cd NuRadioMC/examples/01_Veff_simulation
mkdir -p output
python3 T01generate_event_list.py
python3 T02RunSimulation.py 1e19_n1e3.hdf5 surface_station_1GHz.json config.yaml output/output.hdf5 output.nur
python3 T03visualizeVeff.py

cd ../02_DnR
python3 E01detector_simulation.py event_input/1e19_n1e3comparison1.hdf5 detector/string_to_100m.json config.yaml output.hdf5
python3 A01analyse_simulation.py output.hdf5

cd ../05_pulser_calibration_measurement
python3 A01generate_pulser_events.py
python3 A02RunSimulation.py input_spice.hdf5 detector_db.json config_spice.yaml output.hdf5 output.nur
python3 A03reconstruct_sim.py output.nur
python3 A04plot_results.py
