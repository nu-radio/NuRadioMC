set -e
cd NuRadioMC/examples/05_pulser_calibration_measurement
python3 A01generate_pulser_events.py
python3 A02RunSimulation.py input_spice.hdf5 detector_db.json config_spice.yaml output.hdf5 output.nur
python3 A03reconstruct_sim.py output.nur
python3 A04plot_results.py
rm input_spice.hdf5
rm output.nur
rm -r plots
rm sim_results_02.pkl