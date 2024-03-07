set -e
cd NuRadioMC/examples/05_pulser_calibration_measurement/SPICE_ARIANNA
python3 A01generate_pulser_events.py
python3 A02RunSimulation.py
python3 A03reconstruct_sim.py output_reco.nur
python3 A04plot_results.py
rm output_reco.nur
rm output_MC.hdf5
rm SPICE_drop_event_list.hdf5
rm -r plots
rm sim_results_03.pkl

cd ..
cd SPICE_birefringence/01_SPice_simulation_ARIANNA
python3 A01generate_pulser_events.py
python3 A02RunSimulation.py
rm input_spice.hdf5
rm output_reco.nur
rm output_MC.hdf5

cd ..
cd 02_SPice_simulation_ARA
python3 A01generate_pulser_events.py
python3 A02RunSimulation.py
rm input_spice.hdf5
rm output_reco.nur
rm output_MC.hdf5