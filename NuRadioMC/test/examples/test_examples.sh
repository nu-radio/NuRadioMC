set -e
cd NuRadioMC/examples/01_Veff_simulation
mkdir -p output
python3 T01generate_event_list.py
python3 T02RunSimulation.py 1e19_n1e3.hdf5 surface_station_1GHz.json config.yaml output/output.hdf5 output.nur
python3 T03visualizeVeff.py
rm 1e18_n1e4.hdf5
rm 1e19_n1e3.hdf5
rm Veff.pdf
rm limits.pdf
rm output.nur
rm -r output

#cd ../02_DnR
#python3 E01detector_simulation.py event_input/1e19_n1e3comparison1.hdf5 detector/string_to_100m.json config.yaml output.hdf5
#python3 A01analyse_simulation.py output.hdf5

cd ../05_pulser_calibration_measurement
python3 A01generate_pulser_events.py
python3 A02RunSimulation.py input_spice.hdf5 detector_db.json config_spice.yaml output.hdf5 output.nur
python3 A03reconstruct_sim.py output.nur
python3 A04plot_results.py
rm input_spice.hdf5
rm output.nur
rm -r plots
rm sim_results_02.pkl

cd ../06_webinar
python3 W01_create_input.py
python3 W01_create_input_extended.py
python3 W02RunSimulation.py
python3 W03CheckOutput.py
python3 W04EffectiveVolumes.py
python3 W05ElectricFields.py
rm -r results
rm input_3.2e+19_1.0e+20.hdf5
rm input_3.2e+18_1.0e+19.hdf5.part000?
rm tables/*.txt