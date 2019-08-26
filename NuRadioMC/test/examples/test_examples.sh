cd NuRadioMC/examples/01_Veff_simulation
mkdir output
python T01generate_event_list.py
python T02RunSimulation.py 1e19_n1e3.hdf5 surface_station_1GHz.json config.yaml output/output.hdf5 output.nur
python T03visualizeVeff.py

cd ../02_DnR
python E01detector_simulation.py event_input/1e19_n1e3comparison1.hdf5 detector/string_to_100m.json config.yaml output.hdf5
python A01analyse_simulation.py output.hdf5
