cd NuRadioMC/examples/01_Veff_simulation
mkdir output
python T01generate_event_list.py
python T02RunSimulation.py 1e19_n1e3.hdf5 surface_station_1GHz.json config.yaml output/output.hdf5 output.nur
python T03visualizeVeff.py