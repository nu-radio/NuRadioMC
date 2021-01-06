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