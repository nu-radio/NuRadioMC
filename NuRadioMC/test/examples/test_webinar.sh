set -e
cd NuRadioMC/examples/06_webinar
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