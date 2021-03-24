set -e
cd NuRadioReco/test/tiny_reconstruction
python3 TinyReconstruction.py
python3 compareToReference.py MC_example_station_32.nur reference.json
