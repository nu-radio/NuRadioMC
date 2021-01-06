#!/bin/bash

set -e
cd NuRadioMC/test/Veff/1e18eV/
python T01generate_event_list_noise.py
python D05phased_array_deep.py 1e18_full_noise.hdf5  ../single_pa_200m.json ../config_noise.yaml output_noise.hdf5 output_noise.nur
python T03check_output_noise.py
