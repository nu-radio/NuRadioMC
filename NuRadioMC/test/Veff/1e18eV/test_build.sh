#!/bin/bash

NuRadioMC/test/Veff/1e18eV/T02RunSimulation.py 1e18_full.hdf5  ../dipole_100m.json ../config.yaml output.hdf5 output.nur

NuRadioMC/test/Veff/1e18eV/T03check_output.py