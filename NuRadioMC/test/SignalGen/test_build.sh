#!/bin/bash

set -e
NuRadioMC/test/SignalGen/U01unit_test.py NuRadioMC/test/SignalGen/reference_v2.npy
python3 NuRadioMC/test/SignalGen/T02compare_ARZ_numba_python.py
