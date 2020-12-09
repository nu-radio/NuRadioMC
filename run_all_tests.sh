#!/bin/bash
set -e

NuRadioMC/test/SingleEvents/test_build.sh
NuRadioMC/test/SingleEvents/validate_MB.sh
NuRadioMC/test/SingleEvents/validate_ARZ.sh
NuRadioMC/test/SignalGen/test_build.sh
NuRadioMC/test/SignalProp/run_signal_test.sh
NuRadioMC/test/Veff/1e18eV/test_build.sh
NuRadioMC/test/atmospheric_Aeff/1e18eV/test_build.sh
NuRadioMC/test/examples/test_examples.sh
NuRadioReco/test/tiny_reconstruction/testTinyReconstruction.sh
NuRadioReco/test/trigger_tests/run_trigger_test.sh
NuRadioReco/test/test_examples.sh