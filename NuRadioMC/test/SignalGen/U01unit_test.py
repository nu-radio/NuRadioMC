#!/usr/bin/env python
from NuRadioMC.SignalGen.askaryan import get_time_trace, get_frequency_spectrum
from NuRadioReco.utilities import units
from NuRadioReco.utilities import io_utilities
import numpy as np
from numpy import testing
import sys

try:
    reference_file = sys.argv[1]
except:
    reference_file = "reference_v1.pkl"

print('Using reference file {}'.format(reference_file))

np.random.seed(0)

n_index = 1.78
domega = 0.05 * units.deg
theta = np.arccos(1. / n_index) + domega

dt = 0.5 * units.ns
n_samples = 256
R = 1 * units.km

models = ['Alvarez2009', 'ARZ2019', 'Alvarez2000', 'ARZ2020']
shower_types = ['EM', 'HAD']

Es = 10 ** np.linspace(15, 19, 5) * units.eV
domegas = np.linspace(-5, 5, 10) * units.deg
thetas = np.arccos(1. / n_index) + domegas

reference = io_utilities.read_pickle(reference_file, encoding='latin1')
i = -1
for model in models:
    for E in Es:
        for shower_type in shower_types:
            for theta in thetas:
                i += 1
                trace = get_time_trace(E, theta, n_samples, dt, shower_type, n_index, R, model, seed=1234)
                try:
                    testing.assert_almost_equal(trace, reference[i])
                except AssertionError as e:
                    print(f"error in model {model}, shower type {shower_type} theta = {theta/units.deg:.2f}deg")
                    print(e)
                    raise(e)

print('SignalGen test passed without any issues!')
