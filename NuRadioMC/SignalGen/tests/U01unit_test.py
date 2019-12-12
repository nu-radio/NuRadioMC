from NuRadioMC.SignalGen.askaryan import get_time_trace, get_frequency_spectrum
from NuRadioReco.utilities import units
from NuRadioReco.utilities import io_utilities
import numpy as np
from numpy import testing

np.random.seed(0)

n_index = 1.78
domega = 0.05 * units.deg
theta = np.arccos(1. / n_index) + domega

dt = 0.5 * units.ns
n_samples = 256
R = 1 * units.km

models = ['Alvarez2009', 'ARZ2019', 'Alvarez2000']
shower_types = ['EM', 'HAD']

Es = 10**np.linspace(15,19,5) * units.eV
domegas = np.linspace(-5,5,10) * units.deg
thetas = np.arccos(1./n_index) + domegas

reference = io_utilities.read_pickle("reference_v1.pkl")
i = -1
for model in models:
    for E in Es:
        for shower_type in shower_types:
            for theta in thetas:
                i+=1
                trace = get_time_trace(E, theta, n_samples, dt, shower_type, n_index, R, model)
                testing.assert_almost_equal(trace, reference[i])
