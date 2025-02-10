from NuRadioMC.SignalGen.askaryan import get_time_trace, get_frequency_spectrum
from NuRadioReco.utilities import units
import numpy as np
import pickle

np.random.seed(0)

E = 1e20 * units.eV
n_index = 1.78
domega = 0.05 * units.deg
theta = np.arccos(1. / n_index) + domega

dt = 0.5 * units.ns
n_samples = 256
ff = np.fft.rfftfreq(n_samples, dt)
tt = np.arange(0, n_samples * dt, dt)
R = 1 * units.km

models = ['Alvarez2009', 'ARZ2019', 'Alvarez2000', 'ARZ2020']
shower_types = ['EM', 'HAD']

Es = 10 ** np.linspace(15, 19, 5) * units.eV
domegas = np.linspace(-5, 5, 10) * units.deg
thetas = np.arccos(1. / n_index) + domegas

output = []
for model in models:
    for E in Es:
        for shower_type in shower_types:
            for theta in thetas:
                trace = get_time_trace(E, theta, n_samples, dt, shower_type, n_index, R, model, seed=1234)
                output.append(trace)

with open("reference_v1.pkl", "wb") as fout:
    pickle.dump(output, fout, protocol=4)
