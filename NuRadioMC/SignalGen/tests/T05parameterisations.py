from NuRadioMC.SignalGen.parametrizations import get_time_trace
from NuRadioMC.utilities import units, fft
import numpy as np
import matplotlib.pyplot as plt

E = 1e20 * units.eV
n_index = 1.78
domega = 1 * units.deg
theta = np.arccos(1. / n_index) + domega

dt = .01 * units.ns
n_samples = 2048
ff = np.fft.rfftfreq(n_samples, dt)
tt = np.arange(0, n_samples * dt, dt)
R = 1 * units.km

models = ['Alvarez2009', 'Alvarez2000']
colours = ['blue', 'orange', 'red', 'black']
col_dict = {}
linestyle = {'Alvarez2009': '-', 'Alvarez2000': '--'}
shower_type = 'EM'

Es = 10**np.linspace(18,20,3) * units.eV
domegas = np.linspace(-5,5,5) * units.deg
thetas = np.arccos(1./n_index) + domegas

for iE, E in enumerate(Es):
    col_dict[E] = colours[iE]

for model in models:

    for E in Es:

        trace = get_time_trace(E, theta, n_samples, dt, shower_type, n_index, R, model)
        trace /= units.mV/units.m

        if (model=='Alvarez2009'):
            label = str(E/units.EeV)+' EeV'
        else:
            label = ''

        plt.plot(tt, trace, color=col_dict[E], label=label, linestyle=linestyle[model])

plt.xlabel('Time [ns]')
plt.ylabel('Electric field [mV/m]')
plt.legend()
plt.show()
