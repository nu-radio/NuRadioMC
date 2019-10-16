from NuRadioMC.SignalGen.askaryan import get_time_trace, get_frequency_spectrum
from NuRadioReco.utilities import units
import numpy as np
import matplotlib.pyplot as plt

E = 1e20 * units.eV
n_index = 1.78
domega = 0.05 * units.deg
theta = np.arccos(1. / n_index) + domega

dt = 0.1 * units.ns
n_samples = 2048
ff = np.fft.rfftfreq(n_samples, dt)
tt = np.arange(0, n_samples * dt, dt)
R = 10 * units.km

models = ['Alvarez2009', 'ARZ2019', 'Alvarez2000']
colours = ['blue', 'orange', 'red', 'black']
col_dict = {}
linestyle = {'Alvarez2009': '-', 'ARZ2019': '--', 'Alvarez2000': '-.'}
shower_type = 'EM'

Es = 10**np.linspace(15,17,3) * units.eV
domegas = np.linspace(-5,5,5) * units.deg
thetas = np.arccos(1./n_index) + domegas

for iE, E in enumerate(Es):
    col_dict[E] = colours[iE]

for model in models:

    for E in Es:

        spectrum = get_frequency_spectrum(E, theta, n_samples, dt, shower_type, n_index, R, model)
        spectrum *= R
        freqs = np.fft.rfftfreq(n_samples, dt)

        if (model=='Alvarez2009'):
            label = str(E/units.EeV)+' EeV'
        else:
            label = ''

        #plt.plot(tt, trace, color=col_dict[E], label=label, linestyle=linestyle[model])
        plt.loglog(freqs, np.abs(spectrum) / units.V * units.MHz, color=col_dict[E], label=label, linestyle=linestyle[model])

#plt.xlabel('Time [ns]')
plt.xlabel('Frequency [GHz]')
plt.ylabel('Electric field [V/MHz]')
plt.legend()
plt.show()
