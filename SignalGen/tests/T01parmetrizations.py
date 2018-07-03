import NuRadioMC.SignalGen.RalstonBuniy.create_askaryan as AskaryanModule
import NuRadioMC.SignalGen.parametrizations as param
from NuRadioMC.utilities import units
import numpy as np
import matplotlib.pyplot as plt

E = 1e17 * units.eV
n_index = 1.78
theta = np.arccos(1. / n_index) + 2.5 * units.deg

dt = .1 * units.ns
n_samples = 2048
ff = np.fft.rfftfreq(n_samples, dt)
tt = np.arange(0, n_samples * dt, dt)
R = 5 * units.km

spec1 = AskaryanModule.get_frequency_spectrum(E, theta, ff, 0, n_index, R)
spec2 = param.get_frequency_spectrum(E, theta, ff, 0, n_index, R, model='Alvarez2012') / units.MHz
spec3 = param.get_frequency_spectrum(E, theta, ff, 0, n_index, R, model='Alvarez2000')

fig, ax = plt.subplots(1, 1)
ax.plot(ff / units.MHz, np.abs(spec1[0]) / units.V * units.m * units.MHz, label='Askaryan module eR')
ax.plot(ff / units.MHz, np.abs(spec1[1]) / units.V * units.m * units.MHz, label='Askaryan module eTheta')
ax.plot(ff / units.MHz, np.abs(spec2) / units.V * units.m * units.MHz, label='pyrex Alvarez2012')
ax.plot(ff / units.MHz, np.abs(spec3) / units.V * units.m * units.MHz, label='shelfmc Alvarez2000')
ax.semilogx(True)
ax.semilogy(True)
ax.set_xlim(1, 2e3)
ax.set_ylim(1e-11, 1e-3)
ax.set_xlabel("frequency [MHz]")
ax.set_ylabel("amplitude [V/m/MHz]")
ax.legend()
fig.tight_layout()
fig.savefig("comparison.png")
plt.show()
