from create_askaryan import get_frequency_spectrum
from NuRadioMC.utilities import units
from matplotlib import pyplot as plt
import numpy as np

n_trace = 2 ** 12  # samples of trace
dt = 0.1 * units.ns
tt = np.arange(0, n_trace * dt, dt)
ff = np.fft.rfftfreq(n_trace, dt)
df = ff[1] - ff[0]
cherenkov_angle = np.arccos(1. / 1.78)
R = 500 * units.m
E = 100 * units.TeV

fig, ax = plt.subplots(1, 1)
for theta in np.arange(0, 10.5, 2.5) * units.deg:
    spectrum = get_frequency_spectrum(E, theta + cherenkov_angle, ff, 0, n=1.78, R=R)
    yy = np.abs(spectrum[1]) * R / E / units.V * units.MHz * units.TeV
    ax.plot(ff / units.MHz, yy, label=r'$\Theta - \Theta_C$ = {:.1f}'.format(theta / units.deg))
ax.set_xlim(1, 2e3)
ax.set_ylim(1e-11, 1e-5)
ax.legend()
ax.semilogx(True)
ax.semilogy(True)
ax.set_xlabel("Frequency [MHz]")
ax.set_ylabel("R E/E_C [V/MHz/TeV]")
fig.tight_layout()
fig.savefig("fig5.png")
plt.show()

# distance dependence

E = 100 * units.PeV
fig, ax = plt.subplots(1, 1)
theta = cherenkov_angle + 2.5 * units.deg
for R in [100 * units.m, 500 * units.m, 2 * units.km, 5 * units.km, 100 * units.km]:
    spectrum = get_frequency_spectrum(E, theta, ff, 1, n=1.78, R=R)
    yy = np.abs(spectrum[1]) * R / E / units.V * units.MHz * units.TeV
    ax.plot(ff / units.MHz, yy, label=r'R = {:.0f}m'.format(R / units.m))
ax.legend()
ax.semilogx(True)
ax.semilogy(True)
ax.set_xlabel("Frequency [MHz]")
ax.set_ylabel("R E/E_C [V/MHz/TeV]")
fig.tight_layout()
plt.show()
