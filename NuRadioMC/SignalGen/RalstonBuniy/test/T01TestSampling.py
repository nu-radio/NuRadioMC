import NuRadioMC.SignalGen.RalstonBuniy.askaryan_module as AskaryanModule
from NuRadioMC.utilities import units
from matplotlib import pyplot as plt
import numpy as np

T = 100 * units.ns
LPM = False
EM = True

dt = 0.01 * units.ns
n_trace = 1024
tt = np.arange(0, n_trace * dt, dt)
ff = np.fft.rfftfreq(n_trace, dt)
df = ff[1] - ff[0]
cherenkov_angle = np.arccos(1. / 1.78)

R = 1100 * units.m
theta = 57 * units.deg
E = 1e9 * units.GeV

spectrum = AskaryanModule.get_frequency_spectrum(E, theta, n_trace, dt, EM, 1.78, R, LPM=LPM)
trace = AskaryanModule.get_time_trace(E, theta, n_trace, dt, EM, 1.78, R, LPM=LPM)

fig, (ax, ax2) = plt.subplots(1, 2)
ax.plot(ff/units.GHz, np.abs(spectrum[1]), '-')
ax2.plot(tt/units.ns, trace[1], label='n = {}'.format(n_trace), alpha=1)
ax.set_xlabel("frequency [GHz]")
ax2.set_xlabel("time [ns]")
ax.semilogy(True)

n_trace = 2048
tt = np.arange(0, n_trace * dt, dt)
ff = np.fft.rfftfreq(n_trace, dt)
df = ff[1] - ff[0]
spectrum = AskaryanModule.get_frequency_spectrum(E, theta, n_trace, dt, EM, 1.78, R, LPM=LPM)
trace = AskaryanModule.get_time_trace(E, theta, n_trace, dt, EM, 1.78, R, LPM=LPM)
ax.plot(ff/units.GHz, np.abs(spectrum[1]), '--')
ax2.plot(tt/units.ns, trace[1], '--', label='n = {}'.format(n_trace), alpha=1)

n_trace = 8192
tt = np.arange(0, n_trace * dt, dt)
ff = np.fft.rfftfreq(n_trace, dt)
df = ff[1] - ff[0]
spectrum = AskaryanModule.get_frequency_spectrum(E, theta, n_trace, dt, EM, 1.78, R, LPM=LPM)
trace = AskaryanModule.get_time_trace(E, theta, n_trace, dt, EM, 1.78, R, LPM=LPM)
ax.plot(ff/units.GHz, np.abs(spectrum[1]), ':')
ax2.plot(tt/units.ns, trace[1], ':', label='n = {}'.format(n_trace), alpha=1)
# 
# ax.set_xlim(0, 5)
ax2.legend()

fig.tight_layout()
fig.savefig("plots/T01.png")
plt.show()
