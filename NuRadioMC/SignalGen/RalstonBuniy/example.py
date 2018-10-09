from create_askaryan import get_time_trace, get_frequency_spectrum
from utilities import units
from matplotlib import pyplot as plt
import numpy as np

n_trace = 2 ** 12  # samples of trace
n_trace2 = 2 ** 10  # samples of trace
dt = 0.1 * units.ns
tt = np.arange(0, n_trace * dt, dt)
ff = np.fft.rfftfreq(n_trace, dt)
df = ff[1] - ff[0]
tt2 = np.arange(0, n_trace2 * dt, dt)
ff2 = np.fft.rfftfreq(n_trace2, dt)
df2 = ff2[1] - ff2[0]

spectrum = get_frequency_spectrum(1 * units.EeV, 55.8 * units.deg, ff, 1)
spectrum2 = get_frequency_spectrum(1 * units.EeV, 55.8 * units.deg, ff2, 1)
fig2, (ax, ax2) = plt.subplots(1, 2)
ax2.plot(ff / units.MHz, np.abs(spectrum[0]) / units.mV * units.m * units.MHz, 'C0-', label='eR')
ax2.plot(ff / units.MHz, np.abs(spectrum[1]) / units.mV * units.m * units.MHz, 'C1-', label='eTheta')
ax2.plot(ff2 / units.MHz, np.abs(spectrum2[1]) / units.mV * units.m * units.MHz, 'C3--', label='eTheta, coars')
ax2.plot(ff / units.MHz, np.abs(spectrum[2]) / units.mV * units.m * units.MHz, 'C2-', label='ePhi')
ax2.set_xlabel("frequency [MHz]")
ax2.set_ylabel("amplitude [mV/m/MHz]")

traceR = np.fft.irfft(spectrum[0] * df ** 0.5, norm='ortho') / 2 ** 0.5
traceTheta = np.fft.irfft(spectrum[1] * df ** 0.5, norm='ortho') / 2 ** 0.5
traceTheta2 = np.fft.irfft(spectrum2[1] * df2 ** 0.5, norm='ortho') / 2 ** 0.5
tracePhi = np.fft.irfft(spectrum[2] * df ** 0.5, norm='ortho') / 2 ** 0.5

ax.plot(tt, traceR / units.mV * units.m, "C0-")
ax.plot(tt, traceTheta / units.mV * units.m, "C1-")
ax.plot(tt, tracePhi / units.mV * units.m, "C2-")
ax.plot(tt2, traceTheta2 / units.mV * units.m, "C3--")
# ax.set_xlim(170, 185)

tt, ex, ey, ez = get_time_trace(1 * units.EeV, 55.8 * units.deg, ff, 1)
ax.plot(tt, ey / units.mV * units.m, "C4:")

ax2.legend()

ax.set_xlabel("time [ns]")
ax.set_ylabel("electric-field [mV/m]")
ax.legend()
fig2.tight_layout()
fig2.savefig("example.png")

plt.show()

