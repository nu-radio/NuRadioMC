from create_askaryan import get_time_trace, get_frequency_spectrum
from utilities import units
from matplotlib import pyplot as plt
import numpy as np

ff = np.linspace(100 * units.MHz, 5 * units.GHz, 100)
spectrum = get_frequency_spectrum(1 * units.EeV, 55.8 * units.deg, ff, 1)
fig2, ax2 = plt.subplots(1, 1)
ax2.plot(ff / units.MHz, np.abs(spectrum) / units.mV * units.m * units.MHz)
ax2.set_xlabel("frequency [MHz]")
ax2.set_ylabel("amplitude [mV/m/MHz]")

tt, ex, ey, ez = get_time_trace(1 * units.EeV, 55.8 * units.deg, 100 * units.MHz, 5 * units.GHz, 10 * units.MHz, 1)
fig, ax = plt.subplots(1, 1)
ax.plot(tt / units.ns, ex / units.mV * units.m, label='eR')
ax.plot(tt / units.ns, ey / units.mV * units.m, label='eTheta')
ax.plot(tt / units.ns, ez / units.mV * units.m, label='ePhi')
ax.set_xlabel("time [ns]")
ax.set_ylabel("electric-field [mV/m]")
ax.legend()
plt.show()

