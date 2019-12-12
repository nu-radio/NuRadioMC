import NuRadioMC.SignalGen.RalstonBuniy.askaryan_module as AskaryanModule
import NuRadioMC.SignalGen.parametrizations as param
from NuRadioReco.utilities import units, fft
import numpy as np
import matplotlib.pyplot as plt

E = 1e17 * units.eV
n_index = 1.78
domega = 2.5 * units.deg
theta = np.arccos(1. / n_index) + domega

dt = .1 * units.ns
n_samples = 2048
ff = np.fft.rfftfreq(n_samples, dt)
tt = np.arange(0, n_samples * dt, dt)
R = 5 * units.km
em = False

df = ff[1]  # width of a frequency bin

spec1 = AskaryanModule.get_frequency_spectrum(E, theta, ff, em, n_index, R, True)
spec1_noLPM = AskaryanModule.get_frequency_spectrum(E, theta, ff, em, n_index, R, False)
spec2 = param.get_frequency_spectrum(E, theta, ff, em, n_index, R, model='Alvarez2012')
spec3 = param.get_frequency_spectrum(E, theta, ff, em, n_index, R, model='Alvarez2000')
spec4 = param.get_frequency_spectrum(E, theta, ff, em, n_index, R, model='ZHS1992')

fig, ax = plt.subplots(1, 1)
# ax.plot(ff / units.MHz, np.abs(spec1[0]) / units.V * units.m, 'o', label='Askaryan module eR (without LPM)')
ax.plot(ff / units.MHz, np.abs(spec1[1]) / units.V * units.m, 'o', label='Askaryan module eTheta (with LPM)')
ax.plot(ff / units.MHz, np.abs(spec1_noLPM[1]) / units.V * units.m, 'o', label='Askaryan module eTheta (without LPM)')
ax.plot(ff / units.MHz, np.abs(spec2) / units.V * units.m, 'd', label='pyrex Alvarez2012')
ax.plot(ff / units.MHz, np.abs(spec3) / units.V * units.m, 's', label='shelfmc Alvarez2000')
ax.plot(ff / units.MHz, np.abs(spec4) / units.V * units.m, '>', label='ZHS1992')
ax.semilogx(True)
ax.semilogy(True)
ax.set_xlim(10, 2e3)
ax.set_ylim(1e-9, 1e-3)
ax.set_xlabel("frequency [MHz]")
ax.set_ylabel("amplitude [V/m] per {:.1f}MHz".format(df / units.MHz))
ax.legend(fontsize='small')
ax.set_title("E = {:.1e}eV, $\Delta\Omega$ = {:.1f}deg, R = {:.1f}km, EM={}".format(E/units.eV, domega / units.deg, R / units.km, em))
fig.tight_layout()
if(em):
    fig.savefig("comparison_em.png")
else:
    fig.savefig("comparison_had.png")
plt.show()

# test if different models are normalized such that they give the same time trace
# independent of the bin width

# 1st askaryan module
fig, ax = plt.subplots(1, 1)
n_trace = 2 ** 10  # samples of trace
dt = 0.1 * units.ns
tt = np.arange(0, n_trace * dt, dt)
ff = np.fft.rfftfreq(n_trace, dt)

trace = AskaryanModule.get_time_trace(E, theta, ff, 0, n=1.78, R=R)
ax.plot(tt / units.ns, np.abs(trace[0]), label="eR dt = {:.2f}ns, n={}".format(dt / units.ns, n_trace))
ax.plot(tt / units.ns, np.abs(trace[1]), label="dt = {:.2f}ns, n={}".format(dt / units.ns, n_trace))
print('maximum amplitude for {:d} samples is: {:4g}V/m'.format(n_trace, np.abs(trace).max()))

n_trace = 2 ** 12  # samples of trace
dt = 0.1 * units.ns
tt = np.arange(0, n_trace * dt, dt)
ff = np.fft.rfftfreq(n_trace, dt)
trace = AskaryanModule.get_time_trace(E, theta, ff, 0, n=1.78, R=R)
ax.plot(tt / units.ns, np.abs(trace[0]), label="eR dt = {:.2f}ns, n={}".format(dt / units.ns, n_trace))
ax.plot(tt / units.ns, np.abs(trace[1]), label="dt = {:.2f}ns, n={}".format(dt / units.ns, n_trace))
print('maximum amplitude for {:d} samples is: {:4g}V/m'.format(n_trace, np.abs(trace).max()))
ax.legend()
ax.set_title("askaryan module")
plt.show()
a = 1 / 0

# 2nd pyrex
fig, ax = plt.subplots(1, 1)
n_trace = 2 ** 10  # samples of trace
dt = 0.1 * units.ns
tt = np.arange(0, n_trace * dt, dt)
ff = np.fft.rfftfreq(n_trace, dt)

trace = fft.freq2time(param.get_frequency_spectrum(E, theta, ff, 0, n_index, R, model='Alvarez2012'), 1/dt)
ax.plot(tt / units.ns, trace, label="dt = {:.2f}ns, n={}".format(dt / units.ns, n_trace))

n_trace = 2 ** 12  # samples of trace
dt = 0.1 * units.ns
tt = np.arange(0, n_trace * dt, dt)
ff = np.fft.rfftfreq(n_trace, dt)
trace = fft.freq2time(param.get_frequency_spectrum(E, theta, ff, 0, n_index, R, model='Alvarez2012'), 1/dt)
ax.plot(tt / units.ns, trace, '--', label="dt = {:.2f}ns, n={}".format(dt / units.ns, n_trace))
ax.legend()
ax.set_title("pyrex")
plt.show()
