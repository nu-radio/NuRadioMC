import NuRadioMC.SignalGen.HCRB2017 as HCRB2017
import NuRadioMC.SignalGen.parametrizations as param
from NuRadioReco.utilities import units, fft
import numpy as np
import matplotlib.pyplot as plt

save_plots = False
E = 1e17 * units.eV
n_index = 1.78
domega = 2.5 * units.deg
theta = np.arccos(1. / n_index) + domega

dt = .1 * units.ns
n_samples = 2048
ff = np.fft.rfftfreq(n_samples, dt)
tt = np.arange(0, n_samples * dt, dt)
R = 5 * units.km

df = ff[1]  # width of a frequency bin

em = True
shower_type = "HAD"
if shower_type == "HAD":
    em = False


models = param.get_parametrizations()
print("Using the models: {}".format(models))

fig, ax = plt.subplots(1, 1)
fig1, ax1 = plt.subplots(1, 1)

for model in models:
    time_trace = param.get_time_trace(energy=E, theta=theta, N=n_samples, dt=dt, shower_type=shower_type, n_index=n_index, R=R, model=model)
    ax.plot(tt / units.ns ,time_trace / (units.V*units.m),label=model)
    spectrum = fft.time2freq(time_trace, 1/dt)
    ax1.plot(ff / units.MHz,np.abs(spectrum)/ (units.V*units.m),label=model)

# Needs further debugging
# spec1 = HCRB2017.get_frequency_spectrum(E, theta, ff, False, n_index, R, True)
# ax1.plot(ff / units.MHz,np.abs(spec1m)/ (units.V*units.m),label="HCRB2017")

ax1.semilogy(True)
ax1.set_ylabel("Amplitude [V/m] per {:.1f}MHz".format(df / units.MHz))
ax1.set_xlabel("Frequency [MHz]")
ax1.set_xlim(10, 2e3)
ax1.set_ylim(1e-9, 1e-3)
ax1.legend()
ax1.set_title("E = {:.1e}eV, $\Delta\Omega$ = {:.1f}deg, R = {:.1f}km".format(E/units.eV, domega / units.deg, R / units.km))
fig1.tight_layout()

ax.set_ylabel("Amplitude [V/m]")
ax.set_xlabel("Time [ns]")
ax.legend()
fig.tight_layout()
plt.show()

if save_plots:
    if(em):
        fig.savefig("model_comparison_em.png")
    else:
        fig.savefig("model_comparison_had.png")
plt.show()
