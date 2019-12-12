import NuRadioMC.SignalGen.RalstonBuniy.askaryan_module as AskaryanModule
import NuRadioMC.SignalGen.parametrizations as param
from NuRadioReco.utilities import units
import numpy as np
from radiotools import plthelpers as php
import matplotlib.pyplot as plt

E = 1e17 * units.eV
n_index = 1.78
domega = 2.5 * units.deg
theta = np.arccos(1. / n_index) + domega

dt = .1 * units.ns
n_samples = 2048
ff = np.fft.rfftfreq(n_samples, dt)
df = ff[1]  # width of a frequency bin
tt = np.arange(0, n_samples * dt, dt)
R = 5 * units.km
em = True
a = 1.5 * units.m

fig, ax = plt.subplots(1, 1)
domegas = np.array([0, 2.5, 5, 7.5, 10]) * units.deg
for iOmega, domega in enumerate(domegas):
    theta = np.arccos(1. / n_index) + domega
    spec1 = AskaryanModule.get_frequency_spectrum(E, theta, ff, em, n_index, R, True, a)
    spec1_anoLPM = AskaryanModule.get_frequency_spectrum(E, theta, ff, em, n_index, R, False)
    spec1_aLPM = AskaryanModule.get_frequency_spectrum(E, theta, ff, em, n_index, R, True)
    spec4 = param.get_frequency_spectrum(E, theta, ff, em, n_index, R, model='ZHS1992')
    ax.plot(ff / units.MHz, np.abs(spec1[1]) / units.V * units.m, php.get_color(iOmega) + "-", label=r'JCH/AC a=1.5m $\Delta \Omega$ = {:.1f}deg'.format(domega / units.deg))
#     ax.plot(ff / units.MHz, np.abs(spec1_noLPM[1]) / units.V * units.m, php.get_color_linestyle(1), label='Askaryan module eTheta (without LPM)')
    ax.plot(ff / units.MHz, np.abs(spec4) / units.V * units.m, php.get_color(iOmega) + "--", label=r'ZHS $\Delta \Omega$ = {:.1f}deg'.format(domega / units.deg))
ax.semilogx(True)
ax.semilogy(True)
ax.set_xlim(10, 2e3)
ax.set_ylim(1e-9, 1e-3)
ax.set_xlabel("frequency [MHz]")
ax.set_ylabel("amplitude [V/m] per {:.1f}MHz".format(df / units.MHz))
ax.legend(fontsize='small')
ax.set_title("E = {:.1e}eV, R = {:.1f}km, EM={}".format(E / units.eV, R / units.km, em))
fig.tight_layout()
fig.savefig("plots/AskaryanZHS.png")

domegas = np.array([0, 1, 2.5, 5, 7.5, 10]) * units.deg
for iOmega, domega in enumerate(domegas):
    fig, ax = plt.subplots(1, 1)
    theta = np.arccos(1. / n_index) + domega
    spec1 = AskaryanModule.get_frequency_spectrum(E, theta, ff, em, n_index, R, True, a)
    spec1_anoLPM = AskaryanModule.get_frequency_spectrum(E, theta, ff, em, n_index, R, False)
    spec1_aLPM = AskaryanModule.get_frequency_spectrum(E, theta, ff, em, n_index, R, True)
    spec4 = param.get_frequency_spectrum(E, theta, ff, em, n_index, R, model='ZHS1992')
    ax.plot(ff / units.MHz, np.abs(spec1[1]) / units.V * units.m, php.get_color(0) + "-", label=r'JCH/AC a=1.5m')
#     ax.plot(ff / units.MHz, np.abs(spec1_noLPM[1]) / units.V * units.m, php.get_color_linestyle(1), label='Askaryan module eTheta (without LPM)')
    ax.plot(ff / units.MHz, np.abs(spec4) / units.V * units.m, php.get_color(1) + "--", label=r'ZHS')
    ax.plot(ff / units.MHz, np.abs(spec1_anoLPM[1]) / units.V * units.m, php.get_color(2) + ":", label=r'JCH/AC without LPM')
    ax.plot(ff / units.MHz, np.abs(spec1_aLPM[1]) / units.V * units.m, php.get_color(3) + "-.", label='JCH/AC with LPM')
    ax.semilogx(True)
    ax.semilogy(True)
    ax.set_xlim(10, 2e3)
    ax.set_ylim(1e-9, 1e-3)
    ax.set_xlabel("frequency [MHz]")
    ax.set_ylabel("amplitude [V/m] per {:.1f}MHz".format(df / units.MHz))
    ax.legend(fontsize='small')
    ax.set_title("E = {:.1e}eV, $\Delta\Omega$ = {:.1f}deg, R = {:.1f}km, EM={}".format(E / units.eV, domega / units.deg, R / units.km, em))
    fig.tight_layout()
    fig.savefig("plots/AskaryanZHS_{:.1f}deg.png".format(domega / units.deg))
    plt.close('all')
