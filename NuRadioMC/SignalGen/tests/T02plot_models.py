import NuRadioMC.SignalGen.parametrizations as param
from NuRadioReco.utilities import units
import numpy as np
import matplotlib.pyplot as plt
from radiotools import plthelpers as php
from NuRadioReco.utilities import units, fft

model = 'Alvarez2009'

dt = .1 * units.ns
n_samples = 2048
ff = np.fft.rfftfreq(n_samples, dt)
tt = np.arange(0, n_samples * dt, dt)
df = ff[1]  # width of a frequency bin


# plot dependence on cherenkov cone
E = 1e18 * units.eV
n_index = 1.78
R = 1 * units.km
domegas = np.array([0, 1, 2, 3, 4, 5, 7.5, 10]) * units.deg
emstr = ['EM', 'HAD']
em = np.array([True, False])

# lpm = np.array([False, True, False])
for i in range(len(emstr)):  # loop over EM/EM+LPM/HAD
    fig, ax = plt.subplots(1, 1)
    for iOmega, domega in enumerate(domegas):
        theta = np.arccos(1. / n_index) + domega
        # spec1 = param.get_frequency_spectrum(E, theta, n_samples, dt, em[i], n_index, R, model=model)
        time_trace = param.get_time_trace(energy=E, theta=theta, N=n_samples, dt=dt, 
            shower_type=emstr[i], n_index=n_index, R=R, model=model)
        spectrum = fft.time2freq(time_trace, 1/dt)
        ax.plot(ff / units.MHz, np.abs(spectrum) / units.V * units.m, php.get_color_linestyle(iOmega), label='$\Delta \Omega$ = {:.1f}deg'.format(domega / units.deg))
    ax.semilogx(True)
    ax.semilogy(True)
    ax.set_xlim(10, 2e3)
    ax.set_ylim(1e-7, 1e-1)
    ax.set_xlabel("frequency [MHz]")
    ax.set_ylabel("amplitude [V/m] per {:.1f}MHz".format(df / units.MHz))
    ax.legend(fontsize='small')
    ax.set_title("{} E = {:.1e}eV, R = {:.1f}km, {}".format(model, E / units.eV, R / units.km, emstr[i]))
    fig.tight_layout()
    fig.savefig("plots/{}_E{:.1e}eV_R{:.0f}m_EM{}.png".format(model, E / units.eV, R / units.m, em[i]))
    plt.show()
    plt.close('all')

# plot distance dependence for domegas
E = 1e16 * units.eV
n_index = 1.78
domegas = np.array([0, 2.5, 5]) * units.deg
em = np.array([True, True, False])
lpm = np.array([False, True, False])
Rs = np.array([100, 200, 500, 1000, 2000]) * units.m
for i in range(len(emstr)):  # loop over EM/EM+LPM/HAD
    for iOmega, domega in enumerate(domegas):
        theta = np.arccos(1. / n_index) + domega
        fig, ax = plt.subplots(1, 1)
        for iR, R in enumerate(Rs):
            # spec1 = get_frequency_spectrum(E, theta, ff, em[i], n_index, R, lpm[i]) * R
            time_trace = param.get_time_trace(energy=E, theta=theta, N=n_samples, dt=dt, 
            shower_type=emstr[i], n_index=n_index, R=R, model=model)
            spec1 = fft.time2freq(time_trace, 1/dt)
            # print(spec1.shape, ff.shape)
            ax.plot(ff / units.MHz, np.abs(spec1) / units.V * units.m, php.get_color_linestyle(iR), label='R = {:.0f}m'.format(R / units.m))
        ax.semilogx(True)
        ax.semilogy(True)
        ax.set_xlim(10, 2e3)
        ax.set_ylim(1e-6, 1e-0)
        ax.set_xlabel("frequency [MHz]")
        ax.set_ylabel("amplitude [V/m] * R [m] per {:.1f}MHz".format(df / units.MHz))
        ax.legend(fontsize='small')
        ax.set_title("{} E = {:.1e}eV, $\Delta \Omega$ = {:.1f}deg, {}".format(model, E / units.eV, domega / units.deg, emstr[i]))
        fig.tight_layout()
        fig.savefig("plots/{}_E{:.1e}eV_dOmega{:.1f}_EM{}_LPM{}.png".format(model, E / units.eV, domega / units.deg, em[i], lpm[i]))
        plt.show()
        plt.close('all')
