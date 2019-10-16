import NuRadioMC.SignalGen.askaryan as ask
from NuRadioReco.utilities import units
import numpy as np
import matplotlib.pyplot as plt
from radiotools import plthelpers as php
import logging
logging.basicConfig(level=logging.INFO)

interp_factor = 20
# HAD for different viewing angles
E = 1 * units.EeV
shower_type = "HAD"
n_index = 1.78
theta_C = np.arccos(1. / n_index)
N = 512*2
dt = 0.1 * units.ns
R = 1 * units.km
tt = np.arange(0, dt * N, dt)

thetas = theta_C + np.array([0, 1, 2, 3, 4, 5]) * units.deg
fig, ax = plt.subplots(2, 3, sharex=True)
ax = ax.flatten()
for iTheta, theta in enumerate(thetas):
    trace = ask.get_time_trace(E, theta, N, dt, shower_type, n_index, R, "ARZ2019", interp_factor=interp_factor)
    ax[iTheta].plot(tt, trace[1], '-C0'.format(iTheta), label="$\Delta \Theta$ = {:.1f}".format((theta-theta_C)/units.deg))
    trace = ask.get_time_trace(E, theta, N, dt, shower_type, n_index, R, "Alvarez2000")
    trace = np.roll(trace, int(-1 * units.ns/dt))
    ax[iTheta].plot(tt, trace, '--C1'.format(iTheta), label="$\Delta \Theta$ = {:.1f}".format((theta-theta_C)/units.deg))
    ax[iTheta].set_title("$\Delta \Theta$ = {:.1f}".format((theta-theta_C)/units.deg))
    ax[iTheta].set_xlim(45, 60)
# ax.legend()
fig.tight_layout()
fig.suptitle("HAD, Esh = {:.1f}EeV".format(E/units.EeV))
fig.subplots_adjust(top=0.9)
fig.savefig("plots/04_1EeV_HAD.png")

shower_type = "EM"
fig, ax = plt.subplots(2, 3, sharex=True)
ax = ax.flatten()
for iTheta, theta in enumerate(thetas):
    trace = ask.get_time_trace(E, theta, N, dt, shower_type, n_index, R, "ARZ2019", interp_factor=interp_factor)
    ax[iTheta].plot(tt, trace[1], '-C0'.format(iTheta), label="$\Delta \Theta$ = {:.1f}".format((theta-theta_C)/units.deg))
    trace = ask.get_time_trace(E, theta, N, dt, shower_type, n_index, R, "Alvarez2000")
    trace = np.roll(trace, int(-1 * units.ns/dt))
    ax[iTheta].plot(tt, trace, '--C1'.format(iTheta), label="$\Delta \Theta$ = {:.1f}".format((theta-theta_C)/units.deg))
    ax[iTheta].set_title("$\Delta \Theta$ = {:.1f}".format((theta-theta_C)/units.deg))
    ax[iTheta].set_xlim(45, 70)
# ax.legend()
fig.tight_layout()
fig.suptitle("EM, Esh = {:.1f}EeV".format(E/units.EeV))
fig.subplots_adjust(top=0.9)
fig.savefig("plots/04_1EeV_EM.png")


shower_type = "EM"
E = 0.01 * units.EeV
fig, ax = plt.subplots(2, 3, sharex=True)
ax = ax.flatten()
for iTheta, theta in enumerate(thetas):
    trace = ask.get_time_trace(E, theta, N, dt, shower_type, n_index, R, "ARZ2019", interp_factor=interp_factor)
    ax[iTheta].plot(tt, trace[1], '-C0'.format(iTheta), label="$\Delta \Theta$ = {:.1f}".format((theta-theta_C)/units.deg))
    trace = ask.get_time_trace(E, theta, N, dt, shower_type, n_index, R, "Alvarez2000")
    trace = np.roll(trace, int(-1 * units.ns/dt))
    ax[iTheta].plot(tt, trace, '--C1'.format(iTheta), label="$\Delta \Theta$ = {:.1f}".format((theta-theta_C)/units.deg))
    ax[iTheta].set_title("$\Delta \Theta$ = {:.1f}".format((theta-theta_C)/units.deg))
    ax[iTheta].set_xlim(45, 70)
# ax.legend()
fig.tight_layout()
fig.suptitle("EM, Esh = {:.1f}PeV".format(E/units.PeV))
fig.subplots_adjust(top=0.9)
fig.savefig("plots/04_10PeV_EM.png")
plt.show()
