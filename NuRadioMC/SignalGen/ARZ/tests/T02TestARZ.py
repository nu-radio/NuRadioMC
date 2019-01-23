#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from NuRadioReco.utilities import units
from scipy import interpolate as intp
from scipy import integrate as int
from scipy import constants
from matplotlib import pyplot as plt
import os
import pickle
from time import time
from NuRadioMC.SignalGen.ARZ import ARZ
from radiotools import coordinatesystems as cstrafo
import logging
logging.basicConfig(level=logging.INFO)

shower_energy = 1.24e18 *units.eV
theta = 56 * units.deg
R = 50000 * units.km
N = 512
dt = 0.1 * units.ns
n_index = 1.78
shower_type = "EM"
tt = np.arange(0, dt*N, dt)

cARZ = ARZ.ARZ()

cARZ.set_seed(100)
trace = cARZ.get_time_trace(shower_energy, theta, N, dt, shower_type, n_index, R, shift_for_xmax=False)
fig, ax = plt.subplots(1, 1)
ax.plot(tt, trace[1]/units.V * units.m / units.milli, label='eTheta')
ax.plot(tt, trace[2]/units.V * units.m / units.milli, label='ePhi')
ax.plot(tt, trace[0]/units.V * units.m / units.milli, label='eR')
ax.set_xlabel("time [ns]")
ax.set_ylabel("amplitude [mV/m]")
ax.legend()
fig.tight_layout()

cARZ.set_seed(100)
trace = cARZ.get_time_trace(shower_energy, theta, N, dt, shower_type, n_index, R, shift_for_xmax=False)
fig, ax = plt.subplots(1, 1)
ax.plot(tt, trace[1]/units.V * units.m / units.milli, label='eTheta')
ax.plot(tt, trace[2]/units.V * units.m / units.milli, label='ePhi')
ax.plot(tt, trace[0]/units.V * units.m / units.milli, label='eR')
ax.set_xlabel("time [ns]")
ax.set_ylabel("amplitude [mV/m]")
ax.set_title("2")
ax.legend()
fig.tight_layout()

cARZ.set_seed(100)
trace2 = cARZ.get_time_trace(shower_energy, theta, N, dt, shower_type, n_index, R)

fig, ax = plt.subplots(1, 1)
ax.plot(tt, trace2[1]/units.V * units.m / units.milli, label='eTheta')
ax.plot(tt, trace2[2]/units.V * units.m / units.milli, label='ePhi')
ax.plot(tt, trace2[0]/units.V * units.m / units.milli, label='eR')
ax.set_title("shifted for xmax")
ax.set_xlabel("time [ns]")
ax.set_ylabel("amplitude [mV/m]")
ax.legend()
fig.tight_layout()

plt.show()
