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

shower_energy = 1e18 *units.eV
theta = 55 * units.deg
R = 1 * units.km
N = 512
dt = 0.1 * units.ns
n_index = 1.78
shower_type = "HAD"
tt = np.arange(0, dt*N, dt)

trace = ARZ.get_time_trace(shower_energy, theta, N, dt, shower_type, n_index, R)
fig, ax = plt.subplots(1, 1)
ax.plot(tt, trace[0]/units.V * units.m / units.milli, label='eR')
ax.plot(tt, trace[1]/units.V * units.m / units.milli, label='eTheta')
ax.plot(tt, trace[2]/units.V * units.m / units.milli, label='ePhi')
ax.set_xlabel("time [ns]")
ax.set_ylabel("amplitude [mV/m]")
ax.legend()
fig.tight_layout()

plt.show()
