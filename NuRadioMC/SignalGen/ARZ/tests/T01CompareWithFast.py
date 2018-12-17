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
from NuRadioMC.SignalGen.ARZ.ARZ import *

energy = 1.e6 * units.TeV
theta = 55 * units.deg
R = 1 * units.km
N = 512
dt = 0.1 * units.ns
n_index = 1.78
y = 0
ccnc = 'cc'
flavor = 12  # e = 12, mu = 14, tau = 16

cdir = os.path.dirname(__file__)
bins, depth_e, N_e = np.loadtxt(os.path.join(cdir, "../shower_library/nue_1EeV_CC_1_s0001.t1005"), unpack=True)
bins, depth_p, N_p = np.loadtxt(os.path.join(cdir, "../shower_library/nue_1EeV_CC_1_s0001.t1006"), unpack=True)
depth_e *= units.g / units.cm**2
depth_p *= units.g / units.cm**2
depth_e -= 1000 * units.g/units.cm**2  # all simulations have an artificial offset of 1000 g/cm^2
depth_p -= 1000 * units.g/units.cm**2
# sanity check if files electron and positron profiles are compatible
if (not np.all(depth_e == depth_p)):
    raise ImportError("electron and positron profile have different depths")

t0 = time()
vp2 = get_vector_potential_fast(energy, theta, N, dt, "EM", n_index, R, profile_depth=depth_e, profile_ce=(N_e-N_p))
print("fast calculation took {:.1f} ms".format((time() -t0)*1e3))
t0 = time()
vp = get_vector_potential(energy, theta, N, dt, y, ccnc, flavor, n_index, R, profile_depth=depth_e, profile_ce=(N_e-N_p))
print("slow calculation took {:.4f}s".format(time() -t0))

# generate time array
tt = np.arange(0, (N + 1) * dt, dt)
tt = tt + 0.5 * dt - tt.mean()

fig, ax = plt.subplots(1, 1)
ax.plot(tt, vp[:, 0] / units.V / units.s)
ax.plot(tt, vp[:, 1] / units.V / units.s)
ax.plot(tt, vp[:, 2] / units.V / units.s)
ax.plot(tt, vp2[:, 0] / units.V / units.s, "C0--")
ax.plot(tt, vp2[:, 1] / units.V / units.s, "C1--")
ax.plot(tt, vp2[:, 2] / units.V / units.s, "C2--")
ax.set_xlim(-2, 2)

ax.set_xlabel("time [ns]")
ax.set_ylabel("vector potential")


fig, ax = plt.subplots(1, 1)
ax.plot(tt, vp[:, 0]/vp2[:, 0])
ax.plot(tt, vp[:, 2]/ vp2[:, 0])
ax.set_xlim(-2, 2)
ax.set_xlabel("time [ns]")
ax.set_ylabel("python/fortran implementation")
ax.set_ylim(0.8, 1.2)
plt.show()