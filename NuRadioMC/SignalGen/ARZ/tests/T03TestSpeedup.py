from __future__ import print_function, division
import time
import numpy as np
from NuRadioReco.utilities import units
import os
from scipy import interpolate as intp
import glob
import pickle
import sys
from radiotools import plthelpers as php
from matplotlib import pyplot as plt
from NuRadioMC.SignalGen.ARZ import ARZ
import logging
import matplotlib.gridspec as gridspec


logger = logging.getLogger("test")
logging.basicConfig(level=logging.WARNING)
rho = 0.924 * units.g / units.cm ** 3  # density g cm^-3

if __name__ == "__main__":
    T=100 * units.ns
    dt = 0.1*units.ns
    N = np.int(np.round(T/dt))
    tt = np.arange(0, dt*N, dt)
    n_index = 1.78
    theta = np.arccos(1./n_index)
    R = 1 * units.km
    a = ARZ.ARZ(library="/Users/cglaser/work/ARIANNA/data/ARZ/v1.1/library_v1.1.pkl", interp_factor=1, interp_factor2=1)

    # HAD 1e19
    shower_energy = 10*units.EeV
    shower_type = "HAD"
    n = 10
    thetas = np.linspace(theta - 2 * units.deg, theta + 10 *units.deg, n)

    a.set_interpolation_factor(0.1)
    a.set_interpolation_factor2(1000)
    t0 = time.time()
    for i in range(n):
        trace = a.get_time_trace(shower_energy, thetas[i], N, dt, shower_type, n_index, R, iN=i)
    print("0.1x interpolation, 1000x interpolation2, 10 showers = {:.2f}s".format(time.time() - t0))

    a.set_interpolation_factor(0.1)
    a.set_interpolation_factor2(100)
    t0 = time.time()
    for i in range(n):
        trace = a.get_time_trace(shower_energy, thetas[i], N, dt, shower_type, n_index, R, iN=i)
    print("0.1x interpolation, 100x interpolation2, 10 showers = {:.2f}s".format(time.time() - t0))

    a.set_interpolation_factor(1)
    a.set_interpolation_factor2(1)
    t0 = time.time()
    for i in range(n):
        trace = a.get_time_trace(shower_energy, thetas[i], N, dt, shower_type, n_index, R, iN=i)
    print("no interpolation, 10 showers = {:.2f}s".format(time.time() - t0))

    a.set_interpolation_factor(10)
    a.set_interpolation_factor2(1)
    t0 = time.time()
    for i in range(n):
        trace = a.get_time_trace(shower_energy, thetas[i], N, dt, shower_type, n_index, R, iN=i)
    print("10x interpolation, 10 showers = {:.2f}s".format(time.time() - t0))

    a.set_interpolation_factor(100)
    a.set_interpolation_factor2(1)
    t0 = time.time()
    for i in range(n):
        trace = a.get_time_trace(shower_energy, thetas[i], N, dt, shower_type, n_index, R, iN=i)
    print("100x interpolation, 10 showers = {:.2f}s".format(time.time() - t0))

    a.set_interpolation_factor(1)
    a.set_interpolation_factor2(10)
    t0 = time.time()
    for i in range(n):
        trace = a.get_time_trace(shower_energy, thetas[i], N, dt, shower_type, n_index, R, iN=i)
    print("10x interpolation2, 10 showers = {:.2f}s".format(time.time() - t0))

    a.set_interpolation_factor(1)
    a.set_interpolation_factor2(100)
    t0 = time.time()
    for i in range(n):
        trace = a.get_time_trace(shower_energy, thetas[i], N, dt, shower_type, n_index, R, iN=i)
    print("100x interpolation2, 10 showers = {:.2f}s".format(time.time() - t0))
