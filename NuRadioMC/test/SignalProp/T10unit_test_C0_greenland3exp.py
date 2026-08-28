#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
from numpy import testing
import time
from NuRadioMC.SignalProp import analyticraytracing as ray
from NuRadioMC.utilities import medium
from NuRadioReco.utilities import io_utilities, units
import logging
import json

logger = logging.getLogger("NuRadioMC.T10unit_test_C0_greenland3exp")
logger.setLevel(logging.INFO)

ice = medium.greenland_3exp_layered()

"""
this unit test compares the numerical and analytic calculation of path length and travel time for the Summit Station site using the 3 layer exponential model.
the numerical integration should be better than the analytic formula. For both calculations, the python version is used.
"""


np.random.seed(10)  # set seed to have reproducible results
n_events = int(1e3)
rmin = 50. * units.m
rmax = 3. * units.km
zmin = 0. * units.m
zmax = -3. * units.km
rr = np.random.triangular(rmin, rmax, rmax, n_events)
phiphi = np.random.uniform(0, 2 * np.pi, n_events)
xx = rr * np.cos(phiphi)
yy = rr * np.sin(phiphi)
zz = np.random.uniform(zmin, zmax, n_events)

points = np.array([xx, yy, zz]).T
x_receiver = np.array([0., 0., -5.])

results_C0s_cpp = np.zeros((n_events, 10))
n_freqs = 256//2 + 1
# n_freqs = 5
results_A_cpp = np.zeros((n_events, 2, n_freqs))
t_start = time.time()
ff = np.linspace(0, 500*units.MHz, n_freqs)
# tt = 0
r = ray.ray_tracing(ice)
for iX, x in enumerate(points):
    r.set_start_and_end_point(x, x_receiver)
    r.find_solutions()
    if(r.has_solution()):
        for iS in range(r.get_number_of_solutions()):
            results_C0s_cpp[iX, iS] = r.get_results()[iS]['C0']

# with open("reference_C0_MooresBay.pkl", "wb") as fout:
#     pickle.dump(results_C0s_cpp, fout)
#results_C0s_cpp_ref = io_utilities.read_pickle("reference_C0_greenland3exp.pkl", encoding='latin1')

with open("reference_C0_greenland3exp.json", "r") as fin:
    results_C0s_cpp_ref = np.array(json.load(fin))  # Convert list back to NumPy array
    
testing.assert_allclose(results_C0s_cpp, results_C0s_cpp_ref, rtol=1.e-6)

print('T10unit_test_C0_greenland3exp passed without issues')
