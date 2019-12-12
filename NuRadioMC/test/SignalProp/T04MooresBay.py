#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
from numpy import testing
import time
from NuRadioMC.SignalProp import analyticraytracing as ray
from NuRadioMC.utilities import medium
from NuRadioReco.utilities import units
import logging
logging.basicConfig(level=logging.INFO)

ice = medium.mooresbay_simple()

"""
this unit test compares the numerical and analytic calculation of path length and travel time for the Moore's Bay site with additional reflections off the bottom.
the numerical integration should be better than the analytic formula. For both calculations, the python version is used.
"""


np.random.seed(0)  # set seed to have reproducible results
n_events = int(1e3)
rmin = -10. * units.m
rmax = 1. * units.km
zmin = 0. * units.m
zmax = -500. * units.m
rr = np.random.triangular(rmin, rmax, rmax, n_events)
phiphi = np.random.uniform(0, 2 * np.pi, n_events)
xx = rr * np.cos(phiphi)
yy = rr * np.sin(phiphi)
zz = np.random.uniform(zmin, zmax, n_events)

points = np.array([xx, yy, zz]).T
x_receiver = np.array([0., 0., -5.])

nsol = 6
results_D = np.zeros((n_events, nsol))
results_D_analytic = np.zeros((n_events, nsol))
results_T = np.zeros((n_events, nsol))
results_T_analytic = np.zeros((n_events, nsol))

d_numeric = 0
d_analytic = 0
t_numeric = 0
t_analytic = 0
for iX, x in enumerate(points):
    r = ray.ray_tracing(x, x_receiver, ice, log_level=logging.WARNING, n_reflections=1)
    r.find_solutions()
    if(r.has_solution()):
        for iS in range(r.get_number_of_solutions()):
            t_start = time.time()
            results_D[iX, iS] = r.get_path_length(iS, analytic=False)
            d_numeric += time.time() - t_start
            t_start = time.time()
            results_D_analytic[iX, iS] = r.get_path_length(iS, analytic=True)
            d_analytic += time.time() - t_start


            t_start = time.time()
            results_T[iX, iS] = r.get_travel_time(iS, analytic=False)
            t_numeric += time.time() - t_start
            t_start = time.time()
            results_T_analytic[iX, iS] = r.get_travel_time(iS, analytic=True)
            t_analytic += time.time() - t_start

print("asserting travel times")
mask = results_T != 0
print("average deviation = {:.4f}ns".format(np.mean(results_T[mask] - results_T_analytic[mask])/units.ns))
testing.assert_allclose(results_T, results_T_analytic, atol=1e-2 * units.ns, rtol=1e-3)
print("asserting distances")
testing.assert_allclose(results_D, results_D_analytic, atol=4e-2 * units.m, rtol=1e-3)
print('T04MooresBay passed without issues')
