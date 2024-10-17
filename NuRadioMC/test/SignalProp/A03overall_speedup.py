import matplotlib.pyplot as plt
import numpy as np
import time
from NuRadioMC.SignalProp import analyticraytraycing as ray
from NuRadioReco.utilities import units
from NuRadioMC.utilities import medium
import logging
import time
from radiotools import helper as hp
from radiotools import plthelpers as php
logger = logging.getLogger('NuRadioMC.test_raytracing')
logger.setLevel(logging.INFO)

ice = medium.southpole_simple()

np.random.seed(0)  # set seed to have reproducible results
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

results_C0s_cpp = np.zeros((n_events, 2))
n_freqs = 256/2 + 1
results_A_cpp = np.zeros((n_events, 2, n_freqs))
t_start = time.time()
ff = np.linspace(0, 500*units.MHz, n_freqs)
# tt = 0
for iX, x in enumerate(points):
#     t_start2 = time.time()
    r = ray.ray_tracing(x, x_receiver, ice)
#     tt += (time.time() - t_start2)
    r.find_solutions()
    if(r.has_solution()):
        for iS in range(r.get_number_of_solutions()):
            results_C0s_cpp[iX, iS] = r.get_results()[iS]['C0']
            results_A_cpp[iX, iS] = r.get_attenuation(iS, ff)
            r.get_travel_time(iS)
            r.get_path_length(iS)
t_cpp = time.time() - t_start
print("CPP time = {:.1f} seconds = {:.2f}ms/event".format(t_cpp, 1000. * t_cpp / n_events))
# print("CPP time = {:.1f} seconds = {:.2f}ms/event".format(tt, 1000. * tt / n_events))


results_C0s_python = np.zeros((n_events, 2))
results_A_python = np.zeros((n_events, 2, n_freqs))
ray.cpp_available = False
t_start = time.time()
for iX, x in enumerate(points):
    r = ray.ray_tracing(x, x_receiver, ice)
    r.find_solutions()
    if(r.has_solution()):
        for iS in range(r.get_number_of_solutions()):
            results_C0s_python[iX, iS] = r.get_results()[iS]['C0']
            results_A_python[iX, iS] = r.get_attenuation(iS, ff)
            r.get_travel_time(iS, analytic=False)
            r.get_path_length(iS, analytic=False)
t_python = time.time() - t_start
print("Python time = {:.1f} seconds = {:.2f}ms/event".format(t_python, 1000. * t_python / n_events))
print("overall speedup = {:.2f}".format(t_python/ t_cpp))

print("consistent results for C0: {}".format(np.allclose(results_C0s_cpp, results_C0s_python)))
print("consistent results for attenuation length: {}".format(np.allclose(results_A_cpp, results_A_python)))
print("consistent results for attenuation length rtol=1e-2, atol=1e-3: {}".format(np.allclose(results_A_cpp, results_A_python, rtol=1e-2, atol=1e-3)))
