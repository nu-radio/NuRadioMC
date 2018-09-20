import matplotlib.pyplot as plt
import numpy as np
import time
from NuRadioMC.SignalProp import analyticraytraycing as ray
from NuRadioMC.utilities import units, medium
import logging
import time
from radiotools import helper as hp
from radiotools import plthelpers as php
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test_raytracing')

ice = medium.southpole_simple()

np.random.seed(0)  # set seed to have reproducible results
n_events = int(1e4)
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

results_D = np.zeros((n_events, 2))
results_D_analytic = np.zeros((n_events, 2))
results_T = np.zeros((n_events, 2))
results_T_analytic = np.zeros((n_events, 2))


d_numeric = 0
d_analytic = 0
t_numeric = 0
t_analytic = 0
for iX, x in enumerate(points):
    r = ray.ray_tracing(x, x_receiver, ice)
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
#             print("analytic {:.2f}".format(results_D_analytic[iX, iS]))
#             if(np.abs(results_D[iX, iS] - results_D_analytic[iX, iS]) > 0.1):
#                 print(np.abs(results_D[iX, iS] - results_D_analytic[iX, iS]))
#                 a = 1/0
t_cpp = time.time() - t_start
print("analytic {:.1f} seconds = {:.2f}ms/event".format(d_analytic, 1000. * d_analytic / n_events))
print("numeric {:.1f} seconds = {:.2f}ms/event".format(d_numeric, 1000. * d_numeric / n_events))
print("analytic {:.1f} seconds = {:.2f}ms/event".format(t_analytic, 1000. * t_analytic / n_events))
print("numeric {:.1f} seconds = {:.2f}ms/event".format(t_numeric, 1000. * t_numeric / n_events))
print("distance ", np.allclose(results_D, results_D_analytic, atol=1e-1, rtol=1e-3))
print("travel time ", np.allclose(results_T, results_T_analytic, atol=1e-1, rtol=1e-3))