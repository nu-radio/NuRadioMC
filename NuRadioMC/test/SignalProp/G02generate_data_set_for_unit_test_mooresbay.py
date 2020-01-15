import numpy as np
import pickle
import time
from NuRadioMC.SignalProp import analyticraytracing as ray
from NuRadioMC.utilities import medium
from NuRadioReco.utilities import units
import logging
from numpy import testing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test_raytracing')

ice = medium.mooresbay_simple()

np.random.seed(10)  # set seed to have reproducible results
n_events = int(1e3)
rmin = 50. * units.m
rmax = 3. * units.km
zmin = 0. * units.m
zmax = -0.5 * units.km
rr = np.random.triangular(rmin, rmax, rmax, n_events)
phiphi = np.random.uniform(0, 2 * np.pi, n_events)
xx = rr * np.cos(phiphi)
yy = rr * np.sin(phiphi)
zz = np.random.uniform(zmin, zmax, n_events)

points = np.array([xx, yy, zz]).T
x_receiver = np.array([0., 0., -5.])

results_C0s_cpp = np.zeros((n_events, 10))
n_freqs = 256 // 2 + 1
# n_freqs = 5
results_A_cpp = np.zeros((n_events, 2, n_freqs))
t_start = time.time()
ff = np.linspace(0, 500 * units.MHz, n_freqs)
# tt = 0
for iX, x in enumerate(points):
#     t_start2 = time.time()
    r = ray.ray_tracing(x, x_receiver, ice, n_reflections=2)
#     tt += (time.time() - t_start2)
    r.find_solutions()
    if(r.has_solution()):
        for iS in range(r.get_number_of_solutions()):
            results_C0s_cpp[iX, iS] = r.get_results()[iS]['C0']
with open("reference_C0_MooresBay.pkl", "bw") as fout:
    pickle.dump(results_C0s_cpp, fout, protocol=4)
