import numpy as np
import time
from NuRadioMC.SignalProp import analyticraytracing as ray
from NuRadioMC.utilities import medium
from NuRadioReco.utilities import units
from NuRadioReco.utilities import io_utilities
import logging
import pickle
from numpy import testing
logger = logging.getLogger('NuRadioMC.test_raytracing')
logger.setLevel(logging.INFO)

ice = medium.southpole_simple()

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

results_C0s_cpp = np.zeros((n_events, 2))
n_freqs = 256 // 2 + 1
# n_freqs = 5
results_A_cpp = np.zeros((n_events, 2, n_freqs))
t_start = time.time()
ff = np.linspace(0, 500 * units.MHz, n_freqs)
# tt = 0
r = ray.ray_tracing(ice)
for iX, x in enumerate(points):
    r.set_start_and_end_point(x, x_receiver)
    r.find_solutions()
    if(r.has_solution()):
        for iS in range(r.get_number_of_solutions()):
            results_C0s_cpp[iX, iS] = r.get_results()[iS]['C0']

# with open("reference_C0.pkl", "wb") as fout:
#     pickle.dump(results_C0s_cpp, fout)
results_C0s_cpp_ref = io_utilities.read_pickle("reference_C0.pkl", encoding='latin1')
testing.assert_allclose(results_C0s_cpp, results_C0s_cpp_ref)

print('T05unit_test_c0_SP passed without issues')
