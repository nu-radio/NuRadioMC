import matplotlib.pyplot as plt
import numpy as np
import time
from numpy import testing
from NuRadioMC.SignalProp import analyticraytracing as ray
from NuRadioMC.utilities import medium
from NuRadioReco.utilities import io_utilities, units
from NuRadioReco.framework import base_trace
import logging
from scipy.spatial.tests.test_qhull import points
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test_raytracing')

"""
this unit test checks the output of the birefringence calculations
"""

ice = medium.birefringence_medium(bir_model='southpole_A', exp_model=medium.southpole_2015())

np.random.seed(42)  # set seed to have reproducible results
n_events = int(10)
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

stamps = 200000

try:
    test_pulse = np.load('test_pulse.npy')
    
except:
    T_start = -50
    T_final = 50
    th_max = 3

    T = np.linspace(T_start, T_final, stamps)
    dt = T[1] - T[0]

    t_theta = base_trace.BaseTrace()
    t_phi = base_trace.BaseTrace()

    delta_pulse_theta = np.zeros(stamps)
    delta_pulse_theta[int(stamps/2)] = 1

    t_theta.set_trace(delta_pulse_theta, sampling_rate=1 / dt)

    ff = t_theta.get_frequencies()
    spectrum = t_theta.get_frequency_spectrum()
    spectrum[ff > 300 * units.MHz] = 0
    spectrum[ff < 80 * units.MHz] = 0
    t_theta.set_frequency_spectrum(spectrum, sampling_rate=1 / dt)

    Th = th_max * t_theta.get_trace() / max(t_theta.get_trace())
    Ph = th_max * t_theta.get_trace() / max(t_theta.get_trace())

    test_pulse= np.array([T, Th, Ph])
    np.save('test_pulse.npy', test_pulse)

results_theta = np.array([test_pulse[1]])
results_phi = np.array([test_pulse[2]])

for iX, x in enumerate(points):

    r = ray.ray_tracing(ice)
    r.set_start_and_end_point(x, x_receiver)
    r.find_solutions()
    if(r.has_solution()):
        for iS in range(r.get_number_of_solutions()):

            final_pulse = r.get_pulse_trace_fast(x, x_receiver, 'test_pulse.npy', path_type=iS)

            results_theta = np.vstack((results_theta, final_pulse[2]))
            results_phi = np.vstack((results_phi, final_pulse[2]))

compare_array = np.vstack((results_theta, results_phi))
reference_array = np.load('reference_birefringence.npy')

testing.assert_allclose(compare_array, reference_array, atol=1e-2  * units.V / units.m, rtol=1e-10)
print('T07test_birefringence passed without issues')

