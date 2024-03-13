import numpy as np
from numpy import testing
from NuRadioMC.SignalProp import analyticraytracing as ray
from NuRadioMC.utilities import medium
from NuRadioReco.utilities import units
from NuRadioReco.framework import base_trace
import NuRadioReco.framework.electric_field
import logging
from scipy.spatial.tests.test_qhull import points
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test_raytracing')


import matplotlib.pyplot as plt

"""
this unit test checks the output of the birefringence calculations
"""

ref_index_model = 'southpole_2015'
ice = medium.get_ice_model(ref_index_model)


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

points = np.array([xx, yy, zz]).T * units.m
x_receiver = np.array([0., 0., -150.]) * units.m

size = 500

delta = np.zeros(size)
delta[int(size/2)] = 1

sr = 2*10**(9) * units.hertz

delta_pulse_ana = NuRadioReco.framework.electric_field.ElectricField([1], position=None,
                 shower_id=None, ray_tracing_id=None)

delta_pulse_ana.set_trace(delta, sr)

filter = delta_pulse_ana.get_filtered_trace([50* units.MHz, 300* units.MHz], filter_type = 'rectangular')
filter = 1/np.sqrt(2) * filter/max(filter)

zeros = np.zeros(size)
delta_efield = np.vstack((zeros, filter, filter))

delta_pulse_ana.set_trace(delta_efield, sr)

config = {'propagation': {}}
config['propagation']['attenuate_ice'] = False
config['propagation']['focusing_limit'] = 2
config['propagation']['focusing'] = False
config['propagation']['birefringence'] = True
config['propagation']['birefringence_model'] = 'southpole_A'
config['propagation']['birefringence_propagation'] = 'analytical'

results_theta = np.array(delta_pulse_ana.get_trace()[1])
results_phi = np.array(delta_pulse_ana.get_trace()[2])

for iX, x in enumerate(points):

    r = ray.ray_tracing(ice)
    r.set_start_and_end_point(x, x_receiver)
    r.find_solutions()
    r.set_config(config)
    if r.has_solution():
        for iS in range(r.get_number_of_solutions()):

            delta_efield = np.vstack((zeros, filter, filter))
            delta_pulse_ana.set_trace(delta_efield, sr)
            
            final_pulse = r.apply_propagation_effects(delta_pulse_ana, iS)

            results_theta = np.vstack((results_theta, final_pulse.get_trace()[1]))
            results_phi = np.vstack((results_phi, final_pulse.get_trace()[2]))

compare_array = np.vstack((results_theta, results_phi))
reference_array = np.load('reference_BF.npy')

# The tolerance was chosen to be 0.0002V/m. The amplitudes of the pulses are above 0.1V/m.
# This tolerance is necessary as there are small numerical instabilities in the polarization calculation of the birefringence functions. 
# Over the propagation these differences can add up but seem to remain below 1% of the original pulse amplitude.

testing.assert_allclose(compare_array, reference_array, atol=2e-4  * units.V / units.m, rtol=1e-7)
print('T07test_birefringence passed without issues')


