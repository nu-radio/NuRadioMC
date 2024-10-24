import numpy as np
from numpy import testing
from NuRadioMC.SignalProp import analyticraytracing as ray
from NuRadioMC.utilities import medium
from NuRadioReco.utilities import units
from NuRadioReco.framework import base_trace
import NuRadioReco.framework.electric_field
import logging
from scipy.spatial.tests.test_qhull import points
logger = logging.getLogger('NuRadioMC.test_raytracing')
logger.setLevel(logging.INFO)

import NuRadioReco.modules.io.eventReader as reader


import matplotlib.pyplot as plt

"""
this unit test checks a full emitter simulation with birefringence
"""

#creating the reference traces
"""
event_reader = reader.eventReader()
nur = 'emitter_sim_test/test_output.nur'
event_reader.begin([nur])
list_traces = []

for event in event_reader.run():

    station = event.get_station(51)
    list_trace = []

    for channel in station.iter_channels():

        trace = channel.get_trace()
        list_trace.append(trace)

    event_traces = np.vstack(list_trace)
    list_traces.append(event_traces)

traces = np.dstack(list_traces)
traces = np.transpose(traces, (2, 0, 1))

np.save('reference_emitter.npy', traces)
"""

event_reader = reader.eventReader()
nur = 'emitter_sim_test/test_output.nur'
event_reader.begin([nur])
list_traces = []

for event in event_reader.run():

    station = event.get_station(51)
    list_trace = []

    for channel in station.iter_channels():

        trace = channel.get_trace()
        list_trace.append(trace)

    event_traces = np.vstack(list_trace)
    list_traces.append(event_traces)

traces = np.dstack(list_traces)
traces = np.transpose(traces, (2, 0, 1))

reference_array = np.load('reference_emitter.npy')

# The tolerance was chosen to be 0.002V. The amplitudes of the pulses are above 1V.
# This tolerance is necessary as there are small numerical instabilities in the polarization calculation of the birefringence functions. 
# Over the propagation these differences can add up but seem to remain below 1% of the original pulse amplitude.

testing.assert_allclose(traces, reference_array, atol=2e-3  * units.V, rtol=1e-7)
print('T08test_emitter_birefringence passed without issues')


