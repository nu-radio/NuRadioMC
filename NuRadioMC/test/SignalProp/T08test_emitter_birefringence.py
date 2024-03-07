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

testing.assert_allclose(traces, reference_array, atol=1e-3  * units.V / units.m, rtol=1e-7)
print('T08test_emitter_birefringence passed without issues')


