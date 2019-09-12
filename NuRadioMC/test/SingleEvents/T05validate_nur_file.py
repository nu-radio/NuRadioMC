#!/usr/bin/env python
import sys
from numpy import testing
import numpy as np

import NuRadioReco.modules.io.eventReader
from NuRadioReco.utilities import units

try:
    file1 = sys.argv[1]
    file2 = sys.argv[2]
except:
    print("No files given")
    sys.exit(-1)

print("Testing the files {} and {} for equality".format(file1, file2))

def all_traces(file):
    eventReader1 = NuRadioReco.modules.io.eventReader.eventReader()
    eventReader1.begin(file)
    i = 0
    for iE1, event1 in enumerate(eventReader1.run()):
        for st1, station1 in enumerate(event1.get_stations()):
            for channel1 in station1.iter_channels():
                trace1 = channel1.get_trace()
                if i == 0:
                    all_traces = trace1
                else:
                    all_traces.append(trace1)
    return all_traces

all_traces_1 = all_traces(file1)
all_traces_2 = all_traces(file2)

diff = all_traces_1 - all_traces_2

if np.any(diff != 0):
    print("The arrays are different, difference in traces:", diff)

print("Maximum difference between traces [mV]", np.max(np.abs(diff))/units.mV)

precision = 7
testing.assert_almost_equal(all_traces_1, all_traces_2,decimal=precision)

try:
    testing.assert_equal(all_traces_1, all_traces_2)
except:
    print("Traces agree within {} decimals, but not completely identical".format(precision))

print("Traces are identical")

