#!/usr/bin/env python
import sys
from numpy import testing

import NuRadioReco.modules.io.eventReader

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


try:
    testing.assert_equal(all_traces_1, all_traces_2)
    print("All traces in .nur file are identical.")
except:
    print("Traces in .nur file are NOT identical.")
    sys.exit(-1)

