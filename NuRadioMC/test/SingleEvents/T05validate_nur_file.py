#!/usr/bin/env python3
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

try:
    precision = int(sys.argv[3])
except:
    precision = 7


print("Testing the files {} and {} for equality".format(file1, file2))

def all_traces(file, return_trace_start_times=True):
    eventReader1 = NuRadioReco.modules.io.eventReader.eventReader()
    eventReader1.begin(file)
    i = 0
    trace_start_times = []
    for iE1, event1 in enumerate(eventReader1.run()):
        for st1, station1 in enumerate(event1.get_stations()):
            # print(f"eventid {event1.get_run_number()} station id: {station1.get_id()}")
            for channel1 in station1.iter_channels(sorted=True):
                trace1 = channel1.get_trace()
                # print(channel1.get_id(), channel1.get_trace_start_time(), channel1.get_sampling_rate())
                if i == 0:
                    all_traces = trace1
                else:
                    all_traces = np.append(all_traces, trace1)
                    # print(f"apending trace {len(trace1)} to all_traces {len(all_traces)}")
                trace_start_times += [channel1.get_trace_start_time()]
                i += 1
    if return_trace_start_times:
        return all_traces, np.array(trace_start_times)
    return all_traces

all_traces_1, trace_start_times_1 = all_traces(file1)
all_traces_2, trace_start_times_2 = all_traces(file2)

diff = all_traces_1 - all_traces_2

if np.any(diff != 0):
    print("The arrays are different, difference in traces:", diff)

print("Maximum difference between traces [mV]", np.max(np.abs(diff))/units.mV)

testing.assert_almost_equal(all_traces_1, all_traces_2,decimal=precision)

# check that the trace_start_times are all equal
testing.assert_almost_equal(
    trace_start_times_1, trace_start_times_2, decimal=precision,
    err_msg=f"Trace start times are not equal (maximum difference: {max(np.abs(trace_start_times_1-trace_start_times_2))})")

try:
    testing.assert_equal(all_traces_1, all_traces_2)
except:
    print("Traces agree within {} decimals, but not completely identical".format(precision))

print("Traces are identical")

