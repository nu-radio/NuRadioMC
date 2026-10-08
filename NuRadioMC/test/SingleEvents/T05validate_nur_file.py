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


# relative tolerance for the channel/station parameters (e.g. calculated by the channelSignalReconstructor).
# Tiny platform-dependent differences in the traces are amplified for low-amplitude channels.
parameter_rtol = 1e-2

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

def all_parameters(file):
    """ Returns a dict {(run, event, station, channel or None): {parameter name: value}} of all stored parameters """
    reader = NuRadioReco.modules.io.eventReader.eventReader()
    reader.begin(file)
    parameters = {}
    for event in reader.run():
        for station in event.get_stations():
            key = (event.get_run_number(), event.get_id(), station.get_id())
            parameters[key + (None,)] = {str(k): v for k, v in station._parameters.items()}
            for channel in station.iter_channels(sorted=True):
                parameters[key + (channel.get_id(),)] = {str(k): v for k, v in channel._parameters.items()}
    return parameters


def assert_parameters_allclose(value1, value2, rtol, name):
    if isinstance(value1, dict):
        assert value1.keys() == value2.keys(), f"{name}: keys differ ({sorted(value1)} vs. {sorted(value2)})"
        for key in value1:
            assert_parameters_allclose(value1[key], value2[key], rtol, f"{name}/{key}")
    else:
        testing.assert_allclose(value1, value2, rtol=rtol, atol=0, err_msg=f"Parameter {name} differs")


all_traces_1, trace_start_times_1 = all_traces(file1)
all_traces_2, trace_start_times_2 = all_traces(file2)

diff = all_traces_1 - all_traces_2

if np.any(diff != 0):
    print("The arrays are different, difference in traces:", diff)

print("Maximum difference between traces [mV]", np.max(np.abs(diff))/units.mV)

testing.assert_almost_equal(all_traces_1, all_traces_2,decimal=precision)

# check that the trace_start_times are all equal
# Trace start times depend on the raytraced travel time, which is computed in the C++ raytracer
# via gsl_integration_qags with epsrel=1e-6 (see get_travel_time() in
# NuRadioMC/SignalProp/CPPAnalyticRayTracing/analytic_raytracing.cpp). That tolerance is looser
# than this test's default decimal=7 (~1.5e-7 absolute), so last-bit differences in GSL's
# adaptive quadrature across GSL versions/platforms (observed up to ~9e-5ns) can legitimately
# exceed decimal=7 without indicating an actual regression. A looser tolerance is used here than
# for the trace amplitudes above, which are not affected by this.
start_time_decimal = min(precision, 3)
testing.assert_almost_equal(
    trace_start_times_1, trace_start_times_2, decimal=start_time_decimal,
    err_msg=f"Trace start times are not equal (maximum difference: {max(np.abs(trace_start_times_1-trace_start_times_2))})")

# check that all channel and station parameters agree
parameters_1 = all_parameters(file1)
parameters_2 = all_parameters(file2)
assert parameters_1.keys() == parameters_2.keys(), "Files contain different events/stations/channels"
for key in parameters_1:
    assert_parameters_allclose(parameters_1[key], parameters_2[key], parameter_rtol, str(key))
print(f"Channel and station parameters agree within rtol={parameter_rtol}")

try:
    testing.assert_equal(all_traces_1, all_traces_2)
except:
    print("Traces agree within {} decimals, but not completely identical".format(precision))

print("Traces are identical")

