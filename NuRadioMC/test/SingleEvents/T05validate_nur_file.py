#!/usr/bin/env python3
import argparse
from numpy import testing
import numpy as np

import NuRadioReco.modules.io.eventReader
from NuRadioReco.utilities import units
from NuRadioMC.utilities._test_helpers import TIME_ATOL, PARAMETER_RTOL, NOISE_SENSITIVE_PARAMETERS

parser = argparse.ArgumentParser(description="Compare traces and channel/station parameters of two .nur files")
parser.add_argument('file1')
parser.add_argument('file2', help="reference file")
parser.add_argument('precision', type=int, nargs='?', default=7, help="decimals to which the traces have to agree")
parser.add_argument('--parameter-rtol', type=float, default=PARAMETER_RTOL,
                    help="relative tolerance for parameters which depend only on the signal")
parser.add_argument('--noise-parameter-rtol', type=float, default=None,
                    help="relative tolerance for NOISE_SENSITIVE_PARAMETERS. If not given (default, "
                    "e.g. for noiseless simulations), only their presence is checked")
args = parser.parse_args()
file1, file2, precision = args.file1, args.file2, args.precision

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
    """ Returns a dict {(run, event, station, channel or None): {parameter: value}} of all stored parameters """
    reader = NuRadioReco.modules.io.eventReader.eventReader()
    reader.begin(file)
    parameters = {}
    for event in reader.run():
        for station in event.get_stations():
            key = (event.get_run_number(), event.get_id(), station.get_id())
            parameters[key + (None,)] = dict(station._parameters)
            for channel in station.iter_channels(sorted=True):
                parameters[key + (channel.get_id(),)] = dict(channel._parameters)
    return parameters


def relative_difference(value1, value2):
    """ Maximum relative difference |value1 - value2| / |value2| (0 for identical values, including inf and nan) """
    value1, value2 = np.asarray(value1, dtype=float), np.asarray(value2, dtype=float)
    identical = (value1 == value2) | (np.isnan(value1) & np.isnan(value2))
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.where(identical, 0, np.abs(value1 - value2) / np.abs(value2))
    return float(np.max(np.nan_to_num(rel, nan=np.inf), initial=0))


def compare_parameters(value1, value2, rtol, name, parameter, largest_differences):
    """
    Returns a list of error messages for all (nested) parameter values which differ (only keys are
    compared if ``rtol`` is None). The largest relative difference of each parameter (and where it
    occurs) is stored in ``largest_differences``.
    """
    if isinstance(value1, dict):
        if value1.keys() != value2.keys():
            return [f"{name}: keys differ ({sorted(value1)} vs. {sorted(value2)})"]
        return [error for key in value1 for error in compare_parameters(
            value1[key], value2[key], rtol, f"{name}/{key}", f"{parameter}/{key}", largest_differences)]
    rel = relative_difference(value1, value2)
    if rel > largest_differences.get(parameter, (-1, None))[0]:
        largest_differences[parameter] = (rel, name)
    if rtol is None:
        return []
    try:
        testing.assert_allclose(value1, value2, rtol=rtol, atol=0, err_msg=f"Parameter {name} differs")
    except AssertionError as e:
        return [str(e)]
    return []

all_traces_1, trace_start_times_1 = all_traces(file1)
all_traces_2, trace_start_times_2 = all_traces(file2)

diff = all_traces_1 - all_traces_2

if np.any(diff != 0):
    print("The arrays are different, difference in traces:", diff)

print("Maximum difference between traces [mV]", np.max(np.abs(diff))/units.mV)

testing.assert_almost_equal(all_traces_1, all_traces_2,decimal=precision)

# check that the trace_start_times are all equal (they depend on the raytraced travel times)
testing.assert_allclose(
    trace_start_times_1, trace_start_times_2, rtol=0, atol=TIME_ATOL,
    err_msg=f"Trace start times are not equal (maximum difference: {max(np.abs(trace_start_times_1-trace_start_times_2))})")

# check that all channel and station parameters agree
parameters_1 = all_parameters(file1)
parameters_2 = all_parameters(file2)
assert parameters_1.keys() == parameters_2.keys(), "Files contain different events/stations/channels"
# collect all differences (instead of stopping at the first one) to see the full picture in a single run
errors = []
largest_differences = {}
for key in parameters_1:
    if parameters_1[key].keys() != parameters_2[key].keys():
        errors.append(f"{key}: parameters differ ({sorted(map(str, parameters_1[key]))} vs. "
                      f"{sorted(map(str, parameters_2[key]))})")
        continue
    for parameter in parameters_1[key]:
        rtol = args.noise_parameter_rtol if parameter.name in NOISE_SENSITIVE_PARAMETERS else args.parameter_rtol
        errors += compare_parameters(parameters_1[key][parameter], parameters_2[key][parameter], rtol,
                                     f"{key}/{parameter}", str(parameter), largest_differences)

# print the largest relative difference of each parameter, to know how tight the tolerance can be
print("Largest relative difference per parameter (* = noise sensitive):")
for parameter, (rel, name) in sorted(largest_differences.items(), key=lambda item: -item[1][0]):
    flag = '*' if parameter.split('/')[0].split('.')[-1] in NOISE_SENSITIVE_PARAMETERS else ' '
    print(f"  {rel:9.2e} {flag} {parameter}  (at {name})")
if errors:
    print("\n".join(errors))
    raise AssertionError(
        f"{len(errors)} channel/station parameter(s) differ by more than rtol={args.parameter_rtol} "
        f"(noise sensitive: rtol={args.noise_parameter_rtol})")
print(f"Channel and station parameters agree within rtol={args.parameter_rtol} "
      f"(noise sensitive: {'not compared' if args.noise_parameter_rtol is None else f'rtol={args.noise_parameter_rtol}'})")

try:
    testing.assert_equal(all_traces_1, all_traces_2)
except:
    print("Traces agree within {} decimals, but not completely identical".format(precision))

print("Traces are identical")

