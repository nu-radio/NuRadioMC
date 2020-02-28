#!/usr/bin/env python
import json
import NuRadioReco.modules.io.eventReader
import argparse
import numpy as np
import sys
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import electricFieldParameters as efp

parser = argparse.ArgumentParser()
parser.add_argument('filename', help='name of the file that should be compared to reference')
parser.add_argument('reference_file', help='name of the reference file')
parser.add_argument('--station_id', default=32)
parser.add_argument('--create_reference', help='create new reference instead of comparing to current reference', action='store_true')
args = parser.parse_args()

event_reader = NuRadioReco.modules.io.eventReader.eventReader()
event_reader.begin(args.filename)

parameter_values = {}

for event in event_reader.run():
    parameter_values[f"{event.get_id():d}"] = {
        'station_parameters': {},
        'sim_station_parameters': {},
        'channel_parameters': {},
        'electric_field_parameters': {}
    }
    station = event.get_station(args.station_id)
    sim_station = station.get_sim_station()
    for param_name in stnp:
        if station.has_parameter(param_name):
            parameter_values[f"{event.get_id():d}"]['station_parameters'][param_name.name] = station.get_parameter(param_name)
        else:
            parameter_values[f"{event.get_id():d}"]['station_parameters'][param_name.name] = None
        if sim_station.has_parameter(param_name):
            parameter_values[f"{event.get_id():d}"]['sim_station_parameters'][param_name.name] = sim_station.get_parameter(param_name)
        else:
            parameter_values[f"{event.get_id():d}"]['sim_station_parameters'][param_name.name] = None
    for channel in station.iter_channels():
        for param_name in chp:
            if param_name.name not in parameter_values[f"{event.get_id():d}"]['channel_parameters'].keys():
                parameter_values[f"{event.get_id():d}"]['channel_parameters'][param_name.name] = []
            if channel.has_parameter(param_name):
                parameter_values[f"{event.get_id():d}"]['channel_parameters'][param_name.name].append(channel.get_parameter(param_name))
            else:
                parameter_values[f"{event.get_id():d}"]['channel_parameters'][param_name.name].append(None)
    for electric_field in station.get_electric_fields():
        for param_name in efp:
            if param_name.name not in parameter_values[f"{event.get_id():d}"]['electric_field_parameters'].keys():
                parameter_values[f"{event.get_id():d}"]['electric_field_parameters'][param_name.name] = []
            if electric_field.has_parameter(param_name):
                value = electric_field.get_parameter(param_name)
                if isinstance(value, np.ndarray):
                    value = list(value)
                parameter_values[f"{event.get_id():d}"]['electric_field_parameters'][param_name.name].append(value)
            else:
                parameter_values[f"{event.get_id():d}"]['electric_field_parameters'][param_name.name].append(None)



def assertDeepAlmostEqual(expected, actual, *args, **kwargs):
    """
    Assert that two complex structures have almost equal contents.

    Compares lists, dicts and tuples recursively. Checks numeric values
    using test_case's :py:meth:`unittest.TestCase.assertAlmostEqual` and
    checks all other values with :py:meth:`unittest.TestCase.assertEqual`.
    Accepts additional positional and keyword arguments and pass those
    intact to assertAlmostEqual() (that's how you specify comparison
    precision).

    :param test_case: TestCase object on which we can call all of the basic
    'assert' methods.
    :type test_case: :py:class:`unittest.TestCase` object
    """
    is_root = not '__trace' in kwargs
    trace = kwargs.pop('__trace', 'ROOT')
    try:
        if isinstance(expected, (int, float, complex)):
            np.testing.assert_allclose(actual, expected, *args, **kwargs)
        elif isinstance(expected, (list, tuple, np.ndarray)):
            np.testing.assert_equal(len(expected), len(actual))
            for index in range(len(expected)):
                v1, v2 = expected[index], actual[index]
                assertDeepAlmostEqual(v1, v2,
                                      __trace=repr(index), *args, **kwargs)
        elif isinstance(expected, dict):
            for key in actual:
                if key not in expected:
                    print('Key {} found in reconstruction but not in reference. '\
                        'This is fine if you just added it, but you should update '\
                        'the reference file to include it in future tests!'.format(key))
                else:
                    assertDeepAlmostEqual(expected[key], actual[key],
                                      __trace=repr(key), *args, **kwargs)
        else:
            np.testing.assert_equal(actual, expected)
    except AssertionError as exc:
        exc.__dict__.setdefault('traces', []).append(trace)
        if is_root:
            trace = ' -> '.join(reversed(exc.traces))
            exc = AssertionError("%s\nTRACE: %s" % (exc, trace))
        raise exc


if args.create_reference:
    with open(args.reference_file, 'w') as f:
        json.dump(parameter_values, f)
else:
    with open(args.reference_file, 'r') as f:
        parameter_reference = json.load(f)
        found_error = False
        assertDeepAlmostEqual(parameter_reference, parameter_values, rtol=1e-6)
