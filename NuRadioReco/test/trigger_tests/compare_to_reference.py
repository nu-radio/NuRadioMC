#!/usr/bin/env python
import json
import NuRadioReco.modules.io.eventReader
import argparse
import numpy as np
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--create_reference', help='create new reference instead of comparing to current reference', action='store_true')
args = parser.parse_args()

event_reader = NuRadioReco.modules.io.eventReader.eventReader()
event_reader.begin('NuRadioReco/test/trigger_tests/trigger_test_output.nur')

trigger_names = ['default_simple_threshold', 'default_high_low', 'default_multi_high_low', 'simple_phased_threshold']
properties = ['triggered', 'trigger_time', 'triggered_channels']
trigger_results = {}

if not args.create_reference:
    with open('NuRadioReco/test/trigger_tests/reference.json', 'r') as f:
        reference = json.load(f)

for event in event_reader.run():
    station = event.get_station(1)
    for trigger_name in trigger_names:
        trigger = station.get_trigger(trigger_name)
        if trigger_name not in trigger_results.keys():
            trigger_results[trigger_name] = {}
        for prop in properties:
            if prop not in trigger_results[trigger_name].keys():
                trigger_results[trigger_name][prop] = []
            trigger_results[trigger_name][prop].append(trigger.get_trigger_settings()[prop])

found_error = False
if args.create_reference:
    with open('NuRadioReco/test/trigger_tests/reference.json', 'w') as f:
        json.dump(trigger_results, f, sort_keys=True,
                  indent=4, separators=(',', ': '))
else:
    for trigger_name in trigger_names:
        for prop in properties:
            if(prop == "trigger_time"):
                try:
                    np.testing.assert_allclose(np.array(trigger_results[trigger_name][prop], dtype=np.float64), np.array(reference[trigger_name][prop], dtype=np.float64))
                except AssertionError as e:
                    print('Property {} of trigger {} differs from reference'.format(prop, trigger_name))
                    print(e)
                    found_error = True
            else:
                try:
                    np.testing.assert_equal(trigger_results[trigger_name][prop], reference[trigger_name][prop])
                except AssertionError as e:
                    print('Property {} of trigger {} differs from reference'.format(prop, trigger_name))
                    print(e)
                    found_error = True

if found_error:
    sys.exit(-1)
else:
    print('Trigger test passed without issues')
