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
    parameter_values[event.get_id()] = {
        'station_parameters': {},
        'sim_station_parameters': {},
        'channel_parameters': {},
        'electric_field_parameters': {}
    }
    station = event.get_station(args.station_id)
    sim_station = station.get_sim_station()
    for param_name in stnp:
        if station.has_parameter(param_name):
            parameter_values[event.get_id()]['station_parameters'][param_name.name] = station.get_parameter(param_name)
        else:
            parameter_values[event.get_id()]['station_parameters'][param_name.name] = None
        if sim_station.has_parameter(param_name):
            parameter_values[event.get_id()]['sim_station_parameters'][param_name.name] = sim_station.get_parameter(param_name)
        else:
            parameter_values[event.get_id()]['sim_station_parameters'][param_name.name] = None
    for channel in station.iter_channels():
        for param_name in chp:
            if param_name.name not in parameter_values[event.get_id()]['channel_parameters'].keys():
                parameter_values[event.get_id()]['channel_parameters'][param_name.name] = []
            if channel.has_parameter(param_name):
                parameter_values[event.get_id()]['channel_parameters'][param_name.name].append(channel.get_parameter(param_name))
            else:
                parameter_values[event.get_id()]['channel_parameters'][param_name.name].append(None)
    for electric_field in station.get_electric_fields():
        for param_name in efp:
            if param_name.name not in parameter_values[event.get_id()]['electric_field_parameters'].keys():
                parameter_values[event.get_id()]['electric_field_parameters'][param_name.name] = []
            if electric_field.has_parameter(param_name):
                value = electric_field.get_parameter(param_name)
                if isinstance(value, np.ndarray):
                    value = list(value)
                parameter_values[event.get_id()]['electric_field_parameters'][param_name.name].append(value)
            else:
                parameter_values[event.get_id()]['electric_field_parameters'][param_name.name].append(None)

if args.create_reference:
    with open(args.reference_file, 'w') as f:
        json.dump(parameter_values, f)          
else:
    with open(args.reference_file, 'r') as f:
        parameter_reference = json.load(f)
        found_error = False
        for event in event_reader.run():
            try:
                np.testing.assert_almost_equal(parameter_values[event.get_id()]['station_parameters'], parameter_reference[str(event.get_id())]['station_parameters'])
            except AssertionError as e:
                print('station paramters of event {} differ from reference'.format(event.get_id()))
                print(e)
                found_error = True
            try:
                np.testing.assert_almost_equal(parameter_values[event.get_id()]['sim_station_parameters'], parameter_reference[str(event.get_id())]['sim_station_parameters'])
            except AssertionError as e:
                print('sim station paramters of event {} differ from reference'.format(event.get_id()))
                print(e)
                found_error = True
            try:
                np.testing.assert_almost_equal(parameter_values[event.get_id()]['channel_parameters'], parameter_reference[str(event.get_id())]['channel_parameters'])
            except AssertionError as e:
                print('channel paramters of event {} differ from reference'.format(event.get_id()))
                print(e)
                found_error = True
            try:
                np.testing.assert_almost_equal(parameter_values[event.get_id()]['electric_field_parameters'], parameter_reference[str(event.get_id())]['electric_field_parameters'])
            except AssertionError as e:
                print('E-field paramters of event {} differ from reference'.format(event.get_id()))
                print(e)
                found_error = True
    if found_error:
        sys.exit(-1)
    else:
        print('TinyReconstruction test passed wihtout issues')