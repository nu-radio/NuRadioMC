import NuRadioReco.modules.io.coreas.readFEARIEShower

from NuRadioReco.modules import efieldToVoltageConverter

from NuRadioReco.detector import generic_detector as detector

from NuRadioReco.utilities import units

import argparse
from matplotlib import pyplot as plt
import sys
import os
import numpy as np


# Parse eventfile as argument
parser = argparse.ArgumentParser(description='NuRadioSim file')
parser.add_argument('inputfilename', type=str, nargs='*',
                    default=['example_data/example_event.h5'],
                    help='path to NuRadioMC simulation result')

default_path = os.path.join(os.path.dirname(__file__), "RNO_single_channel.json")
parser.add_argument('--detectordescription', type=str, nargs='?',
                    default=default_path,
                    help='path to detectordescription')

args = parser.parse_args()

det = detector.GenericDetector(
    json_filename=args.detectordescription)

efield_converter = efieldToVoltageConverter.efieldToVoltageConverter()
efield_converter.begin()

readFEARIEShower = NuRadioReco.modules.io.coreas.readFEARIEShower.readFEARIEShower()
readFEARIEShower.begin(args.inputfilename, det=det)

for event, det in readFEARIEShower.run(depth=100):
    print('Event {} {}'.format(event.get_run_number(), event.get_id()))
    print('Number of stations: {}'.format(len(list(event.get_stations()))))
    for station in event.get_stations():

        sim_station = station.get_sim_station()

        if len(sim_station.get_electric_fields()) < 2:
            continue

        print(f"Number of electric fields: {len(sim_station.get_electric_fields())}")
        for efield in sim_station.get_electric_fields():
            print(efield.get_position())

        efield_converter.run(event, station, det)
        print(f"Number of channels: {len(station.get_channel_ids())}")
        channel = station.get_channel(0)

        fig, axs = plt.subplots(1, 2)
        axs[0].plot(channel.get_times(), channel.get_trace() / units.mV)

        axs[0].set_xlabel('time / ns')
        axs[0].set_ylabel('voltage / mV')

        axs[1].plot(channel.get_frequencies(), np.abs(channel.get_frequency_spectrum()))
        axs[1].set_xlabel('frequencies / GHz')
        axs[1].set_xlim(None, 1.2)

        plt.show()
        # sys.exit()