import NuRadioReco.modules.io.coreas.readFAERIEShower

from NuRadioReco.modules import efieldToVoltageConverter


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

args = parser.parse_args()

efield_converter = efieldToVoltageConverter.efieldToVoltageConverter()
efield_converter.begin()

readFAERIEShower = NuRadioReco.modules.io.coreas.readFAERIEShower.readFAERIEShower()
readFAERIEShower.begin(args.inputfilename)

det = NuRadioReco.modules.io.coreas.readFAERIEShower.FAERIEDetector()

for event in readFAERIEShower.run():
    det.set_event(event)

    print('Event {} {}'.format(event.get_run_number(), event.get_id()))
    print('Number of stations: {}'.format(len(list(event.get_stations()))))

    for station in event.get_stations():

        sim_station = station.get_sim_station()

        print(f"Number of electric fields: {len(sim_station.get_electric_fields())}")



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