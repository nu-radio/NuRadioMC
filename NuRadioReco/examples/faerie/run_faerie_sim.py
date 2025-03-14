from NuRadioReco.modules import (
    efieldToVoltageConverter, channelResampler, channelGenericNoiseAdder)

import NuRadioReco.modules.io.coreas.readFAERIEShower
import NuRadioReco.modules.io.eventWriter

import NuRadioReco.modules.trigger.highLowThreshold

import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
import NuRadioReco.modules.RNO_G.triggerBoardResponse

from NuRadioReco.framework.base_trace import BaseTrace

from NuRadioReco.detector import detector
from NuRadioReco.utilities import units, signal_processing

from NuRadioReco.framework.parameters import showerParameters as shp

import argparse
from matplotlib import pyplot as plt
import sys
import numpy as np
from collections import defaultdict
import datetime as dt
import logging


def plot_traces(event):
    station = event.get_station()
    sim_station = station.get_sim_station()
    channel = next(station.iter_channels())

    fig, ax = plt.subplots()
    ax.plot(channel.get_times(), channel.get_trace(), lw=2, color='k')

    for sim_channel in sim_station.iter_channels():
        print(sim_channel.get_id(), channel.get_id())
        if sim_channel.get_id() == channel.get_id():
            ax.plot(sim_channel.get_times(), sim_channel.get_trace(), lw=1)

    fig.tight_layout()
    plt.show()


# Parse eventfile as argument
parser = argparse.ArgumentParser(description='')
parser.add_argument('inputfilename', type=str, nargs='*',
                    help='path to NuRadioMC simulation result')


parser.add_argument('--add_noise', action='store_true', help='Add noise to the traces')
parser.add_argument('--plot_traces', action='store_true', help='Plot the traces')
parser.add_argument('--depth', nargs="?", type=float, default=None, help='If specified, used to select simulated pulses at a given depth.')

parser.add_argument('--output_file', type=str, nargs='?',
                    default=None,
                    help='path to detectordescription')

args = parser.parse_args()

# Load the real detector response
det_rnog = detector.rnog_detector.Detector(select_stations=[23], database_connection="RNOG_public", always_query_entire_description=False)
det_rnog.update(dt.datetime(2023, 8, 1))

resp_st23_ch0 = det_rnog.get_signal_chain_response(23, 0, trigger=True)
vrms_thermal = signal_processing.calculate_vrms_from_temperature(300 * units.kelvin, response=resp_st23_ch0)
print(f"Thermal noise amplitude: {vrms_thermal / units.mV} mV")

efield_converter = efieldToVoltageConverter.efieldToVoltageConverter()
efield_converter.begin()

efield_converter_per_efield = efieldToVoltageConverterPerEfield.efieldToVoltageConverterPerEfield()

channelResampler = channelResampler.channelResampler()

channelGenericNoiseAdder = channelGenericNoiseAdder.channelGenericNoiseAdder()
channelGenericNoiseAdder.begin()

eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
if args.output_file is not None:
    outputfilename = args.output_file
else:
    outputfilename = args.inputfilename[0].replace(".hdf5", ".nur")

eventWriter.begin(filename=outputfilename, max_file_size=1024 * 3)

readFAERIEShower = NuRadioReco.modules.io.coreas.readFAERIEShower.readFAERIEShower()
readFAERIEShower.begin(
    args.inputfilename, logger_level=logging.INFO
)

det = NuRadioReco.modules.io.coreas.readFAERIEShower.FAERIEDetector()

data = defaultdict(list)

mode = {
    'Channels': True,
    'ElectricFields': False,
    'SimChannels': False,
    'SimElectricFields': False
}

for edx, event in enumerate(readFAERIEShower.run(depth=args.depth)):
    det.set_event(event)

    shower = event.get_first_sim_shower()
    for sdx, station in enumerate(event.get_stations()):
        sim_station = station.get_sim_station()

        if (edx + sdx) % 100 == 0:
            print(f"Processing event: {event.get_id()} station {station.get_id()}")
            print(f"Energy: {shower.get_parameter(shp.energy) / units.PeV} PeV, "
                  f"Zenith: {shower.get_parameter(shp.zenith) / units.deg}, "
                  f"Azimuth: {shower.get_parameter(shp.azimuth) / units.deg}")

        efield_converter.run(event, station, det)


        # Sanity checks for the moment
        assert det.get_channel_ids(station.get_id()).tolist() == [0, 1, 2, 3], "Expected channels [0, 1, 2, 3]"
        channel_depths = np.array([det.get_relative_position(station.get_id(), channel_id)[2] for channel_id in det.get_channel_ids(station.get_id())])
        assert np.argsort(channel_depths).tolist() == [0, 1, 2, 3], "Expected channels to be sorted by depth"



        print(f"Number of channels: {len(station.get_channel_ids())}")

        if args.add_noise:
            # The noise amplitude corresponds rougthly to 300K within a bandwidth of 950 MHz
            channelGenericNoiseAdder.run(
                event, station, det,
                amplitude=14 * units.microvolt,
                min_freq=50 * units.MHz,
                max_freq=1000 * units.MHz,
                type='rayleigh',
                bandwidth=950 * units.MHz)

        apply_response(station, resp_st23_ch0)

        # # cuts and padds the channel and sim channel traces to the exact same window
        # cut_channel_trace_to_sim_trace(station)

        # channelResampler.run(event, station, None, sampling_rate=0.472 * units.GHz)
        # channelResampler.run(event, sim_station, None, sampling_rate=0.472 * units.GHz)

        if args.plot_traces:
            plot_traces(event)
            sys.exit()

    eventWriter.run(event, mode=mode)
