from NuRadioReco.modules import (efieldToVoltageConverter, electricFieldBandPassFilter,
                                 channelResampler, channelGenericNoiseAdder,
                                 efieldToVoltageConverterPerEfield, channelBandPassFilter)
import NuRadioReco.modules.io.coreas.readFAERIEShower
import NuRadioReco.modules.trigger.highLowThreshold
import NuRadioReco.modules.io.eventWriter

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
import scipy.constants as constants
import copy


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


def apply_response(station, resp):

    for channel in station.iter_channels():
        channel = channel * resp
        station.add_channel(channel, overwrite=True)


def cut_channel_trace_to_sim_trace(station):

    sim_station = station.get_sim_station()

    channel = next(station.iter_channels())  # only one channel

    sim_channels = list(sim_station.iter_channels())

    readout = BaseTrace()
    if len(sim_channels) == 1:

        readout.set_trace(np.zeros_like(sim_channels[0].get_trace()), sim_channels[0].get_sampling_rate())
        readout.set_trace_start_time(sim_channels[0].get_trace_start_time())

        readout.add_to_trace(channel, raise_error=False)

        channel.set_trace(readout.get_trace(), "same")
        channel.set_trace_start_time(readout.get_trace_start_time())
    else:
        # 2 sim channels
        t_start = [sim_channel.get_trace_start_time() for sim_channel in sim_channels]
        sort = np.argsort(t_start)
        sim_channels = np.array(sim_channels)[sort].tolist()

        t0 = sim_channels[0].get_trace_start_time()
        n_samples = int((sim_channels[1].get_times()[-1] - t0) * sim_channels[0].get_sampling_rate())
        if n_samples % 2:
            n_samples += 1

        readout.set_trace(np.zeros(n_samples), sim_channels[0].get_sampling_rate())
        readout.set_trace_start_time(t0)

        for sim_channel in sim_station.iter_channels():
            readout_tmp = copy.deepcopy(readout)
            readout_tmp.add_to_trace(sim_channel, raise_error=False)
            sim_channel.set_trace(readout_tmp.get_trace(), "same")
            sim_channel.set_trace_start_time(readout_tmp.get_trace_start_time())

        readout.add_to_trace(channel, raise_error=False)
        channel.set_trace(readout.get_trace(), "same")
        channel.set_trace_start_time(readout.get_trace_start_time())


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

bandpass_filter = electricFieldBandPassFilter.electricFieldBandPassFilter()
bandpass_filter.begin()

channel_bandpass_filter = channelBandPassFilter.channelBandPassFilter()
channel_bandpass_filter.begin()

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
    'SimChannels': True,
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

        efield_converter_per_efield.run(event, station, det)
        efield_converter.run(event, station, det)

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
        apply_response(sim_station, resp_st23_ch0)

        # # cuts and padds the channel and sim channel traces to the exact same window
        # cut_channel_trace_to_sim_trace(station)

        # channelResampler.run(event, station, None, sampling_rate=0.472 * units.GHz)
        # channelResampler.run(event, sim_station, None, sampling_rate=0.472 * units.GHz)

        if args.plot_traces:
            plot_traces(event)
            sys.exit()

    eventWriter.run(event, mode=mode)
