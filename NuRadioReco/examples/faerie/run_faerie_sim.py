import NuRadioReco.modules.io.coreas.readFAERIEShower
import NuRadioReco.modules.io.eventWriter

from NuRadioReco.detector import detector
from NuRadioReco.utilities import units, signal_processing

from NuRadioReco.framework.parameters import showerParameters as shp

from NuRadioMC.examples.RNO_G_trigger_simulation.simulate import \
    detector_simulation_with_data_driven_noise, rnog_flower_board_high_low_trigger_simulations

import NuRadioReco.modules.channelReadoutWindowCutter
import NuRadioReco.modules.channelResampler

from NuRadioReco.examples.faerie.detector import FAERIEDetector

from matplotlib import pyplot as plt
from collections import defaultdict
import datetime as dt
import numpy as np
import logging
import argparse


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

parser.add_argument('--station', type=int, nargs='?',
                    default=11,
                    help='station to simulate')

args = parser.parse_args()



# Load the real detector response
det_rnog = detector.rnog_detector.Detector(select_stations=[args.station], database_connection="RNOG_public", always_query_entire_description=False)
det_rnog.update(dt.datetime(2023, 8, 1))

trigger_channels = np.array([0, 1, 2, 3])
thresholds = {
    "hilo_sigma_3": 3,
    "hilo_sigma_3.8": 3.8,
    "hilo_sigma_4": 4,
}

# rnog_resp_ch0 = det_rnog.get_signal_chain_response(args.station, 0, trigger=True)
# vrms_thermal = signal_processing.calculate_vrms_from_temperature(300 * units.kelvin, response=rnog_resp_ch0)
# print(f"Thermal noise amplitude: {vrms_thermal / units.mV} mV")

channelReadoutWindowCutter = NuRadioReco.modules.channelReadoutWindowCutter.channelReadoutWindowCutter()
channelReadoutWindowCutter.begin()

channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelResampler.begin()

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

det = FAERIEDetector()

data = defaultdict(list)

mode = {
    'Channels': True,
    'ElectricFields': False,
    'SimChannels': False,
    'SimElectricFields': False
}

for edx, event in enumerate(readFAERIEShower.run(depth=args.depth, station_id=args.station)):
    det.set_event(event)

    shower = event.get_first_sim_shower()
    for sdx, station in enumerate(event.get_stations()):
        sim_station = station.get_sim_station()

        if (edx + sdx) % 100 == 0:
            print(f"Processing event: {event.get_id()} station {station.get_id()}")
            print(f"Energy: {shower.get_parameter(shp.energy) / units.PeV} PeV, "
                  f"Zenith: {shower.get_parameter(shp.zenith) / units.deg}, "
                  f"Azimuth: {shower.get_parameter(shp.azimuth) / units.deg}")

        # Temporary sanity checks - to apply the correct noise and filter the event
        # can only have 4 channels with IDs [0, 1, 2, 3] (and they should be at the
        # correct depths)
        assert np.all(det.get_channel_ids(station.get_id()) == trigger_channels), "Expected channels [0, 1, 2, 3]"
        channel_depths = np.array([det.get_relative_position(
            station.get_id(), channel_id)[2] for channel_id in det.get_channel_ids(station.get_id())])
        assert np.all(np.argsort(channel_depths) == trigger_channels), "Expected channels to be sorted by depth"

        detector_simulation_with_data_driven_noise(
            event, station, det_rnog, trigger_channels=trigger_channels)

        rnog_flower_board_high_low_trigger_simulations(
            event, station, det_rnog, trigger_channels=trigger_channels,
            trigger_channel_noise_vrms=None,
            high_low_trigger_thresholds=thresholds)

        channelReadoutWindowCutter.run(event, station, det)
        channelResampler.run(event, station, det)

        if args.plot_traces:
            plot_traces(event)

    eventWriter.run(event, mode=mode)
