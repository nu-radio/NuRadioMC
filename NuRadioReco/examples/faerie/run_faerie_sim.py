import NuRadioReco.modules.io.coreas.readFAERIEShower
import NuRadioReco.modules.io.eventWriter

from NuRadioReco.detector import detector
from NuRadioReco.utilities import units, signal_processing

from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.base_trace import BaseTrace

from NuRadioMC.examples.RNO_G_trigger_simulation.simulate import \
    detector_simulation_with_data_driven_noise, rnog_flower_board_high_low_trigger_simulations

import NuRadioReco.modules.channelReadoutWindowCutter
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator

from NuRadioReco.examples.faerie.detector import FAERIEDetector

import datetime as dt
import numpy as np
import logging
import argparse
import copy


def pad_traces(event, det, pad_before=20 * units.ns, pad_after=20 * units.ns):
    """ Makes sure all traces have the same length and starting time. """
    sim_station = event.get_station().get_sim_station()

    tstarts = []
    tends = []
    for electric_field in sim_station.get_electric_fields():
        times = electric_field.get_times()
        tstarts.append(times[0])
        tends.append(times[-1])

    tstart = np.min(tstarts) + pad_before
    tend = np.max(tends) + pad_after

    t_readout_window = det.get_number_of_samples(sim_station.get_id(), 0) / \
        det.get_sampling_frequency(sim_station.get_id(), 0)

    if tend - tstart < t_readout_window:
        tend = tstart + t_readout_window + pad_after * units.ns

    # assumes all efields have the same sampling rate
    n_samples = int((tend - tstart) * electric_field.get_sampling_rate())
    if n_samples % 2 != 0:
        n_samples += 1

    for electric_field in sim_station.get_electric_fields():
        readout = BaseTrace()
        readout.set_trace(np.zeros((3, n_samples)), electric_field.get_sampling_rate(), tstart)

        readout.add_to_trace(electric_field)
        electric_field.set_trace(readout.get_trace(), "same", tstart)



def split_events(event, det, trigger_channels):
    """ Split an event with more than 4 channels into multiple events with 4 channels. """

    det.set_event(event)
    station = event.get_station()
    if (len(det.get_channel_ids(station.get_id())) == len(trigger_channels) and
        np.all(det.get_channel_ids(station.get_id()) == trigger_channels)):
        return [event]

    if len(det.get_channel_ids(station.get_id())) == len(trigger_channels):
        raise ValueError("Some thing unexpected happend. The event has only 4 channels but "
                         f"the channel ids {det.get_channel_ids(station.get_id())} do not "
                         f"match the trigger channels ({trigger_channels})")

    # Split the event into multiple events
    sim_channel_ids = np.unique([efields.get_channel_ids() for efields in station.get_sim_station().get_electric_fields()])
    channel_positions = np.array([det.get_relative_position(station.get_id(), sim_channel_id) for sim_channel_id in sim_channel_ids])

    unique_xy_positions = np.unique(channel_positions[:, :2], axis=0)
    n_batches = len(unique_xy_positions)

    sim_channel_ids_batches = [[] for _ in range(n_batches)]
    for sim_channel_id, xy_position in zip(sim_channel_ids, channel_positions[:, :2]):
        idx = np.arange(n_batches)[np.all(unique_xy_positions == xy_position, axis=1)][0]
        sim_channel_ids_batches[idx].append(sim_channel_id)

    events = []
    for sim_channel_ids_batch in sim_channel_ids_batches:
        new_event = copy.deepcopy(event)


        if len(sim_channel_ids_batch) != len(trigger_channels):
            raise ValueError("Some thing unexpected happend. The batch has not the same number of channels as the trigger channels "
                             f"sim_channel_ids_batch: {sim_channel_ids_batch}, trigger_channels: {trigger_channels}")

        new_sim_station = NuRadioReco.framework.sim_station.SimStation(station.get_id())  # set sim station id to 0
        new_sim_station.set_is_neutrino() # HACK: Since the sim. efields are always at the exact positions as the antenna(channels).
        new_station = NuRadioReco.framework.station.Station(station.get_id())
        new_station.set_sim_station(new_sim_station)
        new_event.set_station(new_station)  # overwrites existing station
        events.append(new_event)

        depth = np.array([det.get_relative_position(station.get_id(), sim_channel_id)[2] for sim_channel_id in sim_channel_ids_batch])
        sort = np.argsort(depth)

        sorted_sim_channel_ids_batch = np.array(sim_channel_ids_batch)[sort]
        for sim_channel_id, new_id in zip(sorted_sim_channel_ids_batch, trigger_channels):
            for efield in station.get_sim_station().get_electric_fields_for_channels([sim_channel_id]):
                efield_new = copy.deepcopy(efield)
                efield_new.set_channel_ids([new_id])
                new_sim_station.add_electric_field(efield_new)


    return events


if __name__ == "__main__":
    # Parse eventfile as argument   
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('inputfilename', type=str, nargs='*',
                        help='path to NuRadioMC simulation result')


    parser.add_argument('--add_noise', action='store_true', help='Add noise to the traces')

    parser.add_argument('--depth', nargs="?", type=float, default=None, help='If specified, used to select simulated pulses at a given depth.')

    parser.add_argument('--detector_file', type=str, nargs='?',
                        default=None,
                        help='path to detectordescription')

    parser.add_argument('--noise_type', type=str, nargs='?',
                        default="rayleigh",
                        help='Specify noise type')

    parser.add_argument('--output_file', type=str, nargs='?',
                        default=None,
                        help='path to detectordescription')

    parser.add_argument('--station', type=int, nargs='?',
                        default=11,
                        help='station to simulate')

    args = parser.parse_args()


    # Load the real detector response
    det_rnog = detector.rnog_detector.Detector(
        select_stations=[args.station], detector_file=args.detector_file, database_connection="RNOG_public", always_query_entire_description=False)
    det_rnog.update(dt.datetime(2023, 8, 1))

    trigger_channels = np.array([0, 1, 2, 3])
    # trigger_channels = np.array([0])
    
    thresholds = {
        "hilo_sigma_3": 3,
        "hilo_sigma_3.8": 3.8,
        "hilo_sigma_4": 4,
    }

    # rnog_resp_ch0 = det_rnog.get_signal_chain_response(args.station, 0, trigger=True)
    # vrms_thermal = signal_processing.calculate_vrms_from_temperature(300 * units.kelvin, response=rnog_resp_ch0)
    # print(f"Thermal noise amplitude: {vrms_thermal / units.mV} mV")
    min_freq = 10 * units.MHz
    max_freq = 1200 * units.MHz
    vrms_300K_in_min_max = signal_processing.calculate_vrms_from_temperature(
        300 * units.kelvin, bandwidth=[min_freq, max_freq])

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

    efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
    efieldToVoltageConverter.begin(post_pulse_time=0 * units.ns, pre_pulse_time=50 * units.ns)

    rnogHarwareResponse = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()
    rnogHarwareResponse.begin(trigger_channels=trigger_channels)

    channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
    channelGenericNoiseAdder.begin()

    dummy_detector_for_positions_only = FAERIEDetector()

    mode = {
        'Channels': True,
        'ElectricFields': False,
        'SimChannels': True,
        'SimElectricFields': False
    }

    for combined_event in readFAERIEShower.run(depth=args.depth, station_id=args.station):

        for edx, event in enumerate(split_events(combined_event, dummy_detector_for_positions_only, trigger_channels)):
            dummy_detector_for_positions_only.set_event(event)
            pad_traces(event, det_rnog)

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
                assert np.all(dummy_detector_for_positions_only.get_channel_ids(station.get_id()) == trigger_channels), "Expected channels [0, 1, 2, 3]"
                channel_depths = np.array([dummy_detector_for_positions_only.get_relative_position(
                    station.get_id(), channel_id)[2] for channel_id in dummy_detector_for_positions_only.get_channel_ids(station.get_id())])
                assert np.all(np.argsort(channel_depths) == trigger_channels), "Expected channels to be sorted by depth"

                if args.add_noise and args.noise_type == "data-driven":
                    detector_simulation_with_data_driven_noise(
                        event, station, det_rnog, trigger_channels=trigger_channels)
                else:
                    assert args.noise_type == "rayleigh", "Only 'rayleigh' and 'data-driven' noise is supported."
                    efieldToVoltageConverter.run(event, station, det_rnog, channel_ids=trigger_channels)

                    if args.add_noise:
                        channelGenericNoiseAdder.run(
                            event, station, det_rnog,
                            amplitude=vrms_300K_in_min_max, min_freq=min_freq, max_freq=max_freq,
                            type='rayleigh')

                    rnogHarwareResponse.run(event, station, det_rnog, sim_to_data=True)

                rnog_flower_board_high_low_trigger_simulations(
                    event, station, det_rnog, trigger_channels=trigger_channels,
                    trigger_channel_noise_vrms=None,
                    high_low_trigger_thresholds=thresholds)

                channelReadoutWindowCutter.run(event, station, det_rnog)
                channelResampler.run(event, station, det_rnog)

            eventWriter.run(event, mode=mode)
