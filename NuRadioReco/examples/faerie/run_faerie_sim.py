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
import NuRadioReco.modules.efieldToVoltageConverterPerEfield
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator

from NuRadioReco.examples.faerie.detector import FAERIEDetector

import datetime as dt
import numpy as np
import logging
import argparse
import copy


def pad_traces(event, det, pad_before=200 * units.ns, pad_after=400 * units.ns,trigger_channels=[0]):
    """ Makes sure all traces have the same length and starting time. """
    sim_station = event.get_station().get_sim_station()

    tstarts = []
    tends = []
    pulse_times = []
    for electric_field in sim_station.get_electric_fields():
        if electric_field.get_channel_ids()[0] not in trigger_channels:
            ## don't care about non-trigger channels regarding start time
            continue
        if len(electric_field.get_times()) <= 200:
            print(f"!!!!!!!!!!!!!! Warning: Electric field with only {len(electric_field.get_times())} samples found. !!!!!!!!!!!!!!")
            print("Event ID:", event.get_id(), "Station ID:", sim_station.get_id(),"ch",electric_field.get_channel_ids())
            print("E-field min/max:", electric_field.get_trace().min(), electric_field.get_trace().max())
            # dtime = np.linalg.norm(electric_field.get_position())/(3e8 * units.m / units.s)  # time it takes for light to travel the distance
            # electric_field.set_trace_start_time(dtime)
            continue
        times = electric_field.get_times()
        tstarts.append(times[0])
        tends.append(times[-1])
        pulse_times.append(times[np.argmax(electric_field.get_hilbert_envelope_mag())])
    print("trace tstarts (ns):", [f"{t/units.ns:.1f}" for t in tstarts])
    print("trace tends   (ns):", [f"{t/units.ns:.1f}" for t in tends])
    print("pulse times  (ns):", [f"{t/units.ns:.1f}" for t in pulse_times])
    if len(tstarts) == 0 or len(tends) == 0:
        tstart = 0 * units.ns
        tend = 0 * units.ns
    else:
        tstart = np.min(tstarts) - pad_before
        tend = np.max(tends) + pad_after
        pulse_time = np.min(pulse_times)
    # if tstart > pulse_time - 20 * units.ns:
    #     ## front padding too large, use the pulse time minus 20 ns
    #     print("\ttrace tstart (ns):", f"{tstart/units.ns:.1f}")
    #     print("\tpulse time  (ns):", f"{pulse_time/units.ns:.1f}")
    #     print( f"front padding [{pad_before/units.ns:.2f}] is too large, use the pulse time minus 20 ns")
    #     tstart = pulse_time - 20 * units.ns

    t_readout_window = det.get_number_of_samples(sim_station.get_id(), 0) / \
        det.get_sampling_frequency(sim_station.get_id(), 0)
    if tend - tstart < t_readout_window:
        tend = tstart + t_readout_window + pad_after * units.ns

    # assumes all efields have the same sampling rate
    n_samples = int((tend - tstart) * electric_field.get_sampling_rate())
    if n_samples % 2 != 0:
        n_samples += 1
    print("event:",event.get_id()," t_readout_window",t_readout_window/units.ns,"ns","tstart",tstart/units.ns,"tend",tend/units.ns)
    print(n_samples,"samples at",electric_field.get_sampling_rate()/units.GHz,"GHz")
    for electric_field in sim_station.get_electric_fields():
        readout = BaseTrace()
        readout.set_trace(np.zeros((3, n_samples)), electric_field.get_sampling_rate(), tstart)

        # if len(electric_field.get_trace()) > 100: ## assumes short traces are not useful
        # new_efield = readout + electric_field ##
        # readout.add_to_trace(new_efield)
        # try:
        readout.add_to_trace(electric_field,raise_error=False)
            # readout.add_to_trace(electric_field)
        # except:
        #     ## typically fail when efield has too few samples or outside readout window (surface channel)
        #     # print(f"!!!!!!!!!!!!!! Warning couldn't add_to_trace, use zero trace !!!!!!!!!!!!!!")
        #     # print("Event ID:", event.get_id(), "Station ID:", sim_station.get_id(),"ch",electric_field.get_channel_ids())
        #     # print("E-field shape",electric_field.get_trace().shape,"\nmin/max:", electric_field.get_trace().min(), electric_field.get_trace().max())
        #     pass
        electric_field.set_trace(readout.get_trace(), "same", tstart)



def split_events(event, det, trigger_channels,num_channels_per_event=4):
    """ Split an event with more than 4 channels into multiple events with 4 channels. """

    det.set_event(event)
    station = event.get_station()
    # if (len(det.get_channel_ids(station.get_id())) == len(trigger_channels) and
    #     np.all(det.get_channel_ids(station.get_id()) == trigger_channels)):
    #     return [event]
    if (len(det.get_channel_ids(station.get_id())) == num_channels_per_event and
        np.all(det.get_channel_ids(station.get_id()) == np.arange(num_channels_per_event))):
        return [event]

    # if len(det.get_channel_ids(station.get_id())) == len(trigger_channels):
    #     raise ValueError("Some thing unexpected happend. The event has only 4 channels but "
    #                      f"the channel ids {det.get_channel_ids(station.get_id())} do not "
    #                      f"match the trigger channels ({trigger_channels})")

    # Split the event into multiple events
    all_sim_channel_ids = np.array([efields.get_channel_ids()[0] for efields in station.get_sim_station().get_electric_fields()])
    string_sim_channel_ids = ["{}".format(ch_ids) for ch_ids in all_sim_channel_ids] ## mimic list of strings

    sorted_string_sim_channel_ids = sorted(string_sim_channel_ids)
    num_from_sorted_string = [ int(ch) for ch in sorted_string_sim_channel_ids ]
    argsort_from_sorted_num = np.argsort(num_from_sorted_string)
    print("argsort_from_num[:20]",argsort_from_sorted_num[:20])
    
    channel_to_index_map = {ch_id: index for ch_id, index in enumerate(argsort_from_sorted_num)}
    print("channel_to_index_map",channel_to_index_map)


    # sim_channel_ids = np.unique([efields.get_channel_ids() for efields in station.get_sim_station().get_electric_fields()])
    # channel_positions = np.array([det.get_relative_position(station.get_id(), sim_channel_id) for sim_channel_id in sim_channel_ids])
    
    ## testing channel ids and positions
    # for chid,chpos in zip(all_sim_channel_ids, channel_positions):
    #     print(f"Sim channel IDs\n{chid[0]} at {chpos}")

    # unique_xy_positions = np.unique(channel_positions[:, :2], axis=0)
    # n_batches = len(unique_xy_positions)
    n_batches = np.ceil( len(all_sim_channel_ids)/ num_channels_per_event ).astype(int) ## round up

    # sim_channel_ids_batches = [[] for _ in range(n_batches)]
    # for sim_channel_id, xy_position in zip(sim_channel_ids, channel_positions[:, :2]):
    #     idx = np.arange(n_batches)[np.all(unique_xy_positions == xy_position, axis=1)][0]
    #     sim_channel_ids_batches[idx].append(sim_channel_id)
    
    # sim_channel_ids_batches = np.array(all_sim_channel_ids[argsort_from_sorted_num]).reshape(n_batches, num_channels_per_event).tolist()
    sim_channel_ids_batches = [[] for _ in range(n_batches)]
    for i in range(n_batches):
        channels_in_batch = np.array([ch for ch in range(num_channels_per_event)]) + i*num_channels_per_event
        for ch in channels_in_batch:
            if ch < len(all_sim_channel_ids):
                sim_channel_ids_batches[i].append(all_sim_channel_ids[channel_to_index_map[ch]])

    events = []

    for ievent,sim_channel_ids_batch in enumerate(sim_channel_ids_batches):
        print("sim_channel_ids_batch",sim_channel_ids_batch)
        new_event = copy.deepcopy(event)
        new_event.set_id(ievent)

        # if len(sim_channel_ids_batch) != len(trigger_channels):
        #     raise ValueError("Some thing unexpected happend. The batch has not the same number of channels as the trigger channels "
        #                      f"sim_channel_ids_batch: {sim_channel_ids_batch}, trigger_channels: {trigger_channels}")

        new_sim_station = NuRadioReco.framework.sim_station.SimStation(station.get_id())  # set sim station id to 0
        new_sim_station.set_is_neutrino() # HACK: Since the sim. efields are always at the exact positions as the antenna(channels).
        new_station = NuRadioReco.framework.station.Station(station.get_id())
        new_station.set_sim_station(new_sim_station)
        new_event.set_station(new_station)  # overwrites existing station
        events.append(new_event)

        # sort the sim_channel_ids_batch by depth
        # depth = np.array([det.get_relative_position(station.get_id(), sim_channel_id)[2] for sim_channel_id in sim_channel_ids_batch])
        # sort = np.argsort(depth)

        # sorted_sim_channel_ids_batch = np.array(sim_channel_ids_batch)[sort]
        sorted_sim_channel_ids_batch = np.array(sim_channel_ids_batch)  # assume batch already sorted by ch-id 
        print("  sorted_sim_channel_ids_batch",sorted_sim_channel_ids_batch)
        # for sim_channel_id, new_id in zip(sorted_sim_channel_ids_batch, trigger_channels):
        for sim_channel_id, new_id in zip(sorted_sim_channel_ids_batch, np.arange(len(sorted_sim_channel_ids_batch))):  ## assume already sorted by ch-id 
            for efield in station.get_sim_station().get_electric_fields_for_channels([sim_channel_id]):
                efield_new = copy.deepcopy(efield)
                efield_new.set_channel_ids([new_id])
                new_sim_station.add_electric_field(efield_new)
                # print(f"  Adding sim efield channel {efield_new.get_channel_ids()[0]} at {efield_new.get_position()}")
                # print(f"    min/max: {efield_new.get_trace().min()}/{efield_new.get_trace().max()} with {(efield_new.get_trace()).shape} samples")


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
    num_channels_per_event = 24  # number of channels per event in the input file
    
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
    # efieldToVoltageConverter.begin(post_pulse_time=100 * units.ns, pre_pulse_time=100 * units.ns)
    efieldToVoltageConverter.begin(post_pulse_time=400 * units.ns, pre_pulse_time=200  * units.ns)

    efieldToVoltageConverterPerEfield = NuRadioReco.modules.efieldToVoltageConverterPerEfield.efieldToVoltageConverterPerEfield()

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

        for edx, event in enumerate(split_events(combined_event, dummy_detector_for_positions_only, trigger_channels,num_channels_per_event=num_channels_per_event)):
            dummy_detector_for_positions_only.set_event(event)
            pad_traces(event, det_rnog,trigger_channels=trigger_channels)

            shower = event.get_first_sim_shower()
            for sdx, station in enumerate(event.get_stations()):
                print("Printing triggers before setting pre_trigger_times:")
                for trigger in station.get_triggers().values():
                    print("trigger from station.get_triggers():", trigger)
                    trigger.set_pre_trigger_times(250 * units.ns)
                    station.set_trigger(trigger)
                sim_station = station.get_sim_station()

                # if (edx + sdx) % 100 == 0:
                if (edx + sdx) % 1 == 0:
                    print(f"Processing event: {event.get_id()} station {station.get_id()}")
                    print(f"Energy: {shower.get_parameter(shp.energy) / units.PeV} PeV, "
                        f"Zenith: {shower.get_parameter(shp.zenith) / units.deg}, "
                        f"Azimuth: {shower.get_parameter(shp.azimuth) / units.deg}")

                # Temporary sanity checks - to apply the correct noise and filter the event
                # can only have 4 channels with IDs [0, 1, 2, 3] (and they should be at the
                # correct depths)

                ## skip assertion for now due to change in expectation
                # assert np.all(dummy_detector_for_positions_only.get_channel_ids(station.get_id()) == trigger_channels), "Expected channels [0, 1, 2, 3]"
                # channel_depths = np.array([dummy_detector_for_positions_only.get_relative_position(
                #     station.get_id(), channel_id)[2] for channel_id in dummy_detector_for_positions_only.get_channel_ids(station.get_id())])
                # assert np.all(np.argsort(channel_depths) == trigger_channels), "Expected channels to be sorted by depth"

                if args.add_noise and args.noise_type == "data-driven":
                    detector_simulation_with_data_driven_noise(
                        event, station, det_rnog, trigger_channels=trigger_channels)
                else:
                    assert args.noise_type == "rayleigh", "Only 'rayleigh' and 'data-driven' noise is supported."
                    efieldToVoltageConverter.run(event, station, det_rnog, channel_ids=np.arange(num_channels_per_event))
                    efieldToVoltageConverterPerEfield.run(event, station, det_rnog)

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
            # print("running eventWriter with mode",mode)
            eventWriter.run(event, mode=mode)
