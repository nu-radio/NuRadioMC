import NuRadioReco.modules.io.coreas.readCoREAS
import NuRadioReco.modules.io.eventWriter
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.eventTypeIdentifier
import NuRadioReco.framework.event
import NuRadioReco.framework.station
from NuRadioReco.modules.io.coreas import coreas
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.detector import generic_detector as detector
import argparse


def event_builder_auger(coreas_evt, det=None):
    """
    Converts event structure from `coreas_evt` to a Auger-style event structure
    (Every observer - CoREAS Efield - gets its own station)

    This replicate the old readCoREASShower event structure.

    Write your own event_builder function to replicate, i.e., the LOFAR/SKA
    event structure.

    The readCoREAS module returns an event with a single station (id 0) whose
    SimStation holds every CoREAS observer as an ElectricField. This helper turns
    that into one Station per observer and, if a GenericDetector is passed, adds a
    station at each observer position on the fly with channel ids 0 and 1
    (as readCoREASShower used to do).

    Parameters
    ----------
    coreas_evt : NuRadioReco.framework.event.Event
        Event as yielded by readCoREAS.run(), with all observers in station 0.
    det : GenericDetector, optional
        If given, a generic station is added for every observer position.

    Returns
    -------
    evt : NuRadioReco.framework.event.Event
        Event with one Station per observer.
    """
    evt = NuRadioReco.framework.event.Event(coreas_evt.get_run_number(), coreas_evt.get_id())
    evt.set_event_time(coreas_evt.get_event_time())

    sim_shower = coreas.create_sim_shower(coreas_evt)
    evt.add_sim_shower(sim_shower)

    corsika_efields = coreas_evt.get_station(0).get_sim_station().get_electric_fields()
    for station_id, corsika_efield in enumerate(corsika_efields):
        station = NuRadioReco.framework.station.Station(station_id)
        sim_station = coreas.create_sim_station(station_id, coreas_evt)

        if det is None:
            channel_ids = [0, 1]
        elif det.has_station(station_id):
            channel_ids = det.get_channel_ids(station_id)
        else:
            channel_ids = det.get_channel_ids(det.get_reference_station_ids()[0])

        efield_trace = corsika_efield.get_trace()
        efield_times = corsika_efield.get_times()

        # adds electric field with relative position 0, 0, 0 (relative to station)
        coreas.add_electric_field_to_sim_station(
            sim_station, channel_ids, efield_trace, efield_times[0],
            sim_shower.get_parameter(shp.zenith), sim_shower.get_parameter(shp.azimuth),
            corsika_efield.get_sampling_rate())

        station.set_sim_station(sim_station)
        evt.set_station(station)

        if det is not None:
            efield_pos = corsika_efield.get_position()
            if not det.has_station(station_id):
                det.add_generic_station({
                    'station_id': station_id,
                    'pos_easting': efield_pos[0],
                    'pos_northing': efield_pos[1],
                    'pos_altitude': efield_pos[2],
                    'reference_station': det.get_reference_station_ids()[0]
                })
            else:
                det.add_station_properties_for_event({
                    'pos_easting': efield_pos[0],
                    'pos_northing': efield_pos[1],
                    'pos_altitude': efield_pos[2]
                }, station_id, evt.get_run_number(), evt.get_id())

    if det is not None:
        det.set_event(evt.get_run_number(), evt.get_id())

    return evt


if __name__ == '__main__':
    # Parse eventfile as argument
    parser = argparse.ArgumentParser(description='NuRadioSim file')
    parser.add_argument('inputfilename', type=str, nargs='*',
                        default=['example_data/example_event.h5'],
                        help='path to NuRadioMC simulation result')

    parser.add_argument('-o', '--output_filename', type=str, nargs='?',
                        default='Full_CoREAS_shower.nur',
                        help='output file name')

    parser.add_argument('--detectordescription', type=str, nargs='?',
                        default='../examples/example_data/dual_polerized_antenna_station.json',
                        help='path to detectordescription')

    parser.add_argument('--set_run_number', dest='set_run_number', action='store_true',
                        help='If set, the run number and event id will be set to an increasing value.')

    args = parser.parse_args()

    # initialize modules
    det = detector.GenericDetector(json_filename=args.detectordescription)
    readCoREAS = NuRadioReco.modules.io.coreas.readCoREAS.readCoREAS()
    readCoREAS.begin(args.inputfilename, set_ascending_run_and_event_number=args.set_run_number)
    efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
    efieldToVoltageConverter.begin()
    eventTypeIdentifier = NuRadioReco.modules.eventTypeIdentifier.eventTypeIdentifier()
    eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
    eventWriter.begin(args.output_filename)

    for coreas_event in readCoREAS.run():
        event = event_builder_auger(coreas_event, det)
        print('Event {} {}'.format(event.get_run_number(), event.get_id()))
        for station in event.get_stations():
            eventTypeIdentifier.run(event, station, 'forced', 'cosmic_ray')
            efieldToVoltageConverter.run(event, station, det)
        eventWriter.run(event, det)

    nevents = eventWriter.end()
    print("Finished processing, {} events".format(nevents))
