import NuRadioReco.modules.io.coreas.readCoREASShower
import NuRadioReco.modules.io.eventWriter
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.eventTypeIdentifier
from NuRadioReco.detector import generic_detector as detector
import argparse

# Parse eventfile as argument
parser = argparse.ArgumentParser(description='NuRadioSim file')
parser.add_argument('inputfilename', type=str, nargs='*',
                    default=['example_data/example_event.h5'],
                    help='path to NuRadioMC simulation result')

parser.add_argument('-o', '--output_filename', type=str, nargs='?',
                    default='Full_CoREAS_shower.nur',
                    help='output file name')

parser.add_argument('--detectordescription', type=str, nargs='?',
                    default='../examples/example_data/dummy_detector.json',
                    help='path to detectordescription')

parser.add_argument('--set_run_number', dest='set_run_number', action='store_true',
                    help='If set, the run number and event id will be set to an increasing value.')


args = parser.parse_args()

# initialize modules
det = detector.GenericDetector(json_filename=args.detectordescription, default_station=102, default_channel=0)
readCoREASShower = NuRadioReco.modules.io.coreas.readCoREASShower.readCoREASShower()
readCoREASShower.begin(args.inputfilename, det, set_ascending_run_and_event_number=args.set_run_number)
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin()
eventTypeIdentifier = NuRadioReco.modules.eventTypeIdentifier.eventTypeIdentifier()
eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
eventWriter.begin(args.output_filename)

for event, gen_det in readCoREASShower.run():
    print('Event {} {}'.format(event.get_run_number(), event.get_id()))
    for station in event.get_stations():
        eventTypeIdentifier.run(event, station, 'forced', 'cosmic_ray')
        efieldToVoltageConverter.run(event, station, gen_det)
    eventWriter.run(event, gen_det)

nevents = eventWriter.end()
print("Finished processing, {} events".format(nevents))
