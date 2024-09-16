from logging import debug
from re import I
import NuRadioReco.modules.io.coreas.readFEARIEShower
import NuRadioReco.modules.io.eventWriter
import NuRadioReco.modules.efieldRadioInterferometricReconstruction
import NuRadioReco.modules.eventTypeIdentifier
import NuRadioReco.modules.electricFieldBandPassFilter

from NuRadioReco.detector import generic_detector as detector
import argparse
from NuRadioReco.utilities import units

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

readFEARIEShower = NuRadioReco.modules.io.coreas.readFEARIEShower.readFEARIEShower()
readFEARIEShower.begin(args.inputfilename)

for event in readFEARIEShower.run():
    print('Event {} {}'.format(event.get_run_number(), event.get_id()))
