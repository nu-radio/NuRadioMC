import NuRadioReco.modules.io.coreas.readCoREASShower
import NuRadioReco.modules.io.eventWriter
from NuRadioReco.utilities import units

from NuRadioReco.framework.parameters import showerParameters as shP

from NuRadioReco.detector import generic_detector as detector

import numpy as np
import argparse
import datetime

# Parse eventfile as argument
parser = argparse.ArgumentParser(description='NuRadioSim file')
parser.add_argument('inputfilename', type=str, nargs='*',
                    help='path to NuRadioMC simulation result')

parser.add_argument('--detectordescription', type=str, nargs='?',
                    default='../examples/example_data/dummy_detector.json',
                    help='path to detectordescription')

args = parser.parse_args()

# initialize modules
readCoREASShower = NuRadioReco.modules.io.coreas.readCoREASShower.readCoREASShower()
readCoREASShower.begin(args.inputfilename)

eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
output_filename = "Full_CoREAS_shower.nur"
eventWriter.begin(output_filename)

for iE, event in enumerate(readCoREASShower.run()):
    print('Event {}'.format(event.get_id()))

    # initialize detector
    det = detector.GenericDetector(json_filename=args.detectordescription, default_station=102)
    det.update(datetime.datetime(2011, 11, 11, 11, 11))

    for station in event.get_stations():
        station.set_station_time(datetime.datetime(2011, 11, 11))

        sim_station = station.get_sim_station()
        position = sim_station.get_position()
        det_station_dict = {'station_id': station.get_id(),
                            'pos_easting': position[0],
                            'pos_northing': position[1],
                            'pos_altitude': position[2]}

        det.add_generic_station(det_station_dict)

    eventWriter.run(event, det)

nevents = eventWriter.end()
print("Finished processing, {} events".format(nevents))
