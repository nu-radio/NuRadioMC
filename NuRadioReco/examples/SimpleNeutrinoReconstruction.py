import logging
import numpy as np
import argparse
import matplotlib.pyplot as plt
import datetime

from NuRadioReco.utilities import units
import NuRadioReco.detector.detector as detector
import NuRadioReco.modules.io.eventReader
import NuRadioReco.modules.io.eventWriter
import NuRadioReco.modules.channelSignalReconstructor

from NuRadioReco.framework.parameters import channelParameters as chp


# Logging level
from NuRadioReco.modules.base import module
logger = module.setup_logger(level=logging.WARNING)

# Parse eventfile as argument
parser = argparse.ArgumentParser(description='NuRadioSim file')
parser.add_argument('inputfilename', type=str,
                    help='path to NuRadioMC simulation result')
parser.add_argument('detectordescription', type=str,
                    help='path to detectordescription')
args = parser.parse_args()


# read in detector positions (this is a dummy detector)
det = detector.Detector(json_filename=args.detectordescription)
det.update(datetime.datetime(2018, 10, 1))

# initialize modules
eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
eventReader = NuRadioReco.modules.io.eventReader.eventReader()
channelSignalReconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()

# Name outputfile
output_filename = "Simple_reconstruction_results.nur"
eventReader.begin(args.inputfilename)
eventWriter.begin(output_filename, max_file_size=1000)

i_events_saved = 0
for iE, event in enumerate(eventReader.run()):
    logger.info("Event ID {}".format(event.get_id()))
    for st, station in enumerate(event.get_stations()):
        channelSignalReconstructor.run(event, station, det)
        for channel in station.iter_channels():
            signal_to_noise = channel.get_parameter(chp.SNR)['peak_2_peak_amplitude']
            print('Event{}, Station {}, Channel {}, SNR = {:.2f}'.format(event.get_id(), station.get_id(), channel.get_id(), signal_to_noise))
    eventWriter.run(event)
    i_events_saved += 1
    if((i_events_saved % 100) == 0):
        print("saving event {}".format(i_events_saved))

