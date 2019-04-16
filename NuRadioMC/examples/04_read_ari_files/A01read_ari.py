import logging
import numpy as np
import argparse
import matplotlib.pyplot as plt

from NuRadioReco.utilities import units
import NuRadioReco.detector.detector as detector
import NuRadioReco.modules.io.eventReader

from NuRadioReco.framework.parameters import stationParameters as stnp

logging.basicConfig(level=logging.INFO)

# Parse eventfile as argument
parser = argparse.ArgumentParser(description='NuRadioSim file')
parser.add_argument('inputfilename', type=str,
                    help='path to NuRadioMC simulation result')
parser.add_argument('detectordescription', type=str,
                    help='path to detectordescription')
args = parser.parse_args()


# read in detector positions (this is a dummy detector)
det = detector.Detector(json_filename=args.detectordescription)

# initialize modules
eventReader = NuRadioReco.modules.io.eventReader.eventReader()
eventReader.begin(args.inputfilename)

for event in eventReader.run():
    for station in event.get_stations():
        station_id = station.get_id()
        for channel in station.iter_channels():
            channel_id = channel.get_id()
            
            # get time trace and times of bins
            trace = channel.get_trace()
            times = channel.get_times()
            
            # or get the frequency spetrum instead
            spectrum = channel.get_frequency_spectrum()
            frequencies = channel.get_frequencies()
            
            # obtain the position of the channel/antenna from the detector description
            position = det.get_relative_position(station_id, channel_id)
            
            
            
