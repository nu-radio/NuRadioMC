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
args = parser.parse_args()

# initialize modules
eventReader = NuRadioReco.modules.io.eventReader.eventReader()
eventReader.begin(args.inputfilename)

fig, ax = plt.subplots(nrows = 4, ncols = 4)

a = np.zeros((4,4))

for event in eventReader.run():
    for station in event.get_stations():
        station_id = station.get_id()
        for channel in station.iter_channels():
            channel_id = channel.get_id()

            # get time trace and times of bins
            trace = channel.get_trace()
            times = channel.get_times()
            
            #print(channel.get_ant_type())            
            
            j = channel_id % 4
            i = None
            if channel_id < 4:
                i = 0
            elif channel_id < 8:
                i = 1
            elif channel_id < 12:
                i = 2
            else:
                i = 3
                
            #a[i,j] = channel_id
            ax[i, j].plot(times, trace)
            ax[i,j].set_title('channel ' + str(channel_id))

#print(a)
plt.tight_layout()
plt.show()
