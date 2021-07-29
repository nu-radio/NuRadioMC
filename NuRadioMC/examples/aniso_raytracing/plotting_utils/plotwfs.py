import logging
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr

from scipy.signal import hilbert

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

#fig, ax = plt.subplots(nrows = 4, ncols = 4)

#a = np.zeros((4,4))

fig, ax = plt.subplots(nrows = 2, ncols = 4, figsize=[16,9])

maxtimes = np.zeros(16)

def set_plt(ax, times, trace, channel_id):
    ax.plot(times, trace, label = 'channel '+str(channel_id))
    ax.legend()
    ax.grid(True)
    ax.xaxis.set_minor_locator(tkr.MultipleLocator(50))

for event in eventReader.run():
    for station in event.get_stations():
        station_id = station.get_id()
        for channel in station.iter_channels():
            channel_id = channel.get_id()

            # get time trace and times of bins
            #trace = np.abs(hilbert(channel.get_trace()))
            #trace = channel.get_trace()/np.max(np.abs(channel.get_trace()))
            trace = channel.get_trace()
            times = channel.get_times()
            
            maxtimes[channel_id] = times[np.argwhere(trace == np.max(trace))[0]]

            if channel_id == 0 or channel_id == 8:
                set_plt(ax[0, 0], times, trace, channel_id)
            elif channel_id == 1 or channel_id == 9:
                set_plt(ax[0, 1], times, trace, channel_id)
            elif channel_id == 2 or channel_id == 10:
                set_plt(ax[0, 2], times, trace, channel_id) 
            elif channel_id == 3 or channel_id == 11:
                set_plt(ax[0, 3], times, trace, channel_id) 
            elif channel_id == 4 or channel_id == 12:
                set_plt(ax[1, 0], times, trace, channel_id) 
            elif channel_id == 5 or channel_id == 13:
                set_plt(ax[1, 1], times, trace, channel_id)
            elif channel_id == 6 or channel_id == 14:
                set_plt(ax[1, 2], times, trace, channel_id)
            elif channel_id == 7 or channel_id == 15:
                set_plt(ax[1, 3], times, trace, channel_id)

            if channel_id == 15:
                for i in range(8):
                    print('ch '+str(8+i)+' and '+str(i), maxtimes[8+i] - maxtimes[i])

fig.suptitle('deep -> ARA02')
fig.tight_layout()
#plt.show()
plt.savefig('../waveforms/deep_to_ARA02_test.pdf')
