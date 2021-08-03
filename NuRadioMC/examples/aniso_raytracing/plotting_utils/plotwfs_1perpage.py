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
parser.add_argument('outputfilename', type=str,
                    help='path to plotwf.py output result')
args = parser.parse_args()

# initialize modules
eventReader = NuRadioReco.modules.io.eventReader.eventReader()
eventReader.begin(args.inputfilename)

tmp = [plt.subplots(nrows=1, ncols=1, figsize=[16,9]) for i in range(8)]
figs = [tmp[i][0] for i in range(8)]
axs = [tmp[i][1] for i in range(8)]

def set_plt(ax, times, trace, channel_id):
    ax.plot(times, trace, label = 'channel '+str(channel_id))
    ax.legend()
    ax.grid(True)
    ax.xaxis.set_minor_locator(tkr.MultipleLocator(50))

def make_plt(fig, flname):
    fig.tight_layout()
    fig.savefig(flname)

for event in eventReader.run():
    for station in event.get_stations():
        station_id = station.get_id()
        for channel in station.iter_channels():
            channel_id = channel.get_id()

            trace = channel.get_trace()
            times = channel.get_times()

            set_plt(axs[channel_id % 8], times, trace, channel_id)
           
for i in range(8):
    make_plt(figs[i], args.outputfilename+str(i)+'-'+str(i+ 8)+'.pdf')             
