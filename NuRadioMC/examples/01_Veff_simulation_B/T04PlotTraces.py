import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import h5py
import glob
from scipy import interpolate
import json
import os
import sys

# Setup logging
from NuRadioReco.utilities.logging import setup_logger
logger = setup_logger(name="")

import NuRadioReco.modules.io.eventReader
from NuRadioReco.utilities import units

if __name__ == "__main__":

    path = sys.argv[1]
    trigger_channels=[8,9,2,3]
    eventReader = NuRadioReco.modules.io.eventReader.eventReader()
    eventReader.begin(path, read_detector=False)
    for evt in eventReader.run():
        for stn in evt.get_stations():
            fig, ax = plt.subplots(4, 2, sharey=True, sharex=True, figsize=(10, 10))
            ax = ax.flatten()
            for i, channel in enumerate(stn.iter_channels()):
            # for i, ich in enumerate(trigger_channels):
                # channel = stn.get_channel(ich)
                ax[i].plot(channel.get_times()/units.ns, channel.get_trace()/units.mV)
                if channel.get_id() in trigger_channels:
                    ax[i].set_title(f'trigger channel id = {channel.get_id()}')
                else:
                    ax[i].set_title(f'non-trigger channel id = {channel.get_id()}')
            ax[6].set_xlabel('time [ns]')
            ax[7].set_xlabel('time [ns]')
            ax[0].set_ylabel('voltage [mV]')
            ax[2].set_ylabel('voltage [mV]')
            ax[4].set_ylabel('voltage [mV]')
            ax[6].set_ylabel('voltage [mV]')

            fig.tight_layout()
            plt.show()

        