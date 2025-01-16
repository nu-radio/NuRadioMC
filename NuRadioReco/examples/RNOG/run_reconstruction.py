import NuRadioReco.modules.io.RNO_G.readRNOGDataMattak
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.sphericalWaveFitter
import NuRadioReco.modules.channelAddCableDelay

from NuRadioReco.detector import detector
from NuRadioReco.utilities import units

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import os


"""
An example to show how to read RNO-G data and how to perform simple reconstructions.
The data used is pulser data and a simple (brute force, not optimized) spherical
wave reconstruction is performed to obtain the pulser position.
"""

""" Initiazize modules needed"""

channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
sphericalWaveFitter = NuRadioReco.modules.sphericalWaveFitter.sphericalWaveFitter()
channelAddCableDelay = NuRadioReco.modules.channelAddCableDelay.channelAddCableDelay()

use_channels = [10, 9, 0, 3, 21]
sphericalWaveFitter.begin(channel_ids = use_channels)

""" Specify the detector. """

json_file_path = os.path.dirname(__file__) + "/../../detector/RNO_G/RNO_season_2021.json"

det = detector.Detector(json_filename=json_file_path)
det.update(datetime.datetime(2022, 10, 1))

""" Get positions for the pulsers from the detector file as starting positions for the fit """

station_id = 21
pulser_id = 3  #Helper string C
rel_pulser_position = det.get_relative_position(station_id, pulser_id, mode = 'device')

plots = True
""" read in data """
list_of_root_files = ['pulser_data_21.root']


readRNOGData = NuRadioReco.modules.io.RNO_G.readRNOGDataMattak.readRNOGData()
readRNOGData.begin(list_of_root_files)

for i_event, event in enumerate(readRNOGData.run()):
	print("reconstruction for event", i_event)
	station_id = event.get_station_ids()[0]
	station = event.get_station(station_id)
	channelAddCableDelay.run(event, station, det, mode = 'subtract')
	channelBandPassFilter.run(event, station, det, passband = [5*units.MHz, 450*units.MHz])
	sphericalWaveFitter.run(event, station, det, start_pulser_position = rel_pulser_position, n_index = 1.78, debug =True)

	if plots:
		fig, axs = plt.subplots(3, 2)
		for ax, channel in zip(axs.flatten(), station.iter_channels()):
			if channel.get_id() in use_channels:
				ax.plot(channel.get_trace())
				ax.title.set_text("channel id {}".format(channel.get_id()))
				ax.grid()

		fig.tight_layout()
		fig.savefig("trace_{}.pdf".format(i_event))
