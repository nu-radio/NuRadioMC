import NuRadioReco
import matplotlib.pyplot as plt
from NuRadioReco.modules.io.rno_g.readRNOGData import readRNOGData
import pandas as pd
import numpy as np
from NuRadioReco.utilities import units
from NuRadioReco.modules import channelBandPassFilter
from NuRadioReco.detector import detector
import datetime
from NuRadioReco.modules import sphericalWaveFitter
from NuRadioReco.modules import channelAddCableDelay

""" An example to show how to read RNO-G data and how to perform simple reconstructions. The data used is pulser data and a simple (brute force, not optimized) spherical wave reconstruction is performed to obtain the pulser position """""

""" Initiazize modules needed"""

channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
sphericalWaveFitter = NuRadioReco.modules.sphericalWaveFitter.sphericalWaveFitter()
channelAddCableDelay = NuRadioReco.modules.channelAddCableDelay.channelAddCableDelay()

use_channels = [10, 9, 0, 3, 21]
sphericalWaveFitter.begin(channel_ids = use_channels)

""" Specify the detector. """

det = detector.Detector(json_filename = "../../../detector/RNO_G/RNO_season_2021.json")
det.update(datetime.datetime(2022, 10, 1))

""" Get positions for the pulsers from the detector file as starting positions for the fit """

station_id = 21
pulser_id = 3  #Helper string C
rel_pulser_position = det.get_relative_position(station_id, pulser_id, mode = 'device')

plots = True
""" read in data """
list_of_root_files = ['pulser_data_21.root']


readRNOGData = NuRadioReco.modules.io.rno_g.readRNOGData.readRNOGData()
readRNOGData.begin(list_of_root_files)

for i_event, event in enumerate(readRNOGData.run()):
	print("reconstruction for event", i_event)
	station_id = event.get_station_ids()[0]
	station = event.get_station(station_id)
	channelAddCableDelay.run(event, station, det, mode = 'subtract')
	channelBandPassFilter.run(event, station, det, passband = [5*units.MHz, 450*units.MHz])
	sphericalWaveFitter.run(event, station, det, start_pulser_position = rel_pulser_position, n_index = 1.78, debug =True)

	if plots:
		fig = plt.figure()
		i = 1
		for channel in station.iter_channels():
			if channel.get_id() in use_channels:
				ax = fig.add_subplot(3, 2, i)
				ax.plot(channel.get_trace())
				ax.title.set_text("channel id {}".format(channel.get_id()))
				ax.grid()
				i+= 1

		fig.tight_layout()
		fig.savefig("trace_{}.pdf".format(i_event))
    



                    
