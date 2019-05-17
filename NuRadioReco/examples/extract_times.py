################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
######																			      ##########
###### This script copies data trees from snowflake, creates/organizes plots, and ... ##########
######																			      ##########
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################

import os
import numpy as np
import subprocess
from optparse import OptionParser
from matplotlib import cm
import math
import matplotlib.pyplot  as plt
from scipy import signal
from scipy.ndimage.filters import maximum_filter
from numpy.random import randn
import datetime
import logging
import time
from NuRadioReco.detector import detector
from NuRadioReco.utilities import units
from NuRadioReco.modules.io.snowshovel import readARIANNADataCalib as CreadARIANNAData
from NuRadioReco.modules import channelResampler as CchannelResampler
from NuRadioReco.modules.ARIANNA import hardwareResponseIncorporator as ChardwareResponseIncorporator
from NuRadioReco.modules import correlationDirectionFitter as CcorrelationDirectionFitter
import NuRadioReco.modules.voltageToEfieldConverterPerChannel
import NuRadioReco.modules.channelSignalReconstructor
import NuRadioReco.modules.channelBandPassFilter
import matplotlib.pyplot as plt



readARIANNAData = CreadARIANNAData.readARIANNAData()
channelResampler = CchannelResampler.channelResampler()
channelResampler.begin(debug=False)
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
hardwareResponseIncorporator = ChardwareResponseIncorporator.hardwareResponseIncorporator()
hardwareResponseIncorporator.begin(debug=False)
correlationDirectionFitter = CcorrelationDirectionFitter.correlationDirectionFitter()
voltageToEfieldConverterPerChannel = NuRadioReco.modules.voltageToEfieldConverterPerChannel.voltageToEfieldConverterPerChannel()
channelSignalReconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
voltageToEfieldConverterPerChannel.begin()



def printHeaderDetailsPerEvent(path,file):
	rootFile = path + file

	#tfile = ROOT.TFile(rootFile)
	#RawTree = tfile.Get("RawTree")
	ConsecutiveEntryNum = 0
	n_events = readARIANNAData.begin([rootFile])
	print n_events
	event_count = 0
	for evt in readARIANNAData.run():
		for station_object in evt.get_stations():
			status = 'file: ' + rootFile + ' event: ' + str(event_count) + ' utc_time: ' + str(station_object.get_station_time()) + ' Thermal? ' + str(station_object.has_triggered())
			print status
			time = str(station_object.get_station_time())[11:]
			event_count += 1

def main():
	printHeaderDetailsPerEvent('/home/geoffrey/ARIANNA/Jan8Spice/','CalTree.RawTree.SnEvtsM0002F7F2E7B9r00212s00084.root')

if __name__== "__main__":
	main()
