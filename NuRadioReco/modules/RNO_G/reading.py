import mattak.Dataset
import time
import numpy
import ROOT
import os.path
import read_mod
from read_mod import readRNOGData
import sys
import os
from scipy import interpolate
import astropy.time
import matplotlib
from matplotlib import pyplot
import NuRadioReco
import NuRadioReco.framework.station
import NuRadioReco.framework.channel
import NuRadioReco.framework.trigger
import NuRadioReco.modules.channelSignalReconstructor
signal_reconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
#confirm and print backend being used (pyroot or uproot)

for backend in ("pyroot","uproot"):
    print('Using backend: ', backend)

    #input data directory (path to directory with station numbers), station number, and run number. Confirm that path has a combined.root file and generate full file paths 
    data_str=input("Please enter path to data:  ")
    station_select=input("Please enter station number:  ")
    station_select=int(station_select)
    run_select=input("Please enter run number:  ")
    run_select=int(run_select)
    run_str="run{}".format(run_select)
    station_str="station{}".format(station_select)
    big_str= data_str + '/' + station_str + '/' + run_str
    file_exists=os.path.exists(big_str)
    print('Found the data file?', file_exists)
    file_exists=bool(file_exists)
    if not file_exists:
       print("selected run has no combined.root file")
       sys.exit()
    def list_full_paths(directory):
       return [os.path.join(directory, file) for file in os.listdir(directory)]
    list_files=list_full_paths(big_str)
    print("I found these files ", list_files) 

#execute the begin function from readRNOGData to open data file and form data set
    readRNOGData = read_mod.readRNOGData()
    readRNOGData.stations=station_select
    readRNOGData.runs=run_select
    readRNOGData.datas=data_str
    readRNOGData.begin(station_select, run_select, data_str, list_files)
    #begin function returns number of events, save it
    Nevents=readRNOGData.begin(station_select, run_select, data_str, list_files)
    
    if Nevents == None:
       print ("slected combined.root file is empty...?")
       #you should never get this error since we already checked
       sys.exit()
    print('Number of events in this file: ', Nevents)
      #if you want to do anything with the mattak syntax this gets the dataset so you can work with it here under the name d
    d=readRNOGData.getdataset()
    #Execute the run function. converts from mattak dataset to NuRadioReco Event, yields each event after one iteration 
    readRNOGData.run()
    for i_event, event in enumerate(readRNOGData.run()):
        print('processing new event')
        #you can now use all the NuRadioReco 'Event' features on event, like example below
        print('I just processed an event from run number', event.get_run_number())
    #start doing things with the data

