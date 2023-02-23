import NuRadioReco.framework.event
from NuRadioReco.modules.base.module import register_run
import NuRadioReco.framework.station
import NuRadioReco.framework.channel
import NuRadioReco.framework.trigger
import NuRadioReco.modules.channelSignalReconstructor
signal_reconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
from dataclasses import dataclass
import numpy as np
import datetime
import mattak.Dataset
import sys
import os
import logging
import time
from scipy import interpolate
import six
from collections import OrderedDict
import astropy.time
#import uproot
from collections import OrderedDict

# default uproot readout is awkward arrays, but requires awkward package installed. RNOG data format does not require this. Use numpy instead.


class readRNOGData:

   def __init__(self):
      self.__id_current_event = None
      self.__t = None
      self.__sampling_rate = 3.2
      self._iterator_data = None
      self._iterator_header = None
      self._data_treename = "waveforms"
      self._header_treename = "header"
      self.n_events = None
      self.input_files = []

   def begin(self,stations,runs, datas, input_files, input_files_header=None):
       #stations is a station integer number, runs is a run integer number, datas is a string with the path to directory with stations inside, input files is the full path to the selected run. Don't need to specify header, but it's here if you want to set it manually for some reason
       if isinstance(input_files, six.string_types):
           input_files = [input_files]
       if isinstance(input_files_header, six.string_types):
           input_files_header = [input_files_header]
       if input_files_header is None:
          input_files_header = input_files
       self.input_files = input_files
       self.input_files_header = input_files_header
       self.stations = stations
       self.runs= runs
       self.datas=datas
       #this is the important step, take selected run and make a data set structure in mattak. This is the dataset that we we loop over and process in the run function
       self.d = mattak.Dataset.Dataset(station=stations, run=runs, data_dir=datas)
       return self.d.N()
  #function that returns the dataset 
   def getdataset(self):
      return self.d


   def run(self, channels=np.arange(24),event_numbers=None, run_numbers=None, cut_string=None):
      #start doing things with the data
      #get header info
      self.__t = time.time()
      #loop over events 
      for event in range(self.d.N()):
         print("Processing event number ",event, " out of total ", self.d.N())
         self.d.setEntries(event)
         print(self.d.eventInfo())
         evt = NuRadioReco.framework.event.Event(self.runs, event)
         #print('I just processed an event from run number', evt.get_run_number())
         station = NuRadioReco.framework.station.Station(self.stations)
         trigger_key='trigger_info.'+self.d.eventInfo().triggerType
         if self.d.eventInfo().triggerType =='UNKNOWN':
             print('there is no trigger type recorded for this event, not making a trigger object in NuRadioReco')
             print(trigger_key)
         else:
            trigger= NuRadioReco.framework.trigger.Trigger(trigger_key.split('.')[-1])
            trigger.set_triggered()
            trigger.set_trigger_time(self.d.eventInfo().triggerTime)
            print('can I access the trigger?', trigger.has_triggered())
            print('what is the actual trigger key?' , trigger_key)
         for chan in range(24):
              #fill in radiant data/channel specifics
              channel = NuRadioReco.framework.channel.Channel(chan)    
         yield evt

