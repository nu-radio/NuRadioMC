import NuRadioReco.framework.event
import NuRadioReco.framework.station
import NuRadioReco.framework.channel
import ROOT
import numpy as np
from NuRadioReco.utilities import units
import sys
import os
import logging
import time
from scipy import interpolate

logger = logging.getLogger("readARAData")
logging.basicConfig()

sys.path.append(os.path.expandvars('${ARA_UTIL_INSTALL_DIR}/lib'))

from ctypes import *
libc = CDLL("libAraEvent.so")
libc = CDLL("libAraConfig.so")
libc = CDLL("libAraCorrelator.so")
libc = CDLL("libAraDisplay.so")
libc = CDLL("libAraKvp.so")
libc = CDLL("libRootFftwWrapper.so")

class readARAData:

    """
    This is the AraReader. Prints out event waveform, unixtime and timestamp.
    """
    def begin(self,input_file):
        self.__id_current_event = -1
        self.__t = time.time()
        
        self.f = ROOT.TFile.Open(input_file)
        self.raw_ptr=ROOT.RawAtriStationEvent()
        self.data_tree = self.f.Get("eventTree")
        self.data_tree.SetBranchAddress("event",self.raw_ptr)      
        self.n_events = self.data_tree.GetEntries()

        return self.n_events

    def run(self):
        """

        """
        while True:
            self.__id_current_event += 1
            if(self.__id_current_event >= self.n_events):
                # all events processed
                break
            if(self.__id_current_event % 1000 == 0):
                progress = 1. * self.__id_current_event / self.n_events
                eta = 0
                if(self.__id_current_event > 0):
                    eta = (time.time() - self.__t) / self.__id_current_event * (self.n_events - self.__id_current_event) / 60.
                logger.warning("reading in event {}/{} ({:.0f}%) ETA: {:.1f} minutes".format(self.__id_current_event, self.n_events,100 * progress, eta))

            self.data_tree.GetEntry(self.__id_current_event)
            run_number = self.data_tree.run
            evt_number = self.raw_ptr.eventNumber
            station_id = bytearray(self.raw_ptr.stationId)[0]
            print run_number,evt_number,station_id
            evt = NuRadioReco.framework.event.Event(run_number, evt_number)
            station = NuRadioReco.framework.station.Station(station_id)
            real_ptr=ROOT.UsefulAtriStationEvent(self.raw_ptr,ROOT.AraCalType.kLatestCalib)
            for iCh in xrange(16):
            #ARA always has 16 channels
                channel =NuRadioReco.framework.channel.Channel(iCh)
                
                graph_waveform = real_ptr.getGraphFromRFChan(iCh)
                times = np.array(graph_waveform.GetX())*units.ns
                voltage = np.array(graph_waveform.GetY())*units.mV

                #interpolate to get equal sampling between data points
                f_interpolate=interpolate.interp1d(times, voltage)
                times_new = np.arange(times[0], times[-1], 0.625)
                voltage_new=f_interpolate(times_new)
                
                if voltage_new.shape[0]%2 !=0:
                    voltage_new=voltage_new[:-1]
                sampling_rate = times_new[1]-times_new[0]
                #logger.warning("ARA sampling frequency not constant. Need to add the interpolating function here!!!")
                channel.set_trace(voltage_new,sampling_rate)
                station.add_channel(channel)
                evt.set_station(station)
            yield evt
    def end(self):
        pass
