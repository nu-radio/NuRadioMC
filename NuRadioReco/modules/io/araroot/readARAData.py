import NuRadioReco.framework.event
from NuRadioReco.modules.base.module import register_run
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
import ctypes


sys.path.append(os.path.expandvars('${ARA_UTIL_INSTALL_DIR}/lib'))

libc = ctypes.CDLL("libAraEvent.so")
libc = ctypes.CDLL("libAraConfig.so")
libc = ctypes.CDLL("libAraCorrelator.so")
libc = ctypes.CDLL("libAraDisplay.so")
libc = ctypes.CDLL("libAraKvp.so")
libc = ctypes.CDLL("libRootFftwWrapper.so")


class readARAData:

    """
    This is the AraReader. Reads ARA data in the ARARoot format.


    """
    def __init__(self):
        self.logger = logging.getLogger("NuRadioReco.readARAData")
        self.__id_current_event = None
        self.__t = None
        self.f = None
        self.raw_ptr = None
        self.data_tree = None
        self.n_events = None

    def begin(self, input_file):

        """
        Begin function of the ARA reader

        Parameters
        ----------
        input_file: string
            path to file to read

        """

        self.__id_current_event = -1
        self.__t = time.time()

        self.f = ROOT.TFile.Open(input_file)
        self.raw_ptr = ROOT.RawAtriStationEvent()
        self.data_tree = self.f.Get("eventTree")
        self.data_tree.SetBranchAddress("event", self.raw_ptr)
        self.n_events = self.data_tree.GetEntries()

        return self.n_events

    @register_run()
    def run(self, n_channels=16, sampling=0.625 * units.ns):
        """
        Run function of the ARA reader

        Parameters
        ----------
        n_channels: int
            number of ARA channels to loop over, default 16

        sampling: float (units time)
            default sampling of ARA detector

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
                self.logger.warning("reading in event {}/{} ({:.0f}%) ETA: {:.1f} minutes".format(self.__id_current_event, self.n_events, 100 * progress, eta))

            self.data_tree.GetEntry(self.__id_current_event)
            run_number = self.data_tree.run
            evt_number = self.raw_ptr.eventNumber
            station_id = bytearray(self.raw_ptr.stationId)[0]
            self.logger.info("Reading Run: {0}, Event {1}, Station {2}".format(run_number, evt_number, station_id))

            evt = NuRadioReco.framework.event.Event(run_number, evt_number)
            station = NuRadioReco.framework.station.Station(station_id)
            real_ptr = ROOT.UsefulAtriStationEvent(self.raw_ptr, ROOT.AraCalType.kLatestCalib)

            # Loop over all channels in data
            for iCh in range(n_channels):

                channel = NuRadioReco.framework.channel.Channel(iCh)

                # Get data from array via graph method
                graph_waveform = real_ptr.getGraphFromRFChan(iCh)
                times = np.array(graph_waveform.GetX()) * units.ns
                voltage = np.array(graph_waveform.GetY()) * units.mV

                # interpolate to get equal sampling between data points
                f_interpolate = interpolate.interp1d(times, voltage)
                times_new = np.arange(times[0], times[-1], sampling)
                voltage_new = f_interpolate(times_new)

                if voltage_new.shape[0] % 2 != 0:
                    voltage_new = voltage_new[:-1]
                sampling_rate = times_new[1] - times_new[0]

                channel.set_trace(voltage_new, sampling_rate)
                station.add_channel(channel)
                evt.set_station(station)
            yield evt

    def end(self):
        pass
