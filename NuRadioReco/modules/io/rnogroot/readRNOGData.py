import NuRadioReco.framework.event
from NuRadioReco.modules.base.module import register_run
import NuRadioReco.framework.station
import NuRadioReco.framework.channel

import uproot
# default uproot readout is awkward arrays, but requires awkward package installed. RNOG data format does not require this. Use numpy instead.
uproot.default_library = "np"

import numpy as np
from NuRadioReco.utilities import units
import sys
import os
import logging
import time
from scipy import interpolate
import six

class readRNOGData:

    """
    This is the data reader for RNO-G. Reads RNO-G data from ROOT format using uproot


    """
    def __init__(self):
        self.logger = logging.getLogger("NuRadioReco.readRNOGdata")
        self.__id_current_event = None
        self.__t = None
        self.f = None
        self.raw_ptr = None
        self.data_tree = None
        self.n_events = None

    def begin(self, input_files):

        """
        Begin function of the RNO-G reader

        Parameters
        ----------
        input_file: string
            path to file to read

        """

        self.__id_current_event = -1
        self.__t = time.time()

        if isinstance(input_files, six.string_types):
            input_files = [input_files]

        self.input_files = input_files
        self.input_tree = "waveforms"

        self.n_events = 0
        # get the total number of events of all input files
        for the_f in input_files:
            the_f = uproot.open(input_files[0])
            the_data_tree = the_f[self.input_tree]
            self.n_events += the_data_tree.num_entries

        return self.n_events

    @register_run()
    def run(self, n_channels=24, sampling=0.5 * units.ns):
        """
        Run function of the RNOG reader

        Parameters
        ----------
        n_channels: int
            number of RNOG channels to loop over, default 24

        sampling: float (units time)
            default sampling of RNOG detector

        """
        read_branches = ['run_number', 'event_number', 'station_number', 'radiant_data[24][2048]']
        # TODO for trigger info, can simulataneously iterate also over header tree, trigger info is stored in True/False flags for individual triggers there
        #      time is stored in header->readout_time as posix.
        for uproot_event in uproot.iterate({self.input_files, self.input_tree}, read_branches, step_size=1, how=dict, library="np"):
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
                        
            run_number = uproot_event["run_number"][0]
            evt_number = uproot_event["event_number"][0]
            station_id = uproot_event["station_number"][0]
            self.logger.info("Reading Run: {0}, Event {1}, Station {2}".format(run_number, evt_number, station_id))

            evt = NuRadioReco.framework.event.Event(run_number, evt_number)
            station = NuRadioReco.framework.station.Station(station_id)
            # TODO in future: do need to apply calibrations?

            radiant_data = uproot_event"radiant_data[24][2048]"][0] # returns array of n_channels, n_points
            # Loop over all channels in data
            for iCh in range(n_channels):

                channel = NuRadioReco.framework.channel.Channel(iCh)

                # Get data from array via graph method
                voltage = np.array(radiant_data[iCh]) * units.mV
                times = np.arange(len(voltage)) * sampling


                if voltage.shape[0] % 2 != 0:
                    voltage = voltage[:-1]
                sampling_rate = times[1] - times[0]

                channel.set_trace(voltage, sampling_rate)
                station.add_channel(channel)
                evt.set_station(station)
            yield evt

    def end(self):
        pass
