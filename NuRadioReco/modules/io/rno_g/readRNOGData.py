import NuRadioReco.framework.event
from NuRadioReco.modules.base.module import register_run
import NuRadioReco.framework.station
import NuRadioReco.framework.channel
import NuRadioReco.framework.trigger
import NuRadioReco.modules.channelSignalReconstructor
signal_reconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()

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
from collections import OrderedDict
import astropy.time


class readRNOGData:
    """
    This is the data reader for RNO-G. Reads RNO-G data from ROOT format using uproot
    """
    def __init__(self):
        self.logger = logging.getLogger("NuRadioReco.readRNOGdata")
        self.__id_current_event = None
        self.__t = None
        self.__sampling_rate = 3.2 * units.GHz #TODO: 3.2 at the beginning of deployment. Will change to 2.4 GHz after firmware update eventually, but info not yet contained in the .root files. Read out once available.
        self._iterator_data = None
        self._iterator_header = None
        self._data_treename = "waveforms"
        self._header_treename = "header"
        self.n_events = None
        self.input_files = []

    def begin(self, input_files, input_files_header=None):

        """
        Begin function of the RNO-G reader

        Parameters
        ----------
        input_files: list of paths to files containing waveforms
        input_files_header: list of paths to files containing header,
            if None, headers are expected to be stored in the input_files also
        """

        self.__id_current_event = -1
        self.__t = time.time()

        if isinstance(input_files, six.string_types):
            input_files = [input_files]
        if isinstance(input_files_header, six.string_types):
            input_files_header = [input_files_header]
        if input_files_header is None:
            input_files_header = input_files

        self.input_files = input_files
        self.input_files_header = input_files_header

        self.n_events = 0
        # get the total number of events of all input files
        for filename in input_files:
            file = uproot.open(filename)
            if 'combined' in file:
                file = file['combined']
            self.n_events += file[self._data_treename].num_entries
        self._set_iterators()

        return self.n_events

    def _set_iterators(self, cut=None):
        """
        Set uproot iterators to loop over event trees
  
        Parameters
        ----------
        cut: str
            cut string to apply (e.g. for initial event selection based on event_number, ...
            e.g. "(event_number==1)" or "(run_number==1)&(event_number<10)" 
        """
        self.__id_current_event = -1

        datadict = OrderedDict()
        for filename in self.input_files:
            if 'combined' in uproot.open(filename):
                datadict[filename] = 'combined/' + self._data_treename
            else:
                datadict[filename] = self._data_treename

        headerdict = OrderedDict()
        for filename in self.input_files_header:
            if 'combined' in uproot.open(filename):
                headerdict[filename] = 'combined/' + self._header_treename
            else:
                headerdict[filename] = self._header_treename

        # iterator over single events (step 1), for event looping in NuRadioReco dataformat
        # may restrict which data to read in the iterator by adding second argument
        # read_branches = ['run_number', 'event_number', 'station_number', 'radiant_data[24][2048]']
        self._iterator_data = uproot.iterate(datadict, cut=cut,step_size=1, how=dict, library="np")
        self._iterator_header = uproot.iterate(headerdict, cut=cut, step_size=1, how=dict, library="np")

        self.uproot_iterator_data = uproot.iterate(datadict, cut=cut, step_size=1000)
        self.uproot_iterator_header = uproot.iterate(headerdict, cut=cut, step_size=1000)

    @register_run()
    def run(self, channels=np.arange(24), event_numbers=None, run_numbers=None, cut_string=None):
        """
        Run function of the RNOG reader

        Parameters
        ----------
        n_channels: int
            number of RNOG channels to loop over, default 24

        event_numbers: None or dict
            if dict, use a dict with run number as key and list of event numbers as items

        run_numbers: None or list
            list of run numbers to select
            Caveat: use only if event_numbers are not set

        cut_string: string
            selection string for event pre-selection
            Cavieat: use only if event_numbers and run_numbers are not set
        """
    
        # generate cut string based on passed event_numbers or run_numbers parameters
        if not run_numbers is None:
            event_cuts = "|".join(["(run_number==%i)" for run_number in run_numbers])
            cut_string = "|".join(event_cuts)
        if not event_numbers is None:
            event_cuts = []
            for run in event_numbers:
                events = event_numbers[run]
                for event in events:
                    event_cuts.append("(run_number==%i)&(event_number==%i)" %(run, event))
            cut_string = "|".join(event_cuts)
        self.cut_string = cut_string

        self._set_iterators(cut=self.cut_string)
        root_trigger_keys = [
            'trigger_info.rf_trigger', 'trigger_info.force_trigger',
            'trigger_info.pps_trigger', 'trigger_info.ext_trigger',
            'trigger_info.radiant_trigger', 'trigger_info.lt_trigger',
            'trigger_info.surface_trigger'
        ]
        self.__t = time.time()
        # Note: reading single events is inefficient...
        # for event_header, event in zip(self._iterator_header, self._iterator_data):
        for event_headers, events in zip(self.uproot_iterator_header, self.uproot_iterator_data):
          for event_header, event in zip(event_headers, events):
            self.__id_current_event += 1
            #if self.__id_current_event >= self.n_events:
            #    # all events processed, but iterator should stop before anyways.
            #    break
            if self.__id_current_event % 1000 == 0:
                progress = 1. * self.__id_current_event / self.n_events
                eta = 0
                if self.__id_current_event > 0:
                    eta = (time.time() - self.__t) / self.__id_current_event * (self.n_events - self.__id_current_event) / 60.
                self.logger.warning("reading in event {}/{} ({:.0f}%) ETA: {:.1f} minutes".format(self.__id_current_event, self.n_events, 100 * progress, eta))
                
            run_number = event["run_number"]
            evt_number = event["event_number"]
            station_id = event_header["station_number"]
            self.logger.info("Reading Run: {run_number}, Event {evt_number}, Station {station_id}")

            evt = NuRadioReco.framework.event.Event(run_number, evt_number)
            station = NuRadioReco.framework.station.Station(station_id)
            #TODO in future: do need to apply calibrations?

            unix_time = event_header["trigger_time"]
            event_time = astropy.time.Time(unix_time, format='unix')

            station.set_station_time(event_time)
            for trigger_key in root_trigger_keys:
                try:
                    has_triggered = bool(event_header[trigger_key])
                    trigger = NuRadioReco.framework.trigger.Trigger(trigger_key.split('.')[-1])
                    trigger.set_triggered(has_triggered)
                    station.set_trigger(trigger)
                except ValueError:
                    pass

            radiant_data = event["radiant_data[24][2048]"] # returns array of n_channels, n_points
            # Loop over all requested channels in data
            for chan in channels:
                channel = NuRadioReco.framework.channel.Channel(chan)

                # Get data from array via graph method
                voltage = np.array(radiant_data[chan]) * units.mV
                #times = np.arange(len(voltage)) * sampling

                if voltage.shape[0] % 2 != 0:
                    voltage = voltage[:-1]

                #TODO: need to subtract mean... probably not if running signal reconstructor?
                #channel.set_trace(voltage-np.mean(voltage), sampling_rate)
                channel.set_trace(voltage, self.__sampling_rate)
                station.add_channel(channel)
            evt.set_station(station)
            # we want to have access to basic signal quantities with implementation from NuRadioReco
            #TODO: maybe this should be run in external module?
            signal_reconstructor.run(evt, station, None)
            yield evt

    def get_events(self):
        return self.run()

    def get_n_events(self):
        return self.n_events

    def get_filenames(self):
        return self.input_files

    def end(self):
        pass
