from NuRadioReco.modules.base.module import register_run, setup_logger
import NuRadioReco.framework.event
import NuRadioReco.framework.station
import NuRadioReco.framework.channel
import uproot
import numpy as np
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import ARIANNAParameters as ARIpar
import datetime
import sys
import time
import os
from scipy import interpolate

# copies/excerpts from ARIANNA software:
from NuRadioReco.modules.io.snowshovel import AriUtils_dicts
from NuRadioReco.modules.io.snowshovel import dacs2014

import logging
import awkward


from NuRadioReco.modules.io.snowshovel.arianna_uproot_params import ARIANNA_uproot_interpretation, ARIANNA_TRIGGER, getReadoutConfigTypeFromNum
from NuRadioReco.modules.io.snowshovel.arianna_uproot_params import arianna_calib_dict, arianna_voltage_dict, arianna_temperature_dict, arianna_config_dict
from NuRadioReco.modules.io.snowshovel.arianna_uproot_params import ariannaCalibTreeParameters as par_data
from NuRadioReco.modules.io.snowshovel.arianna_uproot_params import ariannaVoltageTreeParameters as par_voltage
from NuRadioReco.modules.io.snowshovel.arianna_uproot_params import ariannaTemperatureTreeParameters as par_temperature
from NuRadioReco.modules.io.snowshovel.arianna_uproot_params import ariannaConfigTreeParameters as par_config



#def find_first_stop(stop_array):
#    # stop array consists of  8bit blocks
#    # find block numbers with stop bit set
#    stop_set = np.flatnonzero(stop_array)
#    if len(stop_set) == 0:
#        # make sure a stop bit is set otherwise return empty list
#        return np.nan
#    else:
#    # use floor and log to find only first occurence of stop bit
#        found_stop = (stop_set+1)*8-np.floor(np.log2(stop_array[stop_set])+1)
#        return int(found_stop[0])

def find_stops(stop_block_array):
    """
    find the stop bits set in an array of stop blocks
    implementation matches the one ARIANNA ROOT software
    
    Paramteters
    -----------
    stop_block_array: array of 8 bit blocks where (typically 1 or 2) stop bits are nonzero

    Returns
    -------
    array of found stop positions
    """

    # mask to find set bits within a block
    bitcomparison_mask = 2**np.arange(8)


    # stop block consists of 8bit blocks
    # find blocks with stop bit set
    found_stops = []
    stop_set = np.flatnonzero(stop_block_array)
    # loop over blocks with a stop set
    for block_i in stop_set:
        bits_set = np.flatnonzero(stop_block_array[block_i]&bitcomparison_mask)
        for bit_i in bits_set:
            found_stops.append((block_i+1)*8 - (bit_i+1))
    return np.array(found_stops)


class readARIANNAData:
    """
    Reads ARIANNA data as preprocessed in snowShovel. Can read Raw Data, FPN Corrected Data
    and Calibrated Data.
    """

    def __init__(self):
        self.logger = logging.getLogger("NuRadioReco.readARIANNAData")

    def begin(self, input_files, trigger_types=None, time_interval=None,
              tree='AmpOutData', run_number=None, event_ids=None,
              random_iterator=False):
        """
        read FPN and gain calibrated ARIANNA data (snowshovel data format)

        Parameters
        ----------
        input_files: array of strings
            a list of input files
        trigger_types: list of strings or None
            only events that have the specified trigger types are read.
            The trigger types can be ['thermal', 'forced']. Default is None which
            means that all events are read regardless of their trigger type
        time_interval: [datetime, datetime] or None
            all events outside of the specified time interval are skipped
            Default is None, i.e., all events are read
        tree: string
            low level reconstruction stage. Available options are
             * RawData: the raw ADC counts
             * FPNSubData: FPN subtracted data (still ADC counts but with mean zero)
             * AmpOutData: FPN and gain calibrated data (default)
        run_number: int or None
            run number, all events with a different run number will be skipped.
            Default is None which means that all events are read in
        event_ids: dictionary or None
            specify any combination of run and event ids, all other events will be skipped.
            key is the run id, values are the event ids
            Default is None which means that all events are read in
        random_iterator: bool (default False)
            if True walk through events in a random order
        """
        self.__trigger_types = trigger_types
        self.__time_interval = time_interval
        self.__run_number = run_number
        self.__event_ids = event_ids
        self.data_tree_name = "CalibTree"
        self.config_tree_name = "ConfigTree"
        self.trace_branch = par_data[tree+"_data"]


        if isinstance(input_files, str):
            input_files = [input_files]
        self.__input_files = input_files
        self.__uproot_files = [uproot.open(f) for f in self.__input_files]

        # uproot.iterate does not work for ARIANNA data because uproot is not able to find correct interpretation for times and trigger mask
        # so we load a list of data_trees with uproot.open and need to find later, in which file to search for a specific event.
        self.data_trees = [f[self.data_tree_name] for f in self.__uproot_files]
        self.n_events_per_file = [tree.num_entries for tree in self.data_trees]
        self.n_events = np.sum(self.n_events_per_file)

        self.skipped_events = 0
        self.skipped_events_stop = 0

        # shuffle order in which events are read in if requested
        self._evt_range = np.arange(self.n_events, dtype=np.int)
        if(random_iterator):
            np.random.shuffle(self._evt_range)
        self.__id_current_event = -1

        # config tree information is small, so one may read it right away.
        self.config = [self._read_config(f) for f in self.__uproot_files]

        ## temperature is not sampled for every event
        #self._temperatures = [self._read_temperature(f) for f in self.__uproot_files]
        ## battery info is not stored per event
        #self._power_info = [self._read_power(f) for f in self.__uproot_files]
        
        # use interpolators to interpolate the taken temperature/power values to the event times
        self.interpol_temperature = [self._temperature_interpolator(f) for f in self.__uproot_files]
        self.interpol_V1 = [self._power_interpolator(f, 'V1') for f in self.__uproot_files]
        self.interpol_V2 = [self._power_interpolator(f, 'V2') for f in self.__uproot_files]        


        self.config_trees = [f[self.config_tree_name] for f in self.__uproot_files]
        self.n_events_config_per_file = [tree.num_entries for tree in self.config_trees]
        n_events_config = np.sum(self.n_events_config_per_file)
        self.logger.debug("{} entries in config".format(n_events_config))

        self.__t = time.time()
        
        return self.n_events

    @register_run()
    def run(self):
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
                self.logger.warning(
                    "reading in event {}/{} ({:.0f}%) ETA: {:.1f} minutes".format(
                        self.__id_current_event,
                        self.n_events,
                        100 * progress, eta
                    )
                )
            
            # get the 'real' event number this corresponds to (may be shuffled)
            self.__current_event_num = self._evt_range[self.__id_current_event]
            # get the appropriate file in which to look for this event
            self.__current_file_num = np.argmax(self.__current_event_num < np.cumsum(self.n_events_per_file))
            self.__current_event_num_in_file = self.__current_event_num - np.sum(self.n_events_per_file[:self.__current_file_num])

            # data tree for calib and config in which the current event resides
            self.__file_calib = self.data_trees[self.__current_file_num]
            self.__file_config = self.config[self.__current_file_num]
            self.interpolate_temperature = self.interpol_temperature[self.__current_file_num]
            self.interpolate_V1 = self.interpol_V1[self.__current_file_num]
            self.interpolate_V2 = self.interpol_V2[self.__current_file_num]

            # get back the current event
            current = self._read_event_info(self.__file_calib, self.__current_event_num_in_file)


            # convert event unix time to datetime time is a tuple of [[sec][nanosec]] when interpreted as numpy array
            unix_time = current[par_data.time][0][0]
            evt_time = datetime.datetime.utcfromtimestamp(unix_time)
            if(self.__time_interval is not None):
                if(evt_time < self.__time_interval[0]):
                    continue
                if(evt_time > self.__time_interval[1]):
                    continue
           
            # get trigger information
            if(self.__trigger_types is not None):
                use_event = False
                for trigger in self.__trigger_types:
                    if(trigger == 'thermal'):
                        # bitwise comparison of mask with thermal trigger bit
                        if(current[par_data.trigger_mask] & ARIANNA_TRIGGER['thermal']):
                            use_event = True
                    if(trigger == 'forced'):
                        # bitwise comparison of mask with forced trigger bit
                        if(current[par_data.trigger_mask] & ARIANNA_TRIGGER['forced']):
                            use_event = True
                if(use_event is False):
                    self.logger.debug("skipping event because trigger type was not {type}".format(type=self.__trigger_types))
                    continue

            # get the station id
            mac = current[par_data.station_mac]
            station_id = AriUtils_dicts.getStnFromMacAdr(mac)

            # do not use all events if requested reader?
            if(self.__run_number is not None):
                if(current[par_data.run_number] != self.__run_number):
                    continue
            if(self.__event_ids is not None):
                if(current[par_data.run_number] not in self.__event_ids):
                    continue
                if(current[par_data.event_id] not in self.__event_ids[current[par_data.run_number]]):
                    continue

            # get the sequence number end get the corresponding entry in the config
            seq_number = current[par_data.sequence_number]
            run_number = current[par_data.run_number]
            evt_number = current[par_data.event_id]

            # find the corresponding entry in the config
            if (seq_number in self.__file_config[par_config.sequence_number]):
                cfgi = np.argmax(self.__file_config[par_config.sequence_number] == seq_number)
            else:
                self.logger.error("no config entry exists for station {}, run {} and sequence {}. Skipping event...".format(station_id, run_number, seq_number))
                self.skipped_events += 1
                continue

            # extend the current event info by the entry in the config info.... maybe not fastest performing, but makes code below hopefully more readable.
            current.update({cfg: self.__file_config[cfg][cfgi] for cfg in self.__file_config})
            if not (current[par_config.run_number] == run_number):
                raise Exception("config entry has no matching run {}. Skipping event...".format(run_number))
            if not (current[par_config.station_mac] == mac):
                raise Exception("config entry has no matching station {}. Skipping event...".format(station_id))

            # NuRadioReco event
            evt = NuRadioReco.framework.event.Event(run_number, evt_number)
            
            if(self.__id_current_event % 1000 == 0):
                self.logger.info("reading in station {station_id} run {run_number} event {evt_number} at time {time}".format(station_id=station_id, run_number=run_number, evt_number=evt_number, time=evt_time))
            self.logger.debug("reading in station {station_id} run {run_number} event {evt_number} at time {time}".format(station_id=station_id, run_number=run_number, evt_number=evt_number, time=evt_time))
            
            # readout config check... this check is left in there as it was part of the ROOT event reader
            name = getReadoutConfigTypeFromNum(current[par_config.ReadoutConfig_Type])
            self.logger.info("ReadoutConfig Type is {}".format(name))
            if name == 'Custom':
                self.logger.warning("Event {event} of run {run} is skipped, as ReadoutConfig seems empty".format(event=evt_number, run=run_number))
                self.skipped_events += 1
                continue
            
            self.sampling_rate = 1./current[par_config.ReadoutConfig_SampDT] * units.GHz

            station = NuRadioReco.framework.station.Station(station_id)
            
            station.set_station_time(evt_time)

            # do a bit comparison with ARIANNA trigger mask to check if 'thermal' fired
            evt_is_triggered = int(current[par_data.trigger_mask]) & ARIANNA_TRIGGER['thermal']
            station.set_triggered(evt_is_triggered)

            # skip events that don't have a proper information of the stop point
            stop_block_array = np.array(current[par_data.RawData_stop_bits])
            stops = find_stops(stop_block_array)

            # read the voltage traces
            voltages = current[self.trace_branch]
            if (stops.size != 0):
                for iCh in range(len(voltages)):
                    channel = NuRadioReco.framework.channel.Channel(iCh)
                    voltage = voltages[iCh] * units.mV
                    voltage = np.roll(voltage, -stops[0])
                    channel.set_trace(voltage, self.sampling_rate)
                    station.add_channel(channel)
            else:
                self.logger.warning(" Event {event} of run {run} is skipped, no stop point for rolling array!".format(event=evt_number, run=run_number))
                self.skipped_events_stop += 1
                continue
            
            station.set_ARIANNA_parameter(ARIpar.seq_num, seq_number)
            # read and save start and stop time of a sequence
            start = datetime.datetime.utcfromtimestamp(current[par_config.TrigStartClock_CurrTime].fSec[0])
            stop = datetime.datetime.utcfromtimestamp(current[par_config.TrigStopClock_CurrTime].fSec[0])
            if(start < datetime.datetime(1971, 1, 1)):
                start = None
            if(stop < datetime.datetime(1971, 1, 1)):
                stop = None
            station.set_ARIANNA_parameter(ARIpar.seq_start_time, start)
            station.set_ARIANNA_parameter(ARIpar.seq_stop_time, stop)

            station.set_ARIANNA_parameter(ARIpar.comm_duration, current[par_config.DAQConfig_ComWin].fDur[0] * units.s)
            station.set_ARIANNA_parameter(ARIpar.comm_period, current[par_config.DAQConfig_ComWin].fPer[0] * units.s)

            station.set_ARIANNA_parameter(ARIpar.l1_supression_value, np.nan) #TODO skip for now, because of uproot not liking the trigger info map, should be DAQConfig.GetL1SingleFreqRatioCut())

            station.set_ARIANNA_parameter(ARIpar.internal_clock_time, current[par_data.DTms] * units.ms)

            # get the trigger thresholds
            dacset = current[par_config.DAQConfig_Dacs].fDacs
            dacsV = {}

            
            station.set_ARIANNA_parameter(ARIpar.interpolated_temperature, self.interpolate_temperature(unix_time))
            station.set_ARIANNA_parameter(ARIpar.interpolated_v1, self.interpolate_V1(unix_time))
            station.set_ARIANNA_parameter(ARIpar.interpolated_v2, self.interpolate_V2(unix_time))
            station.set_ARIANNA_parameter(ARIpar.power_mode, current[par_config.DAQConfig_PowMode])
            station.set_ARIANNA_parameter(ARIpar.trigger_mask, current[par_data.trigger_mask])
            station.set_ARIANNA_parameter(ARIpar.readout_config_type, getReadoutConfigTypeFromNum(current[par_config.ReadoutConfig_Type]))

            for iCh in range(len(dacset)):
                dacsV[iCh] = {}
                for ihl, hl in enumerate(['low', 'high']):
                    dacsV[iCh][hl] = dacs2014.getVth(AriUtils_dicts.getBoardFromMacAdr(str(mac)), iCh, dacset[iCh][ihl], hl)
            station.set_ARIANNA_parameter(ARIpar.trigger_thresholds, dacsV)
            
            evt.set_station(station)
            yield evt
            


    def _read_event_info(self, tree, entry_num):
        """ read the calib data tree information for one event

        Note: uproot.iterate would be much more convenient, but custom interpretations are necessary due to odd header bytes in the root files
        
        Parameters
        ----------
        tree: the Calib tree of an uproot file
        entry_num: number of entry in the tree to read

        Returns
        -------
        current: dict of parameters, values for the requested current event
        """

        current = {}
        for param, (trash, branch, interpretation) in arianna_calib_dict.items():
            current[param] = np.array(tree[branch].array(entry_start = entry_num, entry_stop = entry_num+1, interpretation=interpretation))[0]
            if interpretation==ARIANNA_uproot_interpretation['trigger']:
                current[param] = current[param][0]
        #awkward.zip throws errors, so we'll have to live with the dictionary
        return current

    def _read_in_dict(self, opened_file, arianna_dict):
        """ read in information for a dict of parameter from an entire opened uproot file

        Note: uproot.iterate would be much more convenient, but custom interpretations are necessary due to odd header bytes in the root files
        
        Parameters
        ----------
        opened_file: a root file opened with uproot.open
        arianna_dict: a dictionary for each key being a 3-item list of
            #: tree name
            #: branch name
            #: uproot interpretation, None if automatically discovered (non None should ideally never occur, but this is not the case for ARIANNA branches)

        Returns
        -------
        out_dict: dict of parameters, values for the events of the file
        """
        out_dict = {}
        for param, (tree, branch, interpretation) in arianna_dict.items():
            out_dict[param] = opened_file[tree][branch].array(interpretation=interpretation)
        
        return out_dict

    def _read_power(self, opened_file):
        return self._read_in_dict(opened_file, arianna_voltage_dict)

    def _read_temperature(self, opened_file):
        return self._read_in_dict(opened_file, arianna_temperature_dict)

    def _read_config(self, opened_file):
        return self._read_in_dict(opened_file, arianna_config_dict)

    def _temperature_interpolator(self, nf):
        """
        Imports the temperature vs. posix time from a noise file (nf)
        The temperature is not taken per event, so the posix_time and temperature value is returned
        if interpolation_times is specified, interpolate to these posix time stamps
        """        
        # get the correct tree
        nt = nf["TemperatureTree"]
        time = np.array(nt['Temperature./Temperature.fTime'].array(interpretation = ARIANNA_uproot_interpretation['time']).fSec)[:,0]
        data = np.array(nt['Temperature./Temperature.fTemp'].array())

        # generate an interpolator for the requested times
        interpolator = interpolate.interp1d(time, data, bounds_error=False, fill_value=(data[0],data[-1]))
        return interpolator

    def _power_interpolator(self, nf, val="V1"):
        """
        Imports the voltage vs. posix time from a noise file (nf)
        # Todo: not sure if V1 and V2 refer to two batteries...
        The voltage is not taken per event, so the posix_time and voltage value is returned
        if interpolation_times is specified, interpolate to these posix time stamps
        """
        # get the correct tree
        nt = nf["VoltageTree"]
        time = np.array(nt['PowerReading./PowerReading.fTime'].array(interpretation = ARIANNA_uproot_interpretation['time']).fSec)[:,0]
        data = np.array(nt['PowerReading./PowerReading.fave'+val].array()) * units.mV

        # generate an interpolator for the requested times
        interpolator = interpolate.interp1d(time, data, bounds_error=False, fill_value=(data[0],data[-1]))
        return interpolator


    def end(self):
        if self.skipped_events > 0:
            self.logger.warning("Skipped {} events due to problems in config".format(self.skipped_events))
        if self.skipped_events_stop > 0:
            self.logger.warning("Skipped {} events due to problems in stop bit".format(self.skipped_events_stop))
