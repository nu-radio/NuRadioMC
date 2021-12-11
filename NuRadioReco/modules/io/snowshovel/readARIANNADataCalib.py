from NuRadioReco.modules.base.module import register_run
import NuRadioReco.framework.event
import NuRadioReco.framework.station
import NuRadioReco.framework.channel
import ROOT
import numpy as np
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import ARIANNAParameters as ARIpar
import datetime
import sys
import time
import os
from scripts.online import AriUtils
from scripts.offline import dacs2014
import logging
sys.path.append(os.path.expandvars('$SNS'))


class readARIANNAData:
    """
    Reads ARIANNA data as preprocessed in snowShovel. Can read RawData, FPNCorrectedData
    and CalibratedData.
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
            The trigger types can ['thermal', 'forced']. Default is None which
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
            if True walk through event in a random order
        """
        self.__trigger_types = trigger_types
        self.__time_interval = time_interval
        self.__run_number = run_number
        self.__event_ids = event_ids
        self.data_tree = ROOT.TChain("CalibTree")
        self.config_tree = ROOT.TChain("ConfigTree")

        for input_file in input_files:
            self.logger.info("adding file {}".format(input_file))
            self.data_tree.Add(input_file)
            self.config_tree.Add(input_file)

        self.calwv = ROOT.TSnCalWvData()
        if(tree == 'RawData'):
            self.calwv = ROOT.TSnRawWaveform()
        self.data_tree.SetBranchAddress("{}.".format(tree), ROOT.AddressOf(self.calwv))
#         self.data_tree.SetBranchAddress("AmpOutData.", self.calwv)

        self.readout_config = ROOT.TSnReadoutConfig()

        self.config_tree.SetBranchAddress("ReadoutConfig.", ROOT.AddressOf(self.readout_config))
        self.skipped_events = 0
        self.skipped_events_stop = 0

        self.n_events = self.data_tree.GetEntries()
        self._evt_range = np.arange(self.n_events, dtype=int)
        if(random_iterator):
            np.random.shuffle(self._evt_range)
        self.__id_current_event = -1

        n_events_config = self.config_tree.GetEntries()
        self.logger.debug("{} entries in config".format(n_events_config))
        self._config_keys = {}
        for i in range(n_events_config):
            self.config_tree.GetEntry(i)
            stn_id = AriUtils.getStnFromMacAdr(self.config_tree.ConfigMetadata.GetStationId())
            run_num = self.config_tree.ConfigMetadata.GetRunNum()
            seq_num = self.config_tree.ConfigMetadata.GetSeqNum()
            self._config_keys[(stn_id, run_num, seq_num)] = i

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
#             try:
            self.data_tree.GetEntry(self._evt_range[self.__id_current_event])

            evt_time = datetime.datetime.fromtimestamp(self.data_tree.EventHeader.GetUnixTime())
            if(self.__time_interval is not None):
                if(evt_time < self.__time_interval[0]):
                    continue
                if(evt_time > self.__time_interval[1]):
                    continue
            if(self.__trigger_types is not None):
                use_event = False
                for trigger in self.__trigger_types:
                    if(trigger == 'thermal'):
                        if(self.data_tree.EventHeader.IsThermal()):
                            use_event = True
                    if(trigger == 'forced'):
                        if(self.data_tree.EventHeader.IsForced()):
                            use_event = True
                if(use_event is False):
                    self.logger.debug("skipping event because trigger type was not {type}".format(type=self.__trigger_types))
                    continue

            mac = self.data_tree.EventMetadata.GetStationId()
            self._station_id = AriUtils.getStnFromMacAdr(mac)

            evt_number = self.data_tree.EventHeader.GetEvtNum()
            run_number = self.data_tree.EventMetadata.GetRunNum()
            if(self.__run_number is not None):
                if(run_number != self.__run_number):
                    continue
            if(self.__event_ids is not None):
                if(run_number not in self.__event_ids):
                    continue
                if(evt_number not in self.__event_ids[run_number]):
                    continue

            seq_number = self.data_tree.EventMetadata.GetSeqNum()
            try:
                self.config_tree.GetEntry(self._config_keys[(self._station_id, run_number, seq_number)])
            except:
                self.logger.error("no config entry exists for station {}, run {} and sequence {}. Skipping event...".format(self._station_id, run_number, seq_number))
                self.skipped_events += 1
                continue
            # check if config and event sequence are the same
            seq_num_config = self.config_tree.ConfigMetadata.GetSeqNum()
            if(seq_number != seq_num_config):
                raise Exception("seq number in config {} does not match sequence number in event {}".format(seq_num_config, seq_number))

            evt_triggered = self.data_tree.EventHeader.IsThermal()
            evt = NuRadioReco.framework.event.Event(run_number, evt_number)

            if(self.__id_current_event % 1000 == 0):
                self.logger.info("reading in station {station_id} run {run_number} event {evt_number} at time {time}".format(station_id=self._station_id, run_number=run_number, evt_number=evt_number, time=evt_time))
            self.logger.debug("reading in station {station_id} run {run_number} event {evt_number} at time {time}".format(station_id=self._station_id, run_number=run_number, evt_number=evt_number, time=evt_time))

            nChan = ord(self.readout_config.GetNchans())  # convert char to int
            name = self.readout_config.GetTypeName()
            if name == 'Custom':
                self.logger.warning("Event {event} of run {run} is skipped, as ReadoutConfig seems empty".format(event=evt_number, run=run_number))
                self.skipped_events += 1
                continue

            self.sampling_rate = self.readout_config.GetSamplingRate() * units.GHz

            station = NuRadioReco.framework.station.Station(self._station_id)
            station.set_station_time(evt_time)
            station.set_triggered(evt_triggered)
            stop = np.array(self.data_tree.RawData.GetStopSamples())

            # skip events that don't have a proper information of the stop point
            if (stop.size != 0):
                for iCh in range(nChan):
                    channel = NuRadioReco.framework.channel.Channel(iCh)
                    voltage = np.array(self.calwv.GetDataOnCh(iCh)) * units.mV
                    voltage = np.roll(voltage, -stop[0])
                    channel.set_trace(voltage, self.sampling_rate)
                    station.add_channel(channel)
            else:
                self.logger.warning(" Event {event} of run {run} is skipped, no stop point for rolling array!".format(event=evt_number, run=run_number))
                self.skipped_events_stop += 1
                continue

            station.set_ARIANNA_parameter(ARIpar.seq_num, seq_number)
            # read and save start and stop time of a sequence
            start = datetime.datetime.fromtimestamp(self.config_tree.TrigStartClock.GetCurrTime().GetSec())
            stop = datetime.datetime.fromtimestamp(self.config_tree.TrigStopClock.GetCurrTime().GetSec())
            if(start < datetime.datetime(1971, 1, 1)):
                start = None
            if(stop < datetime.datetime(1971, 1, 1)):
                stop = None
            station.set_ARIANNA_parameter(ARIpar.seq_start_time, start)
            station.set_ARIANNA_parameter(ARIpar.seq_stop_time, stop)

            station.set_ARIANNA_parameter(ARIpar.comm_duration, self.config_tree.DAQConfig.GetCommWinDuration() * units.s)
            station.set_ARIANNA_parameter(ARIpar.comm_period, self.config_tree.DAQConfig.GetCommWinPeriod() * units.s)
            station.set_ARIANNA_parameter(ARIpar.l1_supression_value, self.config_tree.DAQConfig.GetL1SingleFreqRatioCut())
            station.set_ARIANNA_parameter(ARIpar.internal_clock_time, self.data_tree.EventHeader.GetDTms() * units.ms)
            dacset = self.config_tree.DAQConfig.GetDacSet().GetDacs()
            dacsV = {}
            for iCh in range(len(dacset)):
                dacsV[iCh] = {}
                for ihl, hl in enumerate(['low', 'high']):
                    dacsV[iCh][hl] = dacs2014.getVth(AriUtils.getBoardFromMacAdr(AriUtils.getMacAdrFromStn(self._station_id)), iCh, dacset[iCh][ihl], hl)
            station.set_ARIANNA_parameter(ARIpar.trigger_thresholds, dacsV)

            evt.set_station(station)
            yield evt
#             except:
#                 self.logger.error("error in reading in event station {station_id} run {run_number} event {evt_number} at time {time}".format(station_id=self._station_id,
#                 run_number=run_number,
#                 evt_number=evt_number,
#                 time=evt_time))
#                 continue

    def end(self):
        if self.skipped_events > 0:
            self.logger.warning("Skipped {} events due to problems in config".format(self.skipped_events))
        if self.skipped_events_stop > 0:
            self.logger.warning("Skipped {} events due to problems in stop bit".format(self.skipped_events_stop))
