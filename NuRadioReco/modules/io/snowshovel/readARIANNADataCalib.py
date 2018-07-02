import NuRadioReco.framework.event
import NuRadioReco.framework.station
import NuRadioReco.framework.channel
import ROOT
import numpy as np
from NuRadioReco.utilities import units
import datetime
import sys
import time
import os
sys.path.append(os.path.expandvars('$SNS'))
from scripts.online import AriUtils
import logging
logger = logging.getLogger("readARIANNAData")


class readARIANNAData:
    """
    Assumes a tree with calibrated data, shifted by the stop. Very basic module for now.
    """

    def begin(self, input_files, trigger_types=None, time_interval=None,
              tree='AmpOutData', run_number=None, event_id=None):
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
        event_id: int or None
            event id, all events with a different event id will be skipped.
            Default is None which means that all events are read in
        """
        self.__trigger_types = trigger_types
        self.__time_interval = time_interval
        self.__run_number = run_number
        self.__event_id = event_id
        self.data_tree = ROOT.TChain("CalibTree")
        self.config_tree = ROOT.TChain("ConfigTree")

        for input_file in input_files:
            logger.info("adding file {}".format(input_file))
            self.data_tree.Add(input_file)
            self.config_tree.Add(input_file)

        self.calwv = ROOT.TSnCalWvData()
        if(tree == 'RawData'):
            self.calwv = ROOT.TSnRawWaveform()
        self.data_tree.SetBranchAddress("{}.".format(tree), self.calwv)
#         self.data_tree.SetBranchAddress("AmpOutData.", self.calwv)

        self.readout_config = ROOT.TSnReadoutConfig()
        self.config_tree.SetBranchAddress("ReadoutConfig.", self.readout_config)

        self.n_events = self.data_tree.GetEntries()
        self.__id_current_event = -1
        self.__t = time.time()

        return self.n_events

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
                logger.warning("reading in event {}/{} ({:.0f}%) ETA: {:.1f} minutes".format(self.__id_current_event, self.n_events,
                                                                         100 * progress, eta))
            try:
                self.data_tree.GetEntry(self.__id_current_event)
                self.config_tree.GetEntry(self.__id_current_event)

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
                        logger.debug("skipping event because trigger type was not {type}".format(type=self.__trigger_types))
                        continue

                mac = self.data_tree.EventMetadata.GetStationId()
                self._station_id = AriUtils.getStnFromMacAdr(mac)

                evt_number = self.data_tree.EventHeader.GetEvtNum()
                run_number = self.data_tree.EventMetadata.GetRunNum()
                if(self.__run_number is not None):
                    if(run_number != self.__run_number):
                        continue
                if(self.__event_id is not None):
                    if(evt_number != self.__event_id):
                        continue

                evt_triggered = self.data_tree.EventHeader.IsThermal()
                evt = NuRadioReco.framework.event.Event(run_number, evt_number)

                if(self.__id_current_event % 1000 == 0):
                    logger.info("reading in station {station_id} run {run_number} event {evt_number} at time {time}".format(station_id=self._station_id,
                                                                                                         run_number=run_number,
                                                                                                         evt_number=evt_number,
                                                                                                         time=evt_time))
                logger.debug("reading in station {station_id} run {run_number} event {evt_number} at time {time}".format(station_id=self._station_id,
                                                                                                         run_number=run_number,
                                                                                                         evt_number=evt_number,
                                                                                                         time=evt_time))

                nChan = ord(self.readout_config.GetNchans())  # convert char to int
                self.sampling_rate = self.readout_config.GetSamplingRate() * units.GHz

                station = NuRadioReco.framework.station.Station(self._station_id)
                station.set_station_time(evt_time)
                station.set_triggered(evt_triggered)
                stop = np.array(self.data_tree.RawData.GetStopSamples())

                for iCh in xrange(nChan):
                    channel = NuRadioReco.framework.channel.Channel(iCh)
                    voltage = np.array(self.calwv.GetDataOnCh(iCh)) * units.mV
                    voltage = np.roll(voltage, -stop[0])
                    channel.set_trace(voltage, self.sampling_rate)
                    station.add_channel(channel)

                evt.set_station(station)
                yield evt
            except:
                logger.error("error in reading in event station {station_id} run {run_number} event {evt_number} at time {time}".format(station_id=self._station_id,
                                                                                                         run_number=run_number,
                                                                                                         evt_number=evt_number,
                                                                                                         time=evt_time))
                continue

    def end(self):
        pass
