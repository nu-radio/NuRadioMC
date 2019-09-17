from NuRadioReco.modules.base.module import register_run
import NuRadioReco.framework.event
import NuRadioReco.framework.station
import NuRadioReco.framework.channel
import ROOT
import numpy as np
from NuRadioReco.utilities import units
import datetime


class readARIANNAData:
    """
    Assumes a tree with calibrated data, shifted by the stop. Very basic module for now.
    """

    def begin(self, input_file, station_id):
        self.data_tree = ROOT.TChain("CalibTree")
        self.data_tree.Add(input_file)
        self.calwv = ROOT.TSnCalWvData()
        self.data_tree.SetBranchAddress("AmpOutDataShifted.", self.calwv)

        self.raw = ROOT.TSnRawWaveform()
        self.data_tree.SetBranchAddress("RawData.", self.calwv)

        self.config_tree = ROOT.TChain("ConfigTree")
        self.config_tree.Add(input_file)
        self.readout_config = ROOT.TSnReadoutConfig()
        self.config_tree.SetBranchAddress("ReadoutConfig.", self.readout_config)

        self.n_events = self.data_tree.GetEntries()

        self._station_id = station_id
        self.__id_current_event = 0

        return self.n_events

    @register_run()
    def run(self):
        while True:
            if(self.__id_current_event >= self.n_events):
                # all events processed
                break
            self.data_tree.GetEntry(self.__id_current_event)
            self.config_tree.GetEntry(self.__id_current_event)

            evt_number = self.data_tree.EventHeader.GetEvtNum()
            run_number = self.data_tree.EventMetadata.GetRunNum()
            evt_triggered = self.data_tree.EventHeader.IsThermal()
            evt = NuRadioReco.framework.event.Event(run_number, evt_number)

            nChan = ord(self.readout_config.GetNchans())  # convert char to int
            self.sampling_rate = self.readout_config.GetSamplingRate() * units.GHz

            evt_time = datetime.datetime.fromtimestamp(self.data_tree.EventHeader.GetUnixTime())

            station = NuRadioReco.framework.station.Station(self._station_id)
            station.set_station_time(evt_time)
            station.set_triggered(evt_triggered)

            for iCh in range(nChan):
                channel = NuRadioReco.framework.channel.Channel(iCh)
                voltage = np.array(self.calwv.GetDataOnCh(iCh)) * units.mV
                channel.set_trace(voltage, self.sampling_rate)
                station.add_channel(channel)

            evt.set_station(station)
            self.__id_current_event += 1
            yield evt

    def end(self):
        pass
