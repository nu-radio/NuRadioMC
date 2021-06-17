import numpy as np
import uproot
import NuRadioReco.framework.event
import NuRadioReco.framework.station
import NuRadioReco.framework.channel
from NuRadioReco.utilities import units
import astropy

class RNOGDataReader:

    def __init__(self, filenames, *args, **kwargs):
        self.__filenames = filenames
        self.__event_ids = None
        self.__run_numbers = None
        self.__parse_event_ids()
        self.__parse_run_numbers()
        self.__i_events_per_file = np.zeros((len(filenames), 2), dtype=int)
        i_event = 0
        for i_file, filename in enumerate(filenames):
            file = uproot.open(filename)
            events_in_file = file['waveforms'].num_entries
            self.__i_events_per_file[i_file] = [i_event, i_event + events_in_file]

    def get_filenames(self):
        return self.__filenames

    def get_event_ids(self):
        if self.__event_ids is None:
            return self.__parse_event_ids()
        return self.__event_ids

    def __parse_event_ids(self):
        self.__event_ids = np.array([], dtype=int)
        for filename in self.__filenames:
            file = uproot.open(filename)
            self.__event_ids = np.append(self.__event_ids, file['waveforms']['event_number'].array(library='np').astype(int))

    def get_run_numbers(self):
        if self.__run_numbers is None:
            self.__parse_run_numbers()
        return self.__run_numbers

    def __parse_run_numbers(self):
        self.__run_numbers = np.array([], dtype=int)
        for filename in self.__filenames:
            file = uproot.open(filename)
            self.__run_numbers = np.append(self.__event_ids, file['waveforms']['run_number'].array(library='np').astype(int))

    def get_n_events(self):
        return self.get_event_ids().shape[0]

    def get_event_i(self, i_event):
        event = NuRadioReco.framework.event.Event(self.get_run_numbers()[i_event], self.get_event_ids()[i_event])
        for i_file, filename in enumerate(self.__filenames):
            if self.__i_events_per_file[i_file, 0] <= i_event < self.__i_events_per_file[i_file, 1]:
                i_event_in_file = i_event - self.__i_events_per_file[i_file, 0]
                file = uproot.open(self.__filenames[i_file])
                station = NuRadioReco.framework.station.Station((file['waveforms']['station_number'].array(library='np')[i_event_in_file]))
                station.set_is_neutrino()

                if 'header' in file:
                    unix_time = file['header']['readout_time'].array(library='np')[i_event_in_file]
                    event_time = astropy.time.Time(unix_time, format='unix')
                    station.set_station_time(event_time)

                waveforms = file['waveforms']['radiant_data[24][2048]'].array(library='np', entry_start=i_event_in_file, entry_stop=(i_event_in_file+1))
                for i_channel in range(waveforms.shape[1]):
                    channel = NuRadioReco.framework.channel.Channel(i_channel)
                    channel.set_trace(waveforms[0, i_channel], 2. * units.GHz)
                    station.add_channel(channel)
                event.set_station(station)
                return event
        return None

    def get_event(self, event_id):
        run_numbers = self.get_run_numbers()
        event_ids = self.get_event_ids()

        matching_indices = np.where(event_ids == event_id[1])
        for i_event in matching_indices:
            if event_id[0] == run_numbers[i_event]:
                return self.get_event_i(i_event)
        #for i_event, ev_id in enumerate(event_ids):
        #    if event_id[1] == ev_id and event_id[0] == run_numbers[i_event]:
        #        return self.get_event_i(i_event)
        return None

    def iter_events(self):
        for ev_i in range(self.get_n_events()):
            yield self.get_event_i(ev_i)

    def get_detector(self):
        return None

    def get_header(self):
        return None
