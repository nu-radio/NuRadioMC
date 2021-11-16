import numpy as np
import uproot
import NuRadioReco.framework.event
import NuRadioReco.framework.station
import NuRadioReco.framework.channel
from NuRadioReco.utilities import units
import astropy.time
import glob


class RNOGDataReader:

    def __init__(self, filenames, *args, **kwargs):
        self.__filenames = filenames
        self.__event_ids = None
        self.__sampling_rate = 3.2 * units.GHz #TODO: 3.2 at the beginning of deployment. Will change to 2.4 GHz after firmware update eventually, but info not yet contained in the .root files. Read out once available.
        self.__parse_event_ids()
        self.__i_events_per_file = np.zeros((len(self.__filenames), 2), dtype=int)
        i_event = 0
        for i_file, filename in enumerate(filenames):
            file = self.__open_file(filename)
            events_in_file = file['waveforms'].num_entries
            self.__i_events_per_file[i_file] = [i_event, i_event + events_in_file]
            i_event += events_in_file

        self._root_trigger_keys = [
            'trigger_info.rf_trigger', 'trigger_info.force_trigger',
            'trigger_info.pps_trigger', 'trigger_info.ext_trigger',
            'trigger_info.radiant_trigger', 'trigger_info.lt_trigger',
            'trigger_info.surface_trigger'
        ]

    def get_filenames(self):
        return self.__filenames

    def get_event_ids(self):
        if self.__event_ids is None:
            return self.__parse_event_ids()
        return self.__event_ids

    def __parse_event_ids(self):
        event_ids = np.array([], dtype=int)
        run_numbers = np.array([], dtype=int)
        for filename in self.__filenames:
            file = self.__open_file(filename)
            event_ids = np.append(event_ids, file['waveforms']['event_number'].array(library='np').astype(int))
            run_numbers = np.append(run_numbers, file['header']['run_number'].array(library='np').astype(int))
        self.__event_ids = np.array([run_numbers, event_ids]).T

    def __open_file(self, filename):
        file = uproot.open(filename)
        if 'combined' in file:
            file = file['combined']
        return file

    def get_n_events(self):
        return self.get_event_ids().shape[0]

    def get_event_i(self, i_event):
        event = NuRadioReco.framework.event.Event(*self.get_event_ids()[i_event])
        for i_file, filename in enumerate(self.__filenames):
            if self.__i_events_per_file[i_file, 0] <= i_event < self.__i_events_per_file[i_file, 1]:
                i_event_in_file = i_event - self.__i_events_per_file[i_file, 0]
                file = self.__open_file(self.__filenames[i_file])
                station = NuRadioReco.framework.station.Station((file['waveforms']['station_number'].array(library='np')[i_event_in_file]))
                # station not set properly in first runs, try from header
                if station.get_id() == 0 and 'header' in file:
                    station = NuRadioReco.framework.station.Station((file['header']['station_number'].array(library='np')[i_event_in_file]))
                station.set_is_neutrino()

                if 'header' in file:
                    unix_time = file['header']['trigger_time'].array(library='np')[i_event_in_file]
                    event_time = astropy.time.Time(unix_time, format='unix')
                    station.set_station_time(event_time)
                    ### read in basic trigger data
                    for trigger_key in self._root_trigger_keys:
                        try:
                            has_triggered = bool(file['header'][trigger_key].array(library='np')[i_event_in_file])
                            trigger = NuRadioReco.framework.trigger.Trigger(trigger_key.split('.')[-1])
                            trigger.set_triggered(has_triggered)
                            # trigger.set_trigger_time(file['header']['trigger_time'])
                            station.set_trigger(trigger)
                        except uproot.exceptions.KeyInFileError:
                            pass

                waveforms = file['waveforms']['radiant_data[24][2048]'].array(library='np', entry_start=i_event_in_file, entry_stop=(i_event_in_file+1))
                for i_channel in range(waveforms.shape[1]):
                    channel = NuRadioReco.framework.channel.Channel(i_channel)
                    channel.set_trace(waveforms[0, i_channel]*units.mV, self.__sampling_rate)
                    station.add_channel(channel)
                event.set_station(station)
                return event
        return None

    def get_event(self, event_id):
        find_event = np.where((self.get_event_ids()[:,0] == event_id[0]) & (self.get_event_ids()[:,1] == event_id[1]))[0]
        if len(find_event) == 0:
            return None
        elif len(find_event) == 1:
            return self.get_event_i(find_event[0])
        else:
            raise RuntimeError('There are multiple events with the ID [{}, {}] in the file'.format(event_id[0], event_id[1]))

    def get_events(self):
        for ev_i in range(self.get_n_events()):
            yield self.get_event_i(ev_i)

    def get_detector(self):
        return None

    def get_header(self):
        return None
