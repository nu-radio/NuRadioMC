from __future__ import absolute_import, division, print_function, unicode_literals
import pickle
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.modules.io.NuRadioRecoio import VERSION, VERSION_MINOR
import logging
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.detector import generic_detector
logger = logging.getLogger("eventWriter")


def get_header(evt):
    header = {'stations': {}}
    for iS, station in enumerate(evt.get_stations()):
        header['stations'][station.get_id()] = station.get_parameters().copy()
        if(station.has_sim_station()):
            header['stations'][station.get_id()]['sim_station'] = {}
            header['stations'][station.get_id()]['sim_station'] = station.get_sim_station().get_parameters()
        header['stations'][station.get_id()][stnp.station_time] = station.get_station_time()
    header['event_id'] = (evt.get_run_number(), evt.get_id())
    return header


class eventWriter:
    """
    save events to file
    """
    def __init__(self):
        # initialize attributes
        self.__filename = None
        self.__check_for_duplicates = None
        self.__number_of_events = None
        self.__current_file_size = None
        self.__number_of_files = None
        self.__max_file_size = None
        self.__stored_stations = None
        self.__stored_channels = None
        self.__header_written = None
        self.__event_ids_and_runs = None
        self.__events_per_file = None
        self.__events_in_current_file = 0

    def __write_fout_header(self):
        if self.__number_of_files > 1:
            self.__fout = open("{}_part{:02d}.nur".format(self.__filename, self.__number_of_files), 'wb')
        else:
            self.__fout = open("{}.nur".format(self.__filename), 'wb')
        b = bytearray()
        b.extend(VERSION.to_bytes(6, 'little'))
        b.extend(VERSION_MINOR.to_bytes(6, 'little'))
        self.__fout.write(b)
        self.__header_written = True

    def begin(self, filename, max_file_size=1024, check_for_duplicates=False, events_per_file=None):
        """
        begin method

        Parameters
        ----------
        filename: string
            Name of the file into which events shall be written
        max_file_size: maximum file size in Mbytes
                    (if the file exceeds the maximum file the output will be split into another file)
        check_for_duplicates: bool (default False)
            if True, the event writer raises an exception when an event with a (run,eventid) pair is written that is already
            present in the data file
        events_per_file: int
            Maximum number of events to be written into the same file. After more than events_per_file have been written
            into the same file, the output will be split into another file. If max_file_size and events_per_file are
            both set, the file will be split whenever any of the two conditions is fullfilled.
        """
        if filename[-4:] == '.nur':
            self.__filename = filename[:-4]
        else:
            self.__filename = filename
        if filename[-4:] == '.ari':
            logger.warning('The file ending .ari for NuRadioReco files is deprecated. Please use .nur instead.')
        self.__check_for_duplicates = check_for_duplicates
        self.__number_of_events = 0
        self.__current_file_size = 0
        self.__number_of_files = 1
        self.__max_file_size = max_file_size * 1024 * 1024  # in bytes
        self.__stored_stations = []
        self.__stored_channels = []
        self.__event_ids_and_runs = []  # Remember which event IDs are already in file to catch duplicates
        self.__header_written = False  # Remember if we still have to write the current file header
        self.__events_per_file = events_per_file

    @register_run()
    def run(self, evt, det=None, mode=None):
        """
        writes NuRadioReco event into a file

        Parameters
        ----------
        evt: NuRadioReco event object
        det: detector object
            If a detector object is passed, the detector description for the
            events is written in the file as well
        mode: dictionary, optional 
            Specifies what will saved into the `*.nur` output file.
            Can contain the following keys:

            * 'Channels': if True channel traces of Stations will be saved
            * 'ElectricFields': if True (reconstructed) electric field traces of Stations will be saved
            * 'SimChannels': if True SimChannels of SimStations will be saved
            * 'SimElectricFields': if True electric field traces of SimStations will be saved

            if no dictionary is passed, the default option is to save all of the above

        """
        if mode is None:
            mode = {
                'Channels': True,
                'ElectricFields': True,
                'SimChannels': True,
                'SimElectricFields': True
            }
        self.__check_for_duplicate_ids(evt.get_run_number(), evt.get_id())
        if not self.__header_written:
            self.__write_fout_header()

        event_bytearray = self.__get_event_bytearray(evt, mode)
        self.__fout.write(event_bytearray)
        self.__current_file_size += event_bytearray.__sizeof__()
        self.__number_of_events += 1
        self.__event_ids_and_runs.append([evt.get_run_number(), evt.get_id()])
        self.__events_in_current_file += 1

        if det is not None:
            detector_dict = self.__get_detector_dict(evt, det)  # returns None if detector is already saved
            if detector_dict is not None:
                detector_bytearray = self.__get_detector_bytearray(detector_dict)
                self.__fout.write(detector_bytearray)
                self.__current_file_size += detector_bytearray.__sizeof__()
            if isinstance(det, generic_detector.GenericDetector):
                changes_bytearray = self.__get_detector_changes_byte_array(evt, det)
                if changes_bytearray is not None:
                    self.__fout.write(changes_bytearray)
                    self.__current_file_size += changes_bytearray.__sizeof__()

        logger.debug("current file size is {} bytes, event number {}".format(self.__current_file_size,
                     self.__number_of_events))

        if(self.__current_file_size > self.__max_file_size or self.__events_in_current_file == self.__events_per_file):
            logger.info("current output file exceeds max file size -> closing current output file and opening new one")
            self.__current_file_size = 0
            self.__fout.close()
            self.__number_of_files += 1
            # self.__filename = "{}_part{:02d}".format(self.__filename, self.__number_of_files)
            self.__stored_stations = []
            self.__stored_channels = []
            self.__event_ids_and_runs = []
            self.__header_written = False
            self.__events_in_current_file = 0

    def __get_event_bytearray(self, event, mode):
        evt_header_str = pickle.dumps(get_header(event), protocol=4)
        b = bytearray()
        b.extend(evt_header_str)
        evt_header_length = len(b)
        evt_string = event.serialize(mode)
        b = bytearray()
        b.extend(evt_string)
        evt_length = len(b)
        event_bytearray = bytearray()
        type_marker = 0
        event_bytearray.extend(type_marker.to_bytes(6, 'little'))
        event_bytearray.extend(evt_header_length.to_bytes(6, 'little'))
        event_bytearray.extend(evt_header_str)
        event_bytearray.extend(evt_length.to_bytes(6, 'little'))
        event_bytearray.extend(evt_string)
        return event_bytearray

    def __get_detector_dict(self, event, det):
        is_generic_detector = isinstance(det, generic_detector.GenericDetector)
        det_dict = {
            "generic_detector": is_generic_detector,
            "channels": {},
            "stations": {}
        }
        i_station = 0
        i_channel = 0
        for station in event.get_stations():
            if not self.__is_station_already_in_file(station.get_id(), station.get_station_time()):
                if not is_generic_detector:
                    det.update(station.get_station_time())
                    station_description = det.get_station(station.get_id())
                    self.__stored_stations.append({
                        'station_id': station.get_id(),
                        'commission_time': station_description['commission_time'],
                        'decommission_time': station_description['decommission_time']
                    })
                else:
                    station_description = det.get_raw_station(station.get_id())
                    self.__stored_stations.append({
                        'station_id': station.get_id()
                    })
                det_dict['stations'][str(i_station)] = station_description
                i_station += 1
            for channel in station.iter_channels():
                if not self.__is_channel_already_in_file(station.get_id(), channel.get_id(), station.get_station_time()):
                    if not is_generic_detector:
                        channel_description = det.get_channel(station.get_id(), channel.get_id())
                        self.__stored_channels.append({
                            'station_id': station.get_id(),
                            'channel_id': channel.get_id(),
                            'commission_time': channel_description['commission_time'],
                            'decommission_time': channel_description['decommission_time']
                        })
                    else:
                        channel_description = det.get_raw_channel(station.get_id(), channel.get_id())
                        self.__stored_channels.append({
                          'station_id': station.get_id(),
                          'channel_id': channel.get_id()
                        })
                    det_dict['channels'][str(i_channel)] = channel_description
                    i_channel += 1
            # If we have a genericDetector, the default station may not be in the event.
            # In that case, we have to add it manually to make sure it ends up in the file
            if is_generic_detector:
                for reference_station_id in det.get_reference_station_ids():
                    if not self.__is_station_already_in_file(reference_station_id, None):
                        station_description = det.get_raw_station(reference_station_id)
                        self.__stored_stations.append({
                            'station_id': reference_station_id
                        })
                        det_dict['stations'][str(i_station)] = station_description
                        i_station += 1
                        for channel_id in det.get_channel_ids(reference_station_id):
                            if not self.__is_channel_already_in_file(reference_station_id, channel_id, None):
                                channel_description = det.get_raw_channel(reference_station_id, channel_id)
                                det_dict['channels'][str(i_channel)] = channel_description
                                self.__stored_channels.append({
                                    'station_id': reference_station_id,
                                    'channel_id': channel_id
                                })
                                i_channel += 1
        if i_station == 0 and i_channel == 0:  # All stations and channels have already been saved
            return None
        else:
            return det_dict

    def __get_detector_bytearray(self, detector_dict):
        detector_string = pickle.dumps(detector_dict, protocol=4)
        b = bytearray()
        b.extend(detector_string)
        detector_length = len(b)
        detector_bytearray = bytearray()
        type_marker = 1
        detector_bytearray.extend(type_marker.to_bytes(6, 'little'))
        detector_bytearray.extend(detector_length.to_bytes(6, 'little'))
        detector_bytearray.extend(detector_string)
        return detector_bytearray

    def __is_station_already_in_file(self, station_id, station_time):
        for entry in self.__stored_stations:
            if entry['station_id'] == station_id:
                # if there is no commission and decommission time it is a generic detector and we don't have to check
                if 'commission_time' not in entry.keys() or 'decommission_time' not in entry.keys() or station_time is None:
                    return True
                # it's a normal detector and we have to check commission/decommission times
                if entry['commission_time'] < station_time < entry['decommission_time']:
                    return True
        return False

    def __is_channel_already_in_file(self, station_id, channel_id, station_time):
        for entry in self.__stored_channels:
            if entry['station_id'] == station_id and entry['channel_id'] == channel_id:
                if 'commission_time' not in entry.keys() or 'decommission_time' not in entry.keys() or station_time is None:
                    return True
                # it's a normal detector and we have to check commission/decommission times
                if entry['commission_time'] < station_time < entry['decommission_time']:
                    return True
        return False

    def __get_detector_changes_byte_array(self, event, det):
        changes = det.get_station_properties_for_event(event.get_run_number(), event.get_id())
        if len(changes) == 0:
            return None
        changes_string = pickle.dumps(changes, protocol=4)
        b = bytearray()
        b.extend(changes_string)
        changes_length = len(b)
        changes_bytearray = bytearray()
        type_marker = 2
        changes_bytearray.extend(type_marker.to_bytes(6, 'little'))
        changes_bytearray.extend(changes_length.to_bytes(6, 'little'))
        changes_bytearray.extend(changes_string)
        return changes_bytearray

    def __check_for_duplicate_ids(self, run_number, event_id):
        """"
        Checks if an event with the same ID and run number has already been written to the file
        and throws an error if that is the case.
        """
        if(self.__check_for_duplicates):
            if [run_number, event_id] in self.__event_ids_and_runs:
                raise ValueError("An event with ID {} and run number {} already exists in the file\nif you don't want unique event ids enforced you can turn it of by passing `check_for_duplicates=True` to the begin method.".format(event_id, run_number))
        return

    def end(self):
        if(hasattr(self, "__fout")):
            self.__fout.close()
        return self.__number_of_events
