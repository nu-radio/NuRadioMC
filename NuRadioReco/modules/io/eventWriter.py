from __future__ import absolute_import, division, print_function, unicode_literals
import pickle
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.modules.io.NuRadioRecoio import VERSION, VERSION_MINOR
import logging
import datetime
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.detector import generic_detector
logger = logging.getLogger("eventWriter")


def get_header(evt):
    header = {}
    header['stations'] = {}
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

    def begin(self, filename, max_file_size=1024, check_for_duplicates=False):
        """
        begin method

        Parameters
        ----------
        max_file_size: maximum file size in Mbytes
                    (if the file exceeds the maximum file the output will be split into another file)
        check_for_duplicates: bool (default False)
            if True, the event writer raises an exception when an event with a (run,eventid) pair is written that is already
            present in the data file
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

    @register_run()
    def run(self, evt, det=None, mode='full'):
        """
        writes NuRadioReco event into a file

        Parameters
        ----------
        evt: NuRadioReco event object
        det: detector object
            If a detector object is passed, the detector description for the
            events is written in the file as well
        mode: string
            specifies the output mode:
            * 'full' (default): the full event content is written to disk
            * 'mini': only station traces are written to disc
            * 'micro': no traces are written to disc
        """

        if(mode not in ['full', 'mini', 'micro']):
            logger.error("output mode must be one of ['full', 'mini', 'micro'] but is {}".format(mode))
            raise NotImplementedError
        self.__check_for_duplicate_ids(evt.get_run_number(), evt.get_id())
        if not self.__header_written:
            self.__write_fout_header()

        event_bytearray = self.__get_event_bytearray(evt, mode)
        self.__fout.write(event_bytearray)
        self.__current_file_size += event_bytearray.__sizeof__()
        self.__number_of_events += 1
        self.__event_ids_and_runs.append([evt.get_run_number(), evt.get_id()])

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

        if(self.__current_file_size > self.__max_file_size):
            logger.info("current output file exceeds max file size -> closing current output file and opening new one")
            self.__current_file_size = 0
            self.__fout.close()
            self.__number_of_files += 1
            # self.__filename = "{}_part{:02d}".format(self.__filename, self.__number_of_files)
            self.__stored_stations = []
            self.__stored_channels = []
            self.__event_ids_and_runs = []
            self.__header_written = False

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
        if is_generic_detector:
            det_dict['default_station'] = det.get_default_station_id()
            det_dict['default_channel'] = det.get_default_channel_id()
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
                    else:
                        channel_description = det.get_raw_channel(station.get_id(), channel.get_id())
                    det_dict['channels'][str(i_channel)] = channel_description
                    self.__stored_channels.append({
                        'station_id': station.get_id(),
                        'channel_id': channel.get_id(),
                        'commission_time': channel_description['commission_time'],
                        'decommission_time': channel_description['decommission_time']
                    })
                    i_channel += 1
        # If we have a genericDetector, the default station may not be in the event.
        # In that case, we have to add it manually to make sure it ends up in the file
        if is_generic_detector:
            if not self.__is_station_already_in_file(det.get_default_station_id(), None):
                station_description = det.get_raw_station(det.get_default_station_id())
                self.__stored_stations.append({
                'station_id': station.get_id()
                })
                det_dict['stations'][str(i_station)] = station_description
                for channel_id in det.get_channel_ids(det.get_default_station_id()):
                    if not self.__is_channel_already_in_file(det.get_default_station_id(), channel_id, None):
                        channel_description = det.get_raw_channel(det.get_default_station_id(), channel_id)
                        det_dict['channels'][str(i_channel)] = channel_description
                        self.__stored_channels.append({
                            'station_id': det.get_default_station_id(),
                            'channel_id': channel_id,
                            'commission_time': channel_description['commission_time'],
                            'decommission_time': channel_description['decommission_time']
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
                if entry['commission_time'] < station_time and entry['decommission_time'] > station_time:
                    return True
        return False

    def __is_channel_already_in_file(self, station_id, channel_id, station_time):
        for entry in self.__stored_channels:
            if entry['station_id'] == station_id and entry['channel_id'] == channel_id:
                if 'commission_time' not in entry.keys() or 'decommission_time' not in entry.keys() or station_time is None:
                    return True
                if entry['commission_time'] < station_time and entry['decommission_time'] > station_time:
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
