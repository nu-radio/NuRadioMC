from __future__ import absolute_import, division, print_function, unicode_literals
import NuRadioReco.framework.event
import NuRadioReco.detector.detector
import NuRadioReco.detector.generic_detector
import NuRadioReco.modules.io.event_parser_factory
import numpy as np
import logging
import time

VERSION = 2
VERSION_MINOR = 2


class NuRadioRecoio(object):

    def __init__(self, filenames, parse_header=True, parse_detector=True, fail_on_version_mismatch=True,
                 fail_on_minor_version_mismatch=False,
                 max_open_files=10, log_level=None, buffer_size=104857600):
        """
        Initialize NuRadioReco io

        Parameters
        ----------
        filenames: string or list of strings
            the input file/files
        parse_header: boolean
            If True, the event headers are parsed and can be accessed through
            the get_header() function
        parse_detector: boolean
            If True, detector information in the files is parsed and can be
            accessed through the get_detector() function
        fail_on_version_mismatch: boolean
            Controls if the module should try to read files with a different major version
        fail_on_minor_version_mismatch: boolean
            Controls if the module should try to read files with a different minor version
        max_open_files: int
            the maximum number of files that remain open simultaneously
        log_level: None or log level
            the log level of this class
        buffer_size: int
            the size of the read buffer in bytes (default 100MB)
        """
        if(not isinstance(filenames, list)):
            filenames = [filenames]
        self.__file_scanned = False
        self.logger = logging.getLogger('NuRadioReco.NuRadioRecoio')
        self.logger.info("initializing NuRadioRecoio with file {}".format(filenames))
        t = time.time()
        if log_level is not None:
            self.logger.setLevel(log_level)
        # Initialize attributes
        self._filenames = None
        self.__n_events = None
        self._bytes_start_header = None
        self._bytes_length_header = None
        self._bytes_start = None
        self._bytes_length = None
        self.__event_ids = None
        self.__open_files = None
        self._detector_dicts = None
        self.__detectors = None
        self._event_specific_detector_changes = None
        self.__event_headers = None
        self._current_event_id = None
        self._current_run_number = None
        self.__fail_on_version_mismatch = fail_on_version_mismatch
        self.__fail_on_minor_version_mismatch = fail_on_minor_version_mismatch
        self.__parse_header = parse_header
        self.__parse_detector = parse_detector
        self.__read_lock = False
        self.__max_open_files = max_open_files
        self.__buffer_size = buffer_size
        self.openFile(filenames)
        self._current_file_id = 0
        self.logger.info("... finished in {:.0f} seconds".format(time.time() - t))

    def _get_file(self, iF):
        if(iF not in self.__open_files):
            self.logger.debug("file {} is not yet open, opening file".format(iF))
            self.__open_files[iF] = {}
            self.__open_files[iF]['file'] = open(self._filenames[iF], 'rb', buffering=self.__buffer_size)  # 100 MB buffering
            self.__open_files[iF]['time'] = time.time()
            self.__check_file_version(iF)
            if(len(self.__open_files) > self.__max_open_files):
                self.logger.debug("more than {} file are open, closing oldest file".format(self.__max_open_files))
                tnow = time.time()
                iF_close = 0
                for key, value in self.__open_files.items():
                    if(value['time'] < tnow):
                        tnow = value['time']
                        iF_close = key
                self.logger.debug("closing file {} that was opened at {}".format(iF_close, tnow))
                self.__open_files[iF_close]['file'].close()
                del self.__open_files[iF_close]
        return self.__open_files[iF]['file']

    def __check_file_version(self, iF):
        self.__file_version = int.from_bytes(self._get_file(iF).read(6), 'little')
        self.__file_version_minor = int.from_bytes(self._get_file(iF).read(6), 'little')
        if(self.__file_version != VERSION):
            self.logger.error(
                "data file not readable. File has version {}.{} but current version is {}.{}".format(
                    self.__file_version,
                    self.__file_version_minor,
                    VERSION,
                    VERSION_MINOR
                )
            )
            if(self.__fail_on_version_mismatch):
                raise IOError
        if(self.__file_version_minor != VERSION_MINOR):
            self.logger.error(
                "data file might not readable. File has version {}.{} but current version is {}.{}".format(
                    self.__file_version,
                    self.__file_version_minor,
                    VERSION,
                    VERSION_MINOR
                )
            )
            if(self.__fail_on_minor_version_mismatch):
                raise IOError
        self.__scan_files = NuRadioReco.modules.io.event_parser_factory.scan_files_function(self.__file_version, self.__file_version_minor)
        self.__iter_events = NuRadioReco.modules.io.event_parser_factory.iter_events_function(self.__file_version, self.__file_version_minor)

    def openFile(self, filenames):
        self._filenames = filenames
        self.__n_events = 0
        self.__event_ids = []
        self._bytes_start_header = [[]]
        self._bytes_length_header = [[]]
        self._bytes_start = [[]]
        self._bytes_length = [[]]
        self.__open_files = {}
        self._detector_dicts = {}
        self.__detectors = {}
        self._event_specific_detector_changes = {}

        self.__event_headers = {}
        if(self.__parse_header):
            self.__scan_files()

    def close_files(self):
        for f in self.__open_files.values():
            f['file'].close()

    def get_filenames(self):
        return self._filenames

    def _parse_event_header(self, evt_header):
        from NuRadioReco.framework.parameters import stationParameters as stnp
        self.__event_ids.append(evt_header['event_id'])
        for station_id, station in evt_header['stations'].items():
            if station_id not in self.__event_headers:
                self.__event_headers[station_id] = {}
            for key, value in station.items():
                # treat sim_station differently
                if(key == 'sim_station'):
                    pass
                else:
                    if key not in self.__event_headers[station_id]:
                        self.__event_headers[station_id][key] = []
                    if(key == stnp.station_time):
                        import astropy.time
                        if value is not None:
                            if value.format == 'datetime':
                                time_strings = str(value).split(' ')
                                station_time = astropy.time.Time('{}T{}'.format(time_strings[0], time_strings[1]), format='isot')
                            else:
                                station_time = time
                        else:
                            station_time = None
                        self.__event_headers[station_id][key].append(station_time)
                    else:
                        self.__event_headers[station_id][key].append(value)

    def __scan_files(self):
        current_byte = 12  # skip datafile header
        iF = 0
        while True:
            self._get_file(iF).seek(current_byte)
            continue_loop, iF, current_byte = self.__scan_files(self, iF, current_byte)
            if not continue_loop:
                break

        self.__event_ids = np.array(self.__event_ids)
        self.__file_scanned = True

        # compute number of events
        n = 0
        for b in self._bytes_start:
            n += len(b)
        self.__n_events = n

        # convert lists to numpy arrays for convenience
        for station_id, station in self.__event_headers.items():
            for key, value in station.items():
                self.__event_headers[station_id][key] = np.array(value)

    def get_header(self):
        if(not self.__file_scanned):
            self.__scan_files()
        return self.__event_headers

    def get_event_ids(self):
        """
        returns a list of (run, eventid) tuples of all events contained in the data file
        """
        if(not self.__file_scanned):
            self.__scan_files()
        return self.__event_ids

    def get_event_i(self, event_number):
        while(self.__read_lock):
            time.sleep(1)
            self.logger.debug("read lock waiting 1ms")
        self.__read_lock = True

        if(not self.__file_scanned):
            self.__scan_files()
        if(event_number < 0 or event_number >= self.get_n_events()):
            self.logger.error('event number {} out of bounds, only {} present in file'.format(event_number, self.get_n_events()))
            self.__read_lock = False
            return None
        # determine in which file event i is
        istart = 0
        file_id = 0
        for iF in range(len(self._filenames)):
            istop = istart + len(self._bytes_start[iF])
            if((event_number >= istart) and (event_number < istop)):
                file_id = iF
                event_id = event_number - istart
                break
            else:
                istart = istop

        self._get_file(file_id).seek(self._bytes_start[file_id][event_id])
        evtstr = self._get_file(file_id).read(self._bytes_length[file_id][event_id])
        event = NuRadioReco.framework.event.Event(0, 0)
        event.deserialize(evtstr)
        self.__read_lock = False
        self._current_file_id = file_id
        self._current_event_id = event.get_id()
        self._current_run_number = event.get_run_number()
        # # If the event file contains a detector description that is a
        # # generic detector, it might have event-specific properties and we
        # # need to set the detector to the right event
        if self._current_file_id in self.__detectors.keys():
            if 'generic_detector' in self._detector_dicts[self._current_file_id]:
                if self._detector_dicts[self._current_file_id]['generic_detector']:
                    self.__detectors[self._current_file_id].set_event(self._current_run_number, self._current_event_id)
        return event

    def get_event(self, event_id):
        if(not self.__file_scanned):
            self.__scan_files()
        mask = (self.__event_ids[:, 0] == event_id[0]) & (self.__event_ids[:, 1] == event_id[1])
        if(np.sum(mask) == 0):
            self.logger.error('event number {} not found in file'.format(event_id))
            return None
        elif(np.sum(mask) > 1):
            self.logger.warning(f"{np.sum(mask):d} events with the same run event id pair found. Returning first occurence.")
        self._current_run_number = event_id[0]
        self._current_event_id = event_id[1]
        i = np.argwhere(mask)[0][0]
        return self.get_event_i(i)

    def get_events(self):
        self._current_file_id = 0
        self._get_file(self._current_file_id).seek(12)  # skip file header
        for event in self.__iter_events(self):
            self._current_event_id = event.get_id()
            self._current_run_number = event.get_run_number()
            if self._current_file_id in self.__detectors.keys():
                if 'generic_detector' in self._detector_dicts[self._current_file_id]:
                    if self._detector_dicts[self._current_file_id]['generic_detector']:
                        self.__detectors[self._current_file_id].set_event(self._current_run_number, self._current_event_id)
            yield event

    def get_detector(self):
        """
        If parse_detector was set True in the __init__() function, this function return
        the detector description (assuming there is one in the files). If several
        files with different detectors are read, the detector for the last returned
        event is given.
        """
        # Check if detector object for current file already exists
        if self._current_file_id not in self.__detectors.keys():
            # Detector object for current file does not exist, so we create it
            if self._current_file_id not in self._detector_dicts:
                self.__scan_files()  # Maybe we just forgot to scan the file
                if self._current_file_id not in self._detector_dicts:
                    raise AttributeError('The current file does not contain a detector description.')
            detector_dict = self._detector_dicts[self._current_file_id]
            if 'generic_detector' in detector_dict.keys():
                if detector_dict['generic_detector']:
                    # Detector is a generic detector, so we have to consider default
                    # station/channel and event-specific changes
                    self.__detectors[self._current_file_id] = NuRadioReco.detector.generic_detector.GenericDetector.__new__(NuRadioReco.detector.generic_detector.GenericDetector)
                    # the use of default_station and default_channel is deprecated. Allow to set it for now, to ensure backward compatibility
                    if 'default_station' not in detector_dict:
                        detector_dict['default_station'] = None
                    if 'default_channel' not in detector_dict:
                        detector_dict['default_channel'] = None
                    self.__detectors[self._current_file_id].__init__(source='dictionary', json_filename='', dictionary=detector_dict, default_station=detector_dict['default_station'], default_channel=detector_dict['default_channel'])
                    if self._current_file_id in self._event_specific_detector_changes.keys():
                        for change in self._event_specific_detector_changes[self._current_file_id]:
                            self.__detectors[self._current_file_id].add_station_properties_for_event(
                                properties=change['properties'],
                                station_id=change['station_id'],
                                run_number=change['run_number'],
                                event_id=change['event_id']
                            )
                    self.__detectors[self._current_file_id].set_event(self._current_run_number, self._current_event_id)
                    return self.__detectors[self._current_file_id]
            # Detector is a normal detector
            self.__detectors[self._current_file_id] = NuRadioReco.detector.detector.Detector.__new__(NuRadioReco.detector.detector.Detector)
            self.__detectors[self._current_file_id].__init__(source='dictionary', json_filename='', dictionary=self._detector_dicts[self._current_file_id])
        # Detector object for current file already exists. If it is a generic detector,
        # we update it to the run number and ID of the last event that was requested
        # (in case there are event-specific changes) and return  it
        if 'generic_detector' in self._detector_dicts[self._current_file_id].keys():
            if self._detector_dicts[self._current_file_id]['generic_detector']:
                self.__detectors[self._current_file_id].set_event(self._current_run_number, self._current_event_id)
        return self.__detectors[self._current_file_id]

    def get_n_events(self):
        if(not self.__file_scanned):
            self.__scan_files()
        return self.__n_events
