from __future__ import absolute_import, division, print_function, unicode_literals
import NuRadioReco.framework.event
import NuRadioReco.detector.detector
import numpy as np
import logging
import pickle
import time
logger = logging.getLogger('NuRadioRecoio')

VERSION = 2
VERSION_MINOR = 1


class NuRadioRecoio(object):

    def __init__(self, filenames, parse_header=True, parse_detector=True, fail_on_version_mismatch=True,
                 fail_on_minor_version_mismatch=False,
                 max_open_files=10, log_level=logging.WARNING):
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
        """
        if(not isinstance(filenames, list)):
            filenames = [filenames]
        self.__file_scanned = False
        logger.info("initializing NuRadioRecoio with file {}".format(filenames))
        t = time.time()
        logger.setLevel(log_level)
        self.__fail_on_version_mismatch = fail_on_version_mismatch
        self.__fail_on_minor_version_mismatch = fail_on_minor_version_mismatch
        self.__parse_header = parse_header
        self.__parse_detector = parse_detector
        self.__read_lock = False
        self.__max_open_files = max_open_files
        self.openFile(filenames)
        logger.info("... finished in {:.0f} seconds".format(time.time() - t))

    def __get_file(self, iF):
        if(iF not in self.__open_files):
            logger.debug("file {} is not yet open, opening file".format(iF))
            self.__open_files[iF] = {}
            self.__open_files[iF]['file'] = open(self.__filenames[iF], 'rb')
            self.__open_files[iF]['time'] = time.time()
            self.__check_file_version(iF)
            if(len(self.__open_files) > self.__max_open_files):
                logger.debug("more than {} file are open, closing oldest file".format(self.__max_open_files))
                tnow = time.time()
                iF_close = 0
                for key, value in self.__open_files.iteritems():
                    if(value['time'] < tnow):
                        tnow = value['time']
                        iF_close = key
                logger.debug("closing file {} that was opened at {}".format(iF_close, tnow))
                self.__open_files[iF_close]['file'].close()
                del self.__open_files[iF_close]
        return self.__open_files[iF]['file']

    def __check_file_version(self, iF):
        self.__file_version = int.from_bytes(self.__get_file(iF).read(6), 'little')
        self.__file_version_minor = int.from_bytes(self.__get_file(iF).read(6), 'little')
        if(self.__file_version != VERSION):
            logger.error("data file not readable. File has version {}.{} but current version is {}.{}".format(self.__file_version, self.__file_version_minor,
                                                                                                              VERSION, VERSION_MINOR))
            if(self.__fail_on_version_mismatch):
                raise IOError
        if(self.__file_version_minor != VERSION_MINOR):
            logger.error("data file might not readable. File has version {}.{} but current version is {}.{}".format(self.__file_version, self.__file_version_minor,
                                                                                                              VERSION, VERSION_MINOR))
            if(self.__fail_on_minor_version_mismatch):
                raise IOError

    def openFile(self, filenames):
        self.__filenames = filenames
        self.__n_events = 0
        self.__event_ids = []
        self.__bytes_start_header = [[]]
        self.__bytes_length_header = [[]]
        self.__bytes_start = [[]]
        self.__bytes_length = [[]]
        self.__open_files = {}
        self.__detector_dicts = {}
        self.__detectors = {}

        self.__event_headers = {}
        if(self.__parse_header):
            self.__scan_files()

    def close_files(self):
        for f in self.__open_files:
            f['file'].close()

    def get_filenames(self):
        return self.__filenames

    def __parse_event_header(self, evt_header):
        self.__event_ids.append(evt_header['event_id'])
        for station_id, station in evt_header['stations'].items():
            if station_id not in self.__event_headers:
                self.__event_headers[station_id] = {}
            for key, value in station.items():
                # treat sim_station differently
                if(key == 'sim_station'):
                    pass
#                     for skey, svalue in station['sim_station'].iteritems():
#                         skey = "sim_" + skey
#                         if skey not in self.__event_headers[station_id]:
#                             self.__event_headers[station_id][skey] = []
#                         self.__event_headers[station_id][skey].append(svalue)
                else:
                    if key not in self.__event_headers[station_id]:
                        self.__event_headers[station_id][key] = []
                    self.__event_headers[station_id][key].append(value)

    def __scan_files(self):
        current_byte = 12  # skip datafile header
        iF = 0
        while True:
            self.__get_file(iF).seek(current_byte)
            object_type_hex = self.__get_file(iF).read(6)
            object_type = int.from_bytes(object_type_hex, 'little')
            current_byte += 6
            bytes_to_read_hex = self.__get_file(iF).read(6)
            bytes_to_read = int.from_bytes(bytes_to_read_hex, 'little')
            if(bytes_to_read == 0):
                # we are at the end of the file
                if(iF < (len(self.__filenames) - 1)):  # are there more files to be parsed?
                    iF += 1
                    current_byte = 12  # skip datafile header
                    self.__get_file(iF).seek(current_byte)
                    object_type_hex = self.__get_file(iF).read(6)
                    object_type = int.from_bytes(object_type_hex, 'little')
                    bytes_to_read_hex = self.__get_file(iF).read(6)
                    bytes_to_read = int.from_bytes(bytes_to_read_hex, 'little')
                    self.__bytes_start_header.append([])
                    self.__bytes_length_header.append([])
                    self.__bytes_start.append([])
                    self.__bytes_length.append([])
                    current_byte += 6
                else:
                    break
            current_byte += 6
            if object_type == 0:    #object is an event
                self.__bytes_start_header[iF].append(current_byte)
                self.__bytes_length_header[iF].append(bytes_to_read)
                current_byte += bytes_to_read

                evt_header = pickle.loads(self.__get_file(iF).read(bytes_to_read))
                self.__parse_event_header(evt_header)

                self.__get_file(iF).seek(current_byte)
                bytes_to_read_hex = self.__get_file(iF).read(6)
                current_byte += 6
                bytes_to_read = int.from_bytes(bytes_to_read_hex, 'little')
                self.__bytes_start[iF].append(current_byte)
                self.__bytes_length[iF].append(bytes_to_read)
            elif object_type == 1:  #object is detector info
                detector_dict = pickle.loads(self.__get_file(iF).read(bytes_to_read))
                if iF not in self.__detector_dicts.keys():
                    self.__detector_dicts[iF] = {
                        'channels': {},
                        'stations': {}
                    }
                for station in detector_dict['stations'].values():
                    if len(self.__detector_dicts[iF]['stations'].keys()) == 0:
                        index = 0
                    else:
                        index = max(self.__detector_dicts[iF]['stations'].keys()) + 1
                    self.__detector_dicts[iF]['stations'][index] = station
                for channel in detector_dict['channels'].values():
                    if len(self.__detector_dicts[iF]['channels'].keys()) == 0:
                        index = 0
                    else:
                        index = max(self.__detector_dicts[iF]['channels'].keys()) + 1
                    self.__detector_dicts[iF]['channels'][index] = channel
            current_byte += bytes_to_read
            #print("reading event {} with length {} from byte {} onwards".format(len(self.__bytes_length[iF]), bytes_to_read, self.__bytes_start[iF][-1]))
        self.__event_ids = np.array(self.__event_ids)
        self.__file_scanned = True

        # compute number of events
        n = 0
        for b in self.__bytes_start:
            n += len(b)
        self.__n_events = n

        # convert lists to numpy arrays for convenience
        for station_id, station in self.__event_headers.items():
            for key, value in station.items():
                self.__event_headers[station_id][key] = np.array(value)
#         print(self.__event_ids, type(self.__event_ids[0]))

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
            logger.debug("read lock waiting 1ms")
        self.__read_lock = True

        if(not self.__file_scanned):
            self.__scan_files()
        if(event_number < 0 or event_number >= self.get_n_events()):
            logger.error('event number {} out of bounds, only {} present in file'.format(event_number, self.get_n_events()))
            return None
        # determine in which file event i is
        istart = 0
        file_id = 0
        for iF in range(len(self.__filenames)):
            istop = istart + len(self.__bytes_start[iF])
            if((event_number >= istart) and (event_number < istop)):
                file_id = iF
                event_id = event_number - istart
                break
            else:
                istart = istop

        self.__get_file(file_id).seek(self.__bytes_start[file_id][event_id])
        evtstr = self.__get_file(file_id).read(self.__bytes_length[file_id][event_id])
        event = NuRadioReco.framework.event.Event(0, 0)
        event.deserialize(evtstr)
        self.__read_lock = False
        self.__current_file_id = file_id
        return event

    def get_event(self, event_id):
        if(not self.__file_scanned):
            self.__scan_files()
        for i in range(self.get_n_events()):
            if self.__event_ids[i][0] == event_id[0] and self.__event_ids[i][1] == event_id[1]:
                return self.get_event_i(i)
        logger.error('event number {} not found in file'.format(event_id))
        return None

    def get_events(self):
        self.__current_file_id = 0
        self.__get_file(self.__current_file_id).seek(12)  # skip file header
        while True:
            object_type_hex = self.__get_file(self.__current_file_id).read(6)
            object_type = int.from_bytes(object_type_hex, 'little')
            bytes_to_read_hex = self.__get_file(self.__current_file_id).read(6)
            bytes_to_read = int.from_bytes(bytes_to_read_hex, 'little')
            if(bytes_to_read == 0):
                # we are at the end of the file
                if(self.__current_file_id < (len(self.__filenames) - 1)):  # are there more files to be parsed?
                    self.__current_file_id += 1
                    self.__get_file(self.__current_file_id).seek(12)  # skip datafile header
                    object_type_hex = self.__get_file(self.__current_file_id).read(6)
                    object_type = int.from_bytes(object_type_hex, 'little')
                    bytes_to_read_hex = self.__get_file(self.__current_file_id).read(6)
                    bytes_to_read = int.from_bytes(bytes_to_read_hex, 'little')
                else:
                    break
            if object_type == 0:
                evt_header_str = self.__get_file(self.__current_file_id).read(bytes_to_read)
                bytes_to_read_hex = self.__get_file(self.__current_file_id).read(6)
                bytes_to_read = int.from_bytes(bytes_to_read_hex, 'little')
                evtstr = self.__get_file(self.__current_file_id).read(bytes_to_read)
                event = NuRadioReco.framework.event.Event(0, 0)
                event.deserialize(evtstr)
                yield event
            elif object_type == 1:
                self.__get_file(self.__current_file_id).read(bytes_to_read)

    def get_detector(self):
        """
        If parse_detector was set True in the __init__() function, this function return
        the detector description (assuming there is one in the files). If several
        files with different detectors are read, the detector for the last returned
        event is given.
        """
        if self.__current_file_id not in self.__detectors.keys():
            self.__detectors[self.__current_file_id] = NuRadioReco.detector.detector.Detector.__new__(NuRadioReco.detector.detector.Detector)
            self.__detectors[self.__current_file_id].__init__(source='dictionary', json_filename='', dictionary=self.__detector_dicts[self.__current_file_id])
        return self.__detectors[self.__current_file_id]
    def get_n_events(self):
        if(not self.__file_scanned):
            self.__scan_files()
        return self.__n_events
