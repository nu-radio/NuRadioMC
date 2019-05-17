from __future__ import absolute_import, division, print_function, unicode_literals
import NuRadioReco.framework.event
import numpy as np
import logging
try:
    import cPickle as pickle
except ImportError:
    import pickle
import time
logger = logging.getLogger('NuRadioRecoio')

VERSION = 2
VERSION_MINOR = 1


class NuRadioRecoio(object):

    def __init__(self, filenames, parse_header=True, fail_on_version_mismatch=True,
                 fail_on_minor_version_mismatch=False):
        if(not isinstance(filenames, list)):
            filenames = [filenames]
        self.__file_scanned = False
        logger.info("initializing NuRadioRecoio with file {}".format(filenames))
        t = time.time()
        self.__fail_on_version_mismatch = fail_on_version_mismatch
        self.__fail_on_minor_version_mismatch = fail_on_minor_version_mismatch
        self.__parse_header = parse_header
        self.__read_lock = False
        self.openFile(filenames)
        logger.info("... finished in {:.0f} seconds".format(time.time() - t))

    def openFile(self, filenames):
        self.__filenames = filenames
        self.__n_events = 0
        self.__event_ids = []
        self.__bytes_start_header = [[]]
        self.__bytes_length_header = [[]]
        self.__bytes_start = [[]]
        self.__bytes_length = [[]]
        self.__fin = []
        for iF, filename in enumerate(filenames):
            self.__fin.append(open(filename, 'rb'))
            self.__file_version = int(self.__fin[iF].read(6), 16)
            self.__file_version_minor = int(self.__fin[iF].read(6), 16)
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

        self.__event_headers = {}
        if(self.__parse_header):
            self.__scan_files()

    def close_files(self):
        for f in self.__fin:
            f.close()

    def get_filenames(self):
        return self.__filenames

    def __parse_event_header(self, evt_header):
        self.__event_ids.append(evt_header['event_id'])
        for station_id, station in evt_header['stations'].iteritems():
            if station_id not in self.__event_headers:
                self.__event_headers[station_id] = {}
            for key, value in station.iteritems():
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
            self.__fin[iF].seek(current_byte)
            bytes_to_read_hex = self.__fin[iF].read(6)
            if(bytes_to_read_hex == ''):
                # we are at the end of the file
                if(iF < (len(self.__fin) - 1)):  # are there more files to be parsed?
                    iF += 1
                    current_byte = 12  # skip datafile header
                    self.__fin[iF].seek(current_byte)
                    bytes_to_read_hex = self.__fin[iF].read(6)
                    self.__bytes_start_header.append([])
                    self.__bytes_length_header.append([])
                    self.__bytes_start.append([])
                    self.__bytes_length.append([])
                else:
                    break
            current_byte += 6
            bytes_to_read = int(bytes_to_read_hex, 16)
            self.__bytes_start_header[iF].append(current_byte)
            self.__bytes_length_header[iF].append(bytes_to_read)
            current_byte += bytes_to_read

            evt_header = pickle.loads(self.__fin[iF].read(bytes_to_read))
            self.__parse_event_header(evt_header)

            self.__fin[iF].seek(current_byte)
            bytes_to_read_hex = self.__fin[iF].read(6)
            current_byte += 6
            bytes_to_read = int(bytes_to_read_hex, 16)
            self.__bytes_start[iF].append(current_byte)
            self.__bytes_length[iF].append(bytes_to_read)
            current_byte += bytes_to_read
#             print("reading event {} with length {} from byte {} onwards".format(len(self.__bytes_length), bytes_to_read, self.__bytes_start[-1]))
        self.__event_ids = np.array(self.__event_ids)
        self.__file_scanned = True

        # compute number of events
        n = 0
        for b in self.__bytes_start:
            n += len(b)
        self.__n_events = n

        # convert lists to numpy arrays for convenience
        for station_id, station in self.__event_headers.iteritems():
            for key, value in station.iteritems():
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
        for iF in range(len(self.__fin)):
            istop = istart + len(self.__bytes_start[iF])
            if((event_number >= istart) and (event_number < istop)):
                file_id = iF
                event_id = event_number - istart
                break
            else:
                istart = istop

        self.__fin[file_id].seek(self.__bytes_start[file_id][event_id])
        evtstr = self.__fin[file_id].read(self.__bytes_length[file_id][event_id])
        event = NuRadioReco.framework.event.Event(0, 0)
        event.deserialize(evtstr)
        self.__read_lock = False
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
        iF = 0
        self.__fin[iF].seek(12)  # skip file header
        while True:
            bytes_to_read_hex = self.__fin[iF].read(6)
            if(bytes_to_read_hex == ''):
                # we are at the end of the file
                if(iF < (len(self.__fin) - 1)):  # are there more files to be parsed?
                    iF += 1
                    self.__fin[iF].seek(12)  # skip datafile header
                    bytes_to_read_hex = self.__fin[iF].read(6)
                else:
                    break
            bytes_to_read = int(bytes_to_read_hex, 16)
            evt_header_str = self.__fin[iF].read(bytes_to_read)

            bytes_to_read_hex = self.__fin[iF].read(6)
            bytes_to_read = int(bytes_to_read_hex, 16)
            evtstr = self.__fin[iF].read(bytes_to_read)
            event = NuRadioReco.framework.event.Event(0, 0)
            event.deserialize(evtstr)
            yield event

    def get_n_events(self):
        if(not self.__file_scanned):
            self.__scan_files()
        return self.__n_events

