from __future__ import absolute_import, division, print_function, unicode_literals
try:
    import cPickle as pickle
except ImportError:
    import pickle
from NuRadioReco.modules.io.NuRadioRecoio import VERSION, VERSION_MINOR
import logging
from NuRadioReco.framework.parameters import stationParameters as stnp
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
        b = bytearray()
        b.extend(b'{:06x}'.format(VERSION))
        b.extend(b'{:06x}'.format(VERSION_MINOR))
        self.__fout.write(b)

    def begin(self, filename, max_file_size=1024):
        """
        begin method

        Parameters
        ----------
        max_file_size: maximum file size in Mbytes
                    (if the file exceeds the maximum file the output will be split into another file)
        """
        if filename[-4:] == '.nur':
            self.__filename = filename[:-4]
        else:
            self.__filename = filename
        if filename[-4:] == '.ari':
            logger.warning('The file ending .ari for NuRadioReco files is deprecated. Please use .nur instead.')
        self.__number_of_events = 0
        self.__fout = open('{}.nur'.format(self.__filename), 'wb')
        self.__write_fout_header()
        self.__current_file_size = 0
        self.__number_of_files = 1
        self.__max_file_size = max_file_size * 1024 * 1024  # in bytes

    def run(self, evt, mode='full'):
        """
        writes NuRadioReco event into a file

        Parameters
        ----------
        evt: NuRadioReco event object
        mode: string
            specifies the output mode:
            * 'full' (default): the full event content is written to disk
            * 'mini': only station traces are written to disc
            * 'micro': no traces are written to disc
        """

        if(mode not in ['full', 'mini', 'micro']):
            logger.error("output mode must be one of ['full', 'mini', 'micro'] but is {}".format(mode))
            raise NotImplementedError
        evt_header_str = pickle.dumps(get_header(evt), protocol=2)
        b = bytearray()
        b.extend(evt_header_str)
        evt_header_length = len(b)

        evtstr = evt.serialize(mode)
        b = bytearray()
        b.extend(evtstr)
        evt_length = len(b)

        b = bytearray()
        b.extend(b'{:06x}'.format(evt_header_length))
        b.extend(evt_header_str)
        b.extend(b'{:06x}'.format(evt_length))
        b.extend(evtstr)
        self.__fout.write(b)
        self.__current_file_size += b.__sizeof__()
        self.__number_of_events += 1
        logger.debug("current file size is {} bytes, event number {}".format(self.__current_file_size,
                     self.__number_of_events))
        if(self.__current_file_size > self.__max_file_size):
            logger.info("current output file exceeds max file size -> closing current output file and opening new one")
            self.__current_file_size = 0
            self.__fout.close()
            self.__number_of_files += 1
            self.__fout = open("{}_part{:02d}.nur".format(self.__filename, self.__number_of_files), 'wb')
            self.__write_fout_header()

    def end(self):
        self.__fout.close()
        return self.__number_of_events
