"""
Reads the pickle based ``.nur`` files (written by NuRadioReco up to version 3.1).

Deprecated: `NuRadioReco.modules.io.eventReader` reads ``.nur`` files with this module (with a warning);
convert them to parquet with ``nuradio-analyse convert``. This module will be removed in a future release.
"""

from NuRadioReco.modules.base.module import register_run
from NuRadioReco.modules.io import NuRadioRecoio
import logging

logger = logging.getLogger('NuRadioReco.eventReaderNur')


class eventReaderNur:
    """
    read events from ``.nur`` files
    """

    def __init__(self):
        self.__fin = None

    def begin(self, filename, read_detector=False, log_level=logging.NOTSET):
        """
        Setup function for the eventReaderNur module

        Parameters
        ----------
        filename: array if strings
            names of the input files
        read_detector: boolean
            If True, the eventReader will parse the detector description and event event headers
             in the event files. This is necessary to use the get_detector functions
        log_level: logging enum
        """

        self.__fin = NuRadioRecoio.NuRadioRecoio(filename, parse_header=read_detector, log_level=log_level)

    @register_run()
    def run(self):
        return self.__fin.get_events()

    def close(self):
        """Close all input files. Safe to call multiple times and before ``begin()``."""
        if self.__fin is not None:
            self.__fin.close_files()

    def end(self):
        self.close()

    def get_n_events(self):
        """Number of events in the files."""
        return self.__fin.get_n_events()

    def get_header(self):
        """
        returns the header information of all events, useful to get a quick overview of all events without
        looping through all events
        """
        return self.__fin.get_header()

    def get_detector(self):
        """
        If read_detector was set True in the begin() function, this function return
        the detector description (assuming there is one in the files). If several
        files with different detectors are read, the detector for the last returned
        event is given.
        """
        return self.__fin.get_detector()
