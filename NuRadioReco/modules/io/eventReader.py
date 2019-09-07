from __future__ import absolute_import, division, print_function, unicode_literals
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.modules.io import NuRadioRecoio
import logging
logger = logging.getLogger('eventReader')


class eventReader:
    """
    read events from file
    """

    def begin(self, filename, log_level=logging.WARNING):
        self.__fin = NuRadioRecoio.NuRadioRecoio(filename, parse_header=False, log_level=log_level)

    @register_run("event")
    def run(self):
        return self.__fin.get_events()

    def end(self):
        self.__fin.close_file()

    def get_header(self):
        """
        returns the header information of all events, useful to get a quick overview of all events without 
        looping through all events
        """
        return self.__fin.get_header()