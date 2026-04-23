from __future__ import absolute_import, division, print_function, unicode_literals
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.modules.io import NuRadioRecoio
import logging
logger = logging.getLogger('NuRadioReco.eventReader')


class eventReader:
    """
    read events from file
    """

    def begin(self, filename, read_detector=False, log_level=logging.NOTSET):
        """
        Setup function for the eventReader module

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

    def end(self):
        self.__fin.close_files()

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

    def get_event_ids(self):
        """
        Return the list of ``(run_number, event_number)`` tuples available in the opened file(s).

        Useful for callers that want random access via ``get_event`` instead
        of iterating the ``run()`` generator.
        """
        return self.__fin.get_event_ids()

    def get_event(self, event_id):
        """
        Return a specific event by ``event_id=(run_number, event_number)``.

        Parameters
        ----------
        event_id: tuple of (int, int)
            ``(run_number, event_number)`` identifying the event.

        Returns
        -------
        event: NuRadioReco.framework.event.Event or None
            The requested event, or None if it is not present in the
            opened file(s).
        """
        return self.__fin.get_event(event_id)

    def get_event_i(self, event_number):
        """
        Return the Nth event in the file by 0-indexed position.

        Note: ``event_number`` here is the file-internal position, NOT
        the event's ``event_id``. Use ``get_event`` for lookup by id.
        """
        return self.__fin.get_event_i(event_number)
