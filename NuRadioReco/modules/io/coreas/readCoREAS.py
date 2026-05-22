"""
This module can be used to read in all simulated "observers" from a CoREAS simulation and return within a single
event/station/sim_station object. This is useful for air shower array experiments like Auger, LOFAR or SKA.
However, it is important to stress that this module will return a single `station` object. If you want to organize
the simulated electric fields (observers) within multiple stations you have to do it yourself.
"""

import NuRadioReco.framework.event
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.modules.io.coreas import coreas
import time
import os

import logging
logger = logging.getLogger('NuRadioReco.coreas.readCoREAS')


class readCoREAS:

    def __init__(self):
        self.__t = 0
        self.__t_event_structure = 0
        self.__t_per_event = 0
        self.__input_files = None
        self.__current_input_file = None
        self.__ascending_run_and_event_number = None

    def begin(self, input_files, logger_level=logging.NOTSET, set_ascending_run_and_event_number=False):
        """
        begin method

        initialize readCoREAS module

        Parameters
        ----------
        input_files: input files
            list of coreas hdf5 files
        det: genericDetector object
            If a genericDetector is passed, the stations from the CoREAS file
            will be added to it and the run method returns both the event and
            the detector
        logger_level: string or logging variable
            Set verbosity level for logger (default: logging.NOTSET)
        set_ascending_run_and_event_number: bool
            If set to True the run number and event id is set to
            self.__ascending_run_and_event_number instead of beeing taken
            from the simulation file. The value is increases monoton.
            This can be used to avoid ambiguities values (default: False)
        """
        self.__input_files = input_files
        self.__current_input_file = 0
        logger.setLevel(logger_level)
        self.__ascending_run_and_event_number = 1 if set_ascending_run_and_event_number else 0

    @register_run()
    def run(self, declination=None):
        """
        Reads in CoREAS file(s) and returns one event containing all simulated observer positions as stations.

        Parameters
        ----------
        declination: float, default=None
            The declination of the magnetic field to use when reading in the CoREAS file

        Yields
        ------
        evt: `NuRadioReco.framework.event.Event`
            The event containing the simulated observer as sim. stations.
        """
        while self.__current_input_file < len(self.__input_files):
            t = time.time()
            t_per_event = time.time()

            filename = self.__input_files[self.__current_input_file]
            filesize = os.path.getsize(filename)
            if filesize < 18456 * 2:  # based on the observation that a file with such a small filesize is corrupt
                logger.warning(
                    "file {} seems to be corrupt, skipping to next file".format(filename))
                self.__current_input_file += 1
                continue

            logger.info(f'Reading {self.__input_files[self.__current_input_file]} ...')

            corsika_evt = coreas.read_CORSIKA7(self.__input_files[self.__current_input_file], declination=declination)

            if self.__ascending_run_and_event_number:
                evt = NuRadioReco.framework.event.Event(
                    self.__ascending_run_and_event_number, self.__ascending_run_and_event_number)
                self.__ascending_run_and_event_number += 1
            else:
                evt = NuRadioReco.framework.event.Event(corsika_evt.get_run_number(), corsika_evt.get_id())

            # create sim shower, core is already set in read_CORSIKA7()
            sim_shower = coreas.create_sim_shower(corsika_evt)
            evt.set_event_time(corsika_evt.get_event_time())
            evt.add_sim_shower(sim_shower)

            self.__t_per_event += time.time() - t_per_event
            self.__t += time.time() - t

            self.__current_input_file += 1
            yield evt

    def get_event(self, declination=None):
        """ Reads the next event """
        return next(self.run(declination=declination))


    def end(self):
        from datetime import timedelta
        dt = timedelta(seconds=self.__t)
        logger.info("total time used by this module is {}".format(dt))
        logger.info("\tcreate event structure {}".format(
            timedelta(seconds=self.__t_event_structure)))
        logger.info("per event {}".format(
            timedelta(seconds=self.__t_per_event)))

        return dt