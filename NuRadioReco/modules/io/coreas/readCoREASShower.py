import NuRadioReco.framework.event
import NuRadioReco.framework.station
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.modules.io.coreas import coreas
import numpy as np
import time
import os
import logging
logger = logging.getLogger('NuRadioReco.coreas.readCoREASShower')


class readCoREASShower:
    """
    This module can be used to read in all simulated "observers" from a CoREAS simulation and return them as stations.
    This is in particular useful for air shower array experiments like Auger, LOFAR or SKA. However, it is important
    to stress that this module will return a `station` object per simulated observer. That fits well the terminology
    used in Auger were a "Station" is a dual-polerized antenna. However, for other experiments like LOFAR or SKA where
    a "Station" is a cluster of (dual-polerized) antennas, the user has to be aware that this module will return a `station`
    object per antenna.
    """

    def __init__(self):
        self.__t = 0
        self.__t_event_structure = 0
        self.__t_per_event = 0
        self.__input_files = None
        self.__current_input_file = None
        self.__det = None
        self.__ascending_run_and_event_number = None

    def begin(self, input_files, det=None, logger_level=logging.NOTSET, set_ascending_run_and_event_number=False):
        """
        begin method

        initialize readCoREASShower module

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
        self.__det = det
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
        evt : `NuRadioReco.framework.event.Event`
            The event containing the simulated observer as sim. stations.

        det : `NuRadioReco.detector.generic_detector.GenericDetector`
            Optional, only if a detector description is passed to the begin method.
            Contains the detector description with the on-the-fly added stations.
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

            logger.info('Reading %s ...' %
                        self.__input_files[self.__current_input_file])

            corsika_evt = coreas.read_CORSIKA7(self.__input_files[self.__current_input_file], declination=declination)

            if self.__ascending_run_and_event_number:
                evt = NuRadioReco.framework.event.Event(self.__ascending_run_and_event_number,
                                                        self.__ascending_run_and_event_number)
                self.__ascending_run_and_event_number += 1
            else:
                evt = NuRadioReco.framework.event.Event(corsika_evt.get_run_number(), corsika_evt.get_id())

            # create sim shower, core is already set in read_CORSIKA7()
            sim_shower = coreas.create_sim_shower(corsika_evt)
            evt.set_event_time(corsika_evt.get_event_time())
            evt.add_sim_shower(sim_shower)

            # add simulated pulses as sim station
            corsika_efields = corsika_evt.get_station(0).get_sim_station().get_electric_fields()
            for station_id, corsika_efield in enumerate(corsika_efields):
                station = NuRadioReco.framework.station.Station(station_id)
                sim_station = coreas.create_sim_station(station_id, corsika_evt)
                efield_trace = corsika_efield.get_trace()
                efield_sampling_rate = corsika_efield.get_sampling_rate()
                efield_times = corsika_efield.get_times()

                if self.__det is None:
                    channel_ids = [0, 1]
                else:
                    if self.__det.has_station(station_id):
                        channel_ids = self.__det.get_channel_ids(station_id)
                    else:
                        channel_ids = self.__det.get_channel_ids(self.__det.get_reference_station_ids()[0])

                coreas.add_electric_field_to_sim_station(
                    sim_station, channel_ids, efield_trace, efield_times[0],
                    sim_shower.get_parameter(shp.zenith), sim_shower.get_parameter(shp.azimuth),
                    efield_sampling_rate)

                station.set_sim_station(sim_station)
                evt.set_station(station)

                if self.__det is not None:
                    efield_pos = corsika_efield.get_position()

                    if not self.__det.has_station(station_id):
                        self.__det.add_generic_station({
                            'station_id': station_id,
                            'pos_easting': efield_pos[0],
                            'pos_northing': efield_pos[1],
                            'pos_altitude': efield_pos[2],
                            'reference_station': self.__det.get_reference_station_ids()[0]
                        })
                    else:
                        self.__det.add_station_properties_for_event({
                            'pos_easting': efield_pos[0],
                            'pos_northing': efield_pos[1],
                            'pos_altitude': efield_pos[2]
                        }, station_id, evt.get_run_number(), evt.get_id())

            self.__t_per_event += time.time() - t_per_event
            self.__t += time.time() - t

            self.__current_input_file += 1
            if self.__det is None:
                yield evt
            else:
                self.__det.set_event(evt.get_run_number(), evt.get_id())
                yield evt, self.__det

    def end(self):
        from datetime import timedelta
        dt = timedelta(seconds=self.__t)
        logger.info("total time used by this module is {}".format(dt))
        logger.info("\tcreate event structure {}".format(
            timedelta(seconds=self.__t_event_structure)))
        logger.info("per event {}".format(
            timedelta(seconds=self.__t_per_event)))
        return dt