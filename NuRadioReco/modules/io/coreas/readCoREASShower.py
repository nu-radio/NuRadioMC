import NuRadioReco.framework.event
import NuRadioReco.framework.station
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.modules.io.coreas import coreas
from NuRadioReco.utilities import units
from radiotools import coordinatesystems

import h5py
import numpy as np
import time
import re
import os

import logging
logger = logging.getLogger('readCoREASShower')


class readCoREASShower:

    def __init__(self):
        self.__t = 0
        self.__t_event_structure = 0
        self.__t_per_event = 0

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


    def run(self):
        """
        Reads in a CoREAS file and returns an event containing all simulated stations

        """
        while (self.__current_input_file < len(self.__input_files)):
            t = time.time()
            t_per_event = time.time()

            filesize = os.path.getsize(self.__input_files[self.__current_input_file])
            if(filesize < 18456 * 2):
                logger.warning("file {} seems to be corrupt, skipping to next file".format(
                       self.__input_files[self.__current_input_file]))
                self.__current_input_file += 1
                continue

            logger.info('Reading %s ...' % self.__input_files[self.__current_input_file])

            corsika = h5py.File(self.__input_files[self.__current_input_file], "r")
            logger.info("using coreas simulation {} with E={:2g} theta = {:.0f}".format(
                self.__input_files[self.__current_input_file], corsika['inputs'].attrs["ERANGE"][0] * units.GeV,
                corsika['inputs'].attrs["THETAP"][0]))

            f_coreas = corsika["CoREAS"]

            if self.__ascending_run_and_event_number:
                evt = NuRadioReco.framework.event.Event(self.__ascending_run_and_event_number,
                                                        self.__ascending_run_and_event_number)
                self.__ascending_run_and_event_number += 1
            else:
                evt = NuRadioReco.framework.event.Event(corsika['inputs'].attrs['RUNNR'], corsika['inputs'].attrs['EVTNR'])

            evt.__event_time = f_coreas.attrs["GPSSecs"]

            # create sim shower, no core is set since no external detector description is given
            sim_shower = coreas.make_sim_shower(corsika)
            sim_shower.set_parameter(shp.core, np.array([0, 0, f_coreas.attrs["CoreCoordinateVertical"] / 100])) # set core
            evt.add_sim_shower(sim_shower)

            # initialize coordinate transformation
            cs = coordinatesystems.cstrafo(sim_shower.get_parameter(shp.zenith), sim_shower.get_parameter(shp.azimuth),
                                           magnetic_field_vector=sim_shower.get_parameter(shp.magnetic_field_vector))

            # add simulated pulses as sim station
            for idx, (name, observer) in enumerate(f_coreas['observers'].items()):
                station_id = antenna_id(name, idx)  # returns proper station id if possible

                station = NuRadioReco.framework.station.Station(station_id)
                sim_station = coreas.make_sim_station(station_id, corsika, observer, channel_ids=[0, 1, 2])

                station.set_sim_station(sim_station)
                evt.set_station(station)
                if self.__det is not None:
                    position = observer.attrs['position']
                    antenna_position = np.zeros(3)
                    antenna_position[0], antenna_position[1], antenna_position[2] = -position[1] * units.cm, position[0] * units.cm, position[2] * units.cm
                    antenna_position = cs.transform_from_magnetic_to_geographic(antenna_position)
                    if not self.__det.has_station(station_id):
                        self.__det.add_generic_station({
                            'station_id': station_id,
                            'pos_easting': antenna_position[0],
                            'pos_northing': antenna_position[1],
                            'pos_altitude': antenna_position[2]
                        })
                    else:
                        self.__det.add_station_properties_for_event({
                            'pos_easting': antenna_position[0],
                            'pos_northing': antenna_position[1],
                            'pos_altitude': antenna_position[2]
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
        logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        logger.info("total time used by this module is {}".format(dt))
        logger.info("\tcreate event structure {}".format(timedelta(seconds=self.__t_event_structure)))
        logger.info("\per event {}".format(timedelta(seconds=self.__t_per_event)))
        return dt


def antenna_id(antenna_name, default_id):
    """
    This function parses the antenna names given in a CoREAS simulation and tries to find an ID
    It can be extended to other name patterns
    """

    if re.match("AERA_", antenna_name):
        new_id = int(antenna_name.strip("AERA_"))
        return new_id
    else:
        return default_id
