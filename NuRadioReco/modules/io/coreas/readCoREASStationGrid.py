from NuRadioReco.modules.base.module import register_run
import h5py
import NuRadioReco.framework.event
import NuRadioReco.framework.station
import NuRadioReco.framework.radio_shower
from radiotools import coordinatesystems as cstrafo
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.modules.io.coreas import coreas
from NuRadioReco.utilities import units
import numpy as np
import numpy.random
import logging
import time
import os


class readCoREAS:
    """
    coreas input module for fixed grid of stations.
    This module distributes core positions randomly within a user defined area and calculates the electric field
    at the detector positions as specified in the detector description by choosing the closest antenna
    of the star shape pattern simulation
    """

    def __init__(self):
        self.__t = 0
        self.__t_event_structure = 0
        self.__t_per_event = 0
        self.__input_files = None
        self.__station_id = None
        self.__n_cores = None
        self.__max_distace = None
        self.__current_input_file = None
        self.__random_generator = None
        self.logger = logging.getLogger('NuRadioReco.readCoREAS')

    def begin(self, input_files, xmin, xmax, ymin, ymax, n_cores=10, seed=None, log_level=logging.INFO):
        """
        begin method

        initialize readCoREAS module

        Parameters
        ----------
        input_files: input files
            list of coreas hdf5 files
        xmin: float
            minimum x coordinate of the area in which core positions are distributed
        xmax: float
            maximum x coordinate of the area in which core positions are distributed
        ymin: float
            minimum y coordinate of the area in which core positions are distributed
        ynax: float
            maximum y coordinate of the area in which core positions are distributed
        n_cores: number of cores (integer)
            the number of random core positions to generate for each input file
        seed: int (default: None)
            Seed for the random number generation. If None is passed, no seed is set
        """
        self.__input_files = input_files
        self.__n_cores = n_cores
        self.__current_input_file = 0
        self.__area = [xmin, xmax, ymin, ymax]

        self.__random_generator = numpy.random.RandomState(seed)
        self.logger.setLevel(log_level)

    @register_run()
    def run(self, detector, output_mode=0):
        """
        Read in a random sample of stations from a CoREAS file.
        For each position the closest observer is selected and a simulated
        event is created for that observer.

        Parameters
        ----------
        detector: Detector object
            Detector description of the detector that shall be simulated
        output_mode: integer (default 0)
            
            * 0: only the event object is returned
            * 1: the function reuturns the event object, the current inputfilename, the distance between the choosen station and the requested core position,
               and the area in which the core positions are randomly distributed


        """
        while (self.__current_input_file < len(self.__input_files)):
            t = time.time()
            t_per_event = time.time()
            filesize = os.path.getsize(self.__input_files[self.__current_input_file])
            if(filesize < 18456 * 2):  # based on the observation that a file with such a small filesize is corrupt
                self.logger.warning("file {} seems to be corrupt, skipping to next file".format(self.__input_files[self.__current_input_file]))
                self.__current_input_file += 1
                continue
            corsika = h5py.File(self.__input_files[self.__current_input_file], "r")
            self.logger.info(
                "using coreas simulation {} with E={:2g} theta = {:.0f}".format(
                    self.__input_files[self.__current_input_file],
                    corsika['inputs'].attrs["ERANGE"][0] * units.GeV,
                    corsika['inputs'].attrs["THETAP"][0]
                )
            )
            positions = []
            for i, observer in enumerate(corsika['CoREAS']['observers'].values()):
                position = observer.attrs['position']
                positions.append(np.array([-position[1], position[0], 0]) * units.cm)
#                 self.logger.debug("({:.0f}, {:.0f})".format(positions[i][0], positions[i][1]))
            positions = np.array(positions)

            zenith, azimuth, magnetic_field_vector = coreas.get_angles(corsika)
            cs = cstrafo.cstrafo(zenith, azimuth, magnetic_field_vector)
            positions_vBvvB = cs.transform_from_magnetic_to_geographic(positions.T)
            positions_vBvvB = cs.transform_to_vxB_vxvxB(positions_vBvvB).T
#             for i, pos in enumerate(positions_vBvvB):
#                 self.logger.debug("star shape")
#                 self.logger.debug("({:.0f}, {:.0f}); ({:.0f}, {:.0f})".format(positions[i, 0], positions[i, 1], pos[0], pos[1]))

            dd = (positions_vBvvB[:, 0] ** 2 + positions_vBvvB[:, 1] ** 2) ** 0.5
            ddmax = dd.max()
            self.logger.info("star shape from: {} - {}".format(-dd.max(), dd.max()))

            # generate core positions randomly within a rectangle
            cores = np.array([self.__random_generator.uniform(self.__area[0], self.__area[1], self.__n_cores),
                              self.__random_generator.uniform(self.__area[2], self.__area[3], self.__n_cores),
                              np.zeros(self.__n_cores)]).T

            self.__t_per_event += time.time() - t_per_event
            self.__t += time.time() - t

            station_ids = detector.get_station_ids()
            for iCore, core in enumerate(cores):
                t = time.time()
                evt = NuRadioReco.framework.event.Event(self.__current_input_file, iCore)  # create empty event
                sim_shower = coreas.make_sim_shower(corsika)
                sim_shower.set_parameter(shp.core, core)
                evt.add_sim_shower(sim_shower)
                rd_shower = NuRadioReco.framework.radio_shower.RadioShower(station_ids=station_ids)
                evt.add_shower(rd_shower)
                for station_id in station_ids:
                    # convert into vxvxB frame to calculate closests simulated station to detecor station
                    det_station_position = detector.get_absolute_position(station_id)
                    det_station_position[2] = 0
                    core_rel_to_station = core - det_station_position
        #             core_rel_to_station_vBvvB = cs.transform_from_magnetic_to_geographic(core_rel_to_station)
                    core_rel_to_station_vBvvB = cs.transform_to_vxB_vxvxB(core_rel_to_station)
                    dcore = (core_rel_to_station_vBvvB[0] ** 2 + core_rel_to_station_vBvvB[1] ** 2) ** 0.5
#                     print(f"{core_rel_to_station}, {core_rel_to_station_vBvvB} -> {dcore}")
                    if(dcore > ddmax):
                        # station is outside of the star shape pattern, create empty station
                        station = NuRadioReco.framework.station.Station(station_id)
                        channel_ids = detector.get_channel_ids(station_id)
                        sim_station = coreas.make_sim_station(station_id, corsika, None, channel_ids)
                        station.set_sim_station(sim_station)
                        evt.set_station(station)
                        self.logger.debug(f"station {station_id} is outside of star shape, channel_ids {channel_ids}")
                    else:
                        distances = np.linalg.norm(core_rel_to_station_vBvvB[:2] - positions_vBvvB[:, :2], axis=1)
                        index = np.argmin(distances)
                        distance = distances[index]
                        key = list(corsika['CoREAS']['observers'].keys())[index]
                        self.logger.debug(
                            "generating core at ground ({:.0f}, {:.0f}), rel to station ({:.0f}, {:.0f}) vBvvB({:.0f}, {:.0f}), nearest simulated station is {:.0f}m away at ground ({:.0f}, {:.0f}), vBvvB({:.0f}, {:.0f})".format(
                                cores[iCore][0],
                                cores[iCore][1],
                                core_rel_to_station[0],
                                core_rel_to_station[1],
                                core_rel_to_station_vBvvB[0],
                                core_rel_to_station_vBvvB[1],
                                distance / units.m,
                                positions[index][0],
                                positions[index][1],
                                positions_vBvvB[index][0],
                                positions_vBvvB[index][1]
                            )
                        )
                        t_event_structure = time.time()
                        observer = corsika['CoREAS']['observers'].get(key)

                        station = NuRadioReco.framework.station.Station(station_id)
                        channel_ids = detector.get_channel_ids(station_id)
                        sim_station = coreas.make_sim_station(station_id, corsika, observer, channel_ids)
                        station.set_sim_station(sim_station)
                        evt.set_station(station)
                if(output_mode == 0):
                    self.__t += time.time() - t
                    yield evt
                elif(output_mode == 1):
                    self.__t += time.time() - t
                    self.__t_event_structure += time.time() - t_event_structure
                    yield evt, self.__current_input_file
                else:
                    self.logger.debug("output mode > 1 not implemented")
                    raise NotImplementedError

            self.__current_input_file += 1

    def end(self):
        from datetime import timedelta
        self.logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        self.logger.info("total time used by this module is {}".format(dt))
        self.logger.info("\tcreate event structure {}".format(timedelta(seconds=self.__t_event_structure)))
        self.logger.info("per event {}".format(timedelta(seconds=self.__t_per_event)))
        return dt
