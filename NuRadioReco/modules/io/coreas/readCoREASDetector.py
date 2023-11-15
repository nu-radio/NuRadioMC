from datetime import timedelta
import logging
import os
import time
import h5py
import numpy as np
from radiotools import coordinatesystems as cstrafo
import cr_pulse_interpolator.signal_interpolation_fourier
import matplotlib.pyplot as plt
from NuRadioReco.modules.base.module import register_run
import NuRadioReco.framework.event
import NuRadioReco.framework.station
import NuRadioReco.framework.radio_shower
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.modules.io.coreas import coreas
from NuRadioReco.utilities import units

conversion_fieldstrength_cgs_to_SI = 2.99792458e10 * units.micro * units.volt / units.meter

class readCoREASDetector:
    """
    Use this as default when reading CoREAS files and combining them with a detector.

    This model reads the electric fields of a CoREAS file with a star shaped pattern and foldes them with it given detector. 
    The core position of the star shape pattern is randomly distributed within a user defined area. 
    If interpolated=True the electric field of the star shaped pattern is interpolated at the detector positions. 
    If interpolated=False the closest observer position of the star shape pattern to the detector is chosen.
    """

    def __init__(self):
            self.__t = 0
            self.__t_event_structure = 0
            self.__t_per_event = 0
            self.__input_files = None
            self.__n_cores = None
            self.__current_input_file = None
            self.__random_generator = None
            self.__interp_lowfreq = None
            self.__interp_highfreq = None
            self.logger = logging.getLogger('NuRadioReco.readCoREASDetector')

    def begin(self, input_files, xmin, xmax, ymin, ymax, n_cores=10, seed=None,
                interp_lowfreq=30, interp_highfreq=500, sampling_period=0.2e-9, log_level=logging.INFO, efield_for_station=True):
            
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
        interp_lowfreq: float (default = 30)
            lower frequency for the bandpass filter in interpolation
        interp_highfreq: float (default = 500)
            higher frequency for the bandpass filter in interpolation
        sampling_period: float (default 0.2e-9)
            sampling period of the signal
        """
        self.__input_files = input_files
        self.__n_cores = n_cores
        self.__current_input_file = 0
        self.__area = [xmin, xmax, ymin, ymax]
        self.__interp_lowfreq = interp_lowfreq
        self.__interp_highfreq = interp_highfreq
        self.__sampling_period = sampling_period
        self.__efield_for_station = efield_for_station
        self.__random_generator = np.random.RandomState(seed)
        self.logger.setLevel(log_level)
     
    def get_interpolated_efield(self, position, core, ddmax, cs, efield_interpolator):
        antenna_position = position
        antenna_position[2] = 0

        # transform antenna position into shower plane with respect to core position, core position is set to 0,0
        antenna_pos_vBvvB = cs.transform_to_vxB_vxvxB(antenna_position, core=core)

        # calculate distance between core position (0,0) and antenna positions in shower plane
        dcore_vBvvB = np.linalg.norm(antenna_pos_vBvvB) 

        # interpolate electric field at antenna position in shower plane which are inside star pattern
        if dcore_vBvvB > ddmax:
            res_efield = None
        else:
            efield_interp = efield_interpolator(antenna_pos_vBvvB[0], antenna_pos_vBvvB[1])
            res_efield = [efield_interp[:,0], efield_interp[:,1], efield_interp[:,2]]
            res_efield = np.array(res_efield)
        return res_efield  

    @register_run()
    def run(self, detector, output_mode=0):
        """
        Parameters
        ----------
        detector: Detector object
            Detector description of the detector that shall be simulated
        output_mode: integer (default 0)
            
            * 0: only the event object is returned
            * 1: the function reuturns the event object, the current inputfilename, 
                 the distance between the choosen station and the requested core position,
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
                                   
            # coreas: x-axis pointing to the magnetic north, the positive y-axis to the west, and the z-axis upwards.
            # NuRadio: x-axis pointing to the east, the positive y-axis geographical north, and the z-axis upwards.
            # NuRadio_x = -coreas_y, NuRadio_y = coreas_x, NuRadio_z = coreas_z and then correct for mag north
            zenith, azimuth, magnetic_field_vector = coreas.get_angles(corsika)
            cs = cstrafo.cstrafo(zenith, azimuth, magnetic_field_vector)

            obs_positions = []
            electric_field_on_sky = []
            for i, observer in enumerate(corsika['CoREAS']['observers'].values()):
                obs_positions.append(np.array([-observer.attrs['position'][1], observer.attrs['position'][0], 0]) * units.cm)

                efield = np.array([observer[()][:,0]*units.second, 
                                   -observer[()][:,2]*conversion_fieldstrength_cgs_to_SI, 
                                   observer[()][:,1]*conversion_fieldstrength_cgs_to_SI, 
                                   observer[()][:,3]*conversion_fieldstrength_cgs_to_SI])

                efield_geo = cs.transform_from_magnetic_to_geographic(efield[1:,:])
                # convert coreas efield to NuRadio spherical coordinated eR, eTheta, ePhi (on sky)
                efield_on_sky = cs.transform_from_ground_to_onsky(efield_geo)
                # insert time column before efield values
                electric_field_on_sky.append(np.insert(efield_on_sky.T, 0, efield[0,:], axis = 1))
            
            # shape: (n_observers, n_samples, (time, eR, eTheta, ePhi))
            electric_field_on_sky = np.array(electric_field_on_sky) 
            
            obs_positions = np.array(obs_positions)
            # second to last dimension has to be 3 for the transformation
            obs_positions_geo = cs.transform_from_magnetic_to_geographic(obs_positions.T)
            # transforms the coreas observer positions into the vxB, vxvxB shower plane
            obs_positions_vBvvB = cs.transform_to_vxB_vxvxB(obs_positions_geo).T
            
            dd = (obs_positions_vBvvB[:, 0] ** 2 + obs_positions_vBvvB[:, 1] ** 2) ** 0.5
            # maximum distance between to observer positions and shower center in shower plane
            ddmax = dd.max() 
            self.logger.info("star shape from: {} - {}".format(-dd.max(), dd.max()))

            # consturct interpolator object for air shower efield in shower plane
            efield_interpolator = cr_pulse_interpolator.signal_interpolation_fourier.interp2d_signal(obs_positions_vBvvB[:,0], obs_positions_vBvvB[:,1], 
                            electric_field_on_sky[:,:,1:], lowfreq = self.__interp_lowfreq, highfreq = self.__interp_highfreq,  sampling_period= self.__sampling_period) 

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
                    station = NuRadioReco.framework.station.Station(station_id)
                    det_station_position = detector.get_absolute_position(station_id)
                    channel_ids = detector.get_channel_ids(station_id)       
                    if self.__efield_for_station:
                        res_efield = self.get_interpolated_efield(det_station_position, core, ddmax, cs, efield_interpolator)
                        sim_station = coreas.make_sim_station(station_id, corsika, res_efield, channel_ids, interpFlag=True)
                        station.set_sim_station(sim_station)
                        evt.set_station(station) 
                        t_event_structure = time.time()
                    else:
                        self.logger.info('interpolate efield at antenna position instead of station')
                        for channel_id in channel_ids:
                            antenna_position_rel = detector.get_relative_position(station_id, channel_id)
                            antenna_position = det_station_position + antenna_position_rel
                            res_efield = self.get_interpolated_efield(antenna_position, core, ddmax, cs, efield_interpolator)
                            sim_station = coreas.make_sim_station(station_id, corsika, res_efield, channel_id, interpFlag=True)
                            station.set_sim_station(sim_station)
                            evt.set_station(station)            
                            t_event_structure = time.time()

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
        self.logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        self.logger.info("total time used by this module is {}".format(dt))
        self.logger.info("\tcreate event structure {}".format(timedelta(seconds=self.__t_event_structure)))
        self.logger.info("per event {}".format(timedelta(seconds=self.__t_per_event)))
        return dt
