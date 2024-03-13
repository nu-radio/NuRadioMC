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
                interp_lowfreq=30*units.MHz, interp_highfreq=1000*units.MHz, log_level=logging.INFO, efield_for_station=True, debug=False):
            
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
        interp_highfreq: float (default = 1000)
            higher frequency for the bandpass filter in interpolation
        """
        self.__input_files = input_files
        self.__n_cores = n_cores
        self.__current_input_file = 0
        self.__area = [xmin, xmax, ymin, ymax]
        self.__interp_lowfreq = interp_lowfreq
        self.__interp_highfreq = interp_highfreq
        self.__efield_for_station = efield_for_station
        self.__random_generator = np.random.RandomState(seed)
        self.logger.setLevel(log_level)
        self.debug = debug

    def get_efield_times(self, efield, sampling_rate):
        """
        calculate the time axis of the electric field from the sampling rate
        """
        efield_times = np.arange(0, len(efield[:,0])) / sampling_rate
        return efield_times

    def apply_hanning(self, efields):
        """
        Apply a hann window to the electric field in the time domain
        """
        if efields is None:
            return None
        else:
            print(f'Smoothen time trace with hann window. Assume efields are {efields.shape[0]} samples and {efields.shape[1]} polarizations')
            smoothed_trace = np.zeros_like(efields)
            hann_window = np.hanning(efields.shape[0])
            for pol in range(efields.shape[1]):
                smoothed_trace[:,pol] = efields[:,pol] * hann_window
            return smoothed_trace

    def get_interpolated_efield(self, position, core, ddmax, cs, efield_interpolator):
        """
        Accesses the interpolated electric field at the position of the detector. Set nu_radio_timing to True to 
        shift all pulses to the center of the trace and account for the physical time delay of the signal."""

        antenna_position = position
        antenna_position[2] = 0

        # transform antenna position into shower plane with respect to core position, core position is set to 0,0
        antenna_pos_vBvvB = cs.transform_to_vxB_vxvxB(antenna_position, core=core)

        # calculate distance between core position (0,0) and antenna positions in shower plane
        dcore_vBvvB = np.linalg.norm(antenna_pos_vBvvB) 

        # interpolate electric field at antenna position in shower plane which are inside star pattern
        if dcore_vBvvB > ddmax:
            efield_interp = None
        else:
            efield_interp = efield_interpolator(antenna_pos_vBvvB[0], antenna_pos_vBvvB[1],
                                                lowfreq=self.__interp_lowfreq/units.MHz,
                                                highfreq=self.__interp_highfreq/units.MHz,
                                                filter_up_to_cutoff=False,
                                                account_for_timing=False,
                                                nu_radio_timing=True,
                                                const_time_offset=20.0e-9,
                                                full_output=False)

        return efield_interp

    def get_interp_observer(self, efield_interp, efield_times):
        """
        add the time axis before the efield and transpose the array
        """
        interp_observer = np.zeros((len(efield_times), 4))
        interp_observer[:, 0] = efield_times
        interp_observer[:, 1:4] = efield_interp
        interp_observer = interp_observer.T
        return interp_observer

    @register_run()
    def run(self, detector):
        """
        Parameters
        ----------
        detector: Detector object
            Detector description of the detector that shall be simulated
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
                    corsika['inputs'].attrs["THETAP"][0]))

            # coreas: x-axis pointing to the magnetic north, the positive y-axis to the west, and the z-axis upwards.
            # NuRadio: x-axis pointing to the east, the positive y-axis geographical north, and the z-axis upwards.
            # NuRadio_x = -coreas_y, NuRadio_y = coreas_x, NuRadio_z = coreas_z and then correct for mag north
            zenith, azimuth, magnetic_field_vector = coreas.get_angles(corsika)
            cs = cstrafo.cstrafo(zenith, azimuth, magnetic_field_vector)

            obs_positions = []
            electric_field_on_sky = []
            for j_obs, observer in enumerate(corsika['CoREAS']['observers'].values()):
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
            electric_field_r_theta_phi = electric_field_on_sky[:,:,1:]
            coreas_dt = (corsika['CoREAS'].attrs['TimeResolution'] * units.second)
            coreas_sampling_rate = 1. / coreas_dt

            obs_positions = np.array(obs_positions)
            # second to last dimension has to be 3 for the transformation
            obs_positions_geo = cs.transform_from_magnetic_to_geographic(obs_positions.T)
            # transforms the coreas observer positions into the vxB, vxvxB shower plane
            obs_positions_vBvvB = cs.transform_to_vxB_vxvxB(obs_positions_geo).T
            dd = (obs_positions_vBvvB[:, 0] ** 2 + obs_positions_vBvvB[:, 1] ** 2) ** 0.5
            # maximum distance between to observer positions and shower center in shower plane
            ddmax = dd.max() 
            self.logger.info("star shape from: {} - {}".format(-dd.max(), dd.max()))

            if self.debug:
                max_efield = []
                for i in range(len(electric_field_on_sky[:,0,1])):
                    max_efield.append(np.max(np.abs(electric_field_on_sky[i,:,1:4])))
                plt.scatter(obs_positions_vBvvB[:,0], obs_positions_vBvvB[:,1], c=max_efield, cmap='viridis', marker='o', edgecolors='k')
                cbar = plt.colorbar()
                cbar.set_label('max amplitude')
                plt.xlabel('v x B [m]')
                plt.ylabel('v x v x B [m]')
                plt.show()
                plt.close()

            # consturct interpolator object for air shower efield in shower plane
            efield_interpolator = cr_pulse_interpolator.signal_interpolation_fourier.interp2d_signal(
                obs_positions_vBvvB[:, 0],
                obs_positions_vBvvB[:, 1],
                electric_field_r_theta_phi,
                lowfreq=self.__interp_lowfreq/units.MHz,
                highfreq=self.__interp_highfreq/units.MHz,
                sampling_period=coreas_dt/units.s,
                phase_method="phasor",
                radial_method='cubic',
                upsample_factor=5,
                coherency_cutoff_threshold=0.9,
                ignore_cutoff_freq_in_timing=False,
                verbose=False
            )
            # generate core positions randomly within a rectangle
            cores = np.array([self.__random_generator.uniform(self.__area[0], self.__area[1], self.__n_cores),
                              self.__random_generator.uniform(self.__area[2], self.__area[3], self.__n_cores),
                              np.zeros(self.__n_cores)]).T

            self.__t_per_event += time.time() - t_per_event
            self.__t += time.time() - t
            efield_times = self.get_efield_times(electric_field_r_theta_phi[0,:,:], coreas_sampling_rate)
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
                        smooth_res_efield = self.apply_hanning(res_efield)
                        interp_observer = self.get_interp_observer(smooth_res_efield, efield_times)
                        sim_station = coreas.make_sim_station(station_id, corsika, interp_observer, channel_ids, interpFlag=True)
                        station.set_sim_station(sim_station)
                        evt.set_station(station) 
                        t_event_structure = time.time()
                    else:
                        self.logger.info('interpolate efield at antenna position instead of station, this is not implemented yet!')
                        for channel_id in channel_ids:
                            antenna_position_rel = detector.get_relative_position(station_id, channel_id)
                            antenna_position = det_station_position + antenna_position_rel
                            res_efield = self.get_interpolated_efield(antenna_position, core, ddmax, cs, efield_interpolator)
                            smooth_res_efield = self.apply_hanning(res_efield)
                            interp_observer = self.get_interp_observer(smooth_res_efield, efield_times)
                            #TODO add channel_id to sim station
                            sim_station = coreas.make_sim_station(station_id, corsika, interp_observer, channel_id, interpFlag=True)
                            station.set_sim_station(sim_station)
                            evt.set_station(station)            
                            t_event_structure = time.time()

                self.__t += time.time() - t
                yield evt
            self.__current_input_file += 1

    def end(self):
        self.logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        self.logger.info("total time used by this module is {}".format(dt))
        self.logger.info("\tcreate event structure {}".format(timedelta(seconds=self.__t_event_structure)))
        self.logger.info("per event {}".format(timedelta(seconds=self.__t_per_event)))
        return dt