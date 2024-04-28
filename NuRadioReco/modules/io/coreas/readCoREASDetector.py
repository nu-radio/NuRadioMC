from datetime import timedelta
import logging
import os
import time
import h5py
import numpy as np
from radiotools import coordinatesystems as cstrafo
import cr_pulse_interpolator.signal_interpolation_fourier
import cr_pulse_interpolator.interpolation_fourier
import matplotlib.pyplot as plt
from NuRadioReco.modules.base.module import register_run
import NuRadioReco.framework.event
import NuRadioReco.framework.station
import NuRadioReco.framework.radio_shower
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.modules.io.coreas import coreas
from NuRadioReco.utilities import units
from collections import defaultdict
from scipy.signal.windows import hann

conversion_fieldstrength_cgs_to_SI = 2.99792458e10 * units.micro * units.volt / units.meter

def get_random_core_positions(xmin, xmax, ymin, ymax, n_cores, seed=None):
    random_generator = np.random.RandomState(seed)

    # generate core positions randomly within a rectangle
    cores = np.array([random_generator.uniform(xmin, xmax, n_cores),
                    random_generator.uniform(ymin, ymax, n_cores),
                    np.zeros(n_cores)]).T
    return cores

def get_efield_times(efield, sampling_rate):
    """
    calculate the time axis of the electric field from the sampling rate

    Parameters
    ----------
    efield: array (n_samples, n_polarizations)
    """
    if efield is None:
        return None
    efield_times = np.arange(0, len(efield[:,0])) / sampling_rate
    return efield_times

def apply_hanning(efields):
    """
    Apply a half hann window to the electric field in the time domain

    Parameters
    ----------
    efield in time domain: array (n_samples, n_polarizations)

    Returns
    ----------
    smoothed_efield: array (n_samples, n_polarizations)
    """

    def _half_hann_window(length, half_percent=None, hann_window_length=None):
        """
        Produce a half-Hann window. This is the Hann window from SciPY with ones inserted in the middle to make the window
        'length' long. Note that this is different from a Hamming window.
        Parameters
        ----------
        length : int
            The desired total length of the window
        half_percent : float, default=None
            The percentage of `length` at the beginning **and** end that should correspond to half of the Hann window
        hann_window_length : int, default=None
            The length of the half the Hann window. If `half_percent` is set, this value will be overwritten by it.

        Returns
        ----------
        half_hann_window : array with shape (length,)
        """
        if half_percent is not None:
            hann_window_length = int(length * half_percent)
        elif hann_window_length is None:
            raise ValueError("Either half_percent or half_window_length should be set!")
        hann_window = hann(2 * hann_window_length)

        half_hann_widow = np.ones(length, dtype=np.double)
        half_hann_widow[:hann_window_length] = hann_window[:hann_window_length]
        half_hann_widow[-hann_window_length:] = hann_window[hann_window_length:]
        return half_hann_widow

    if efields is None:
        return None
    else:
        smoothed_trace = np.zeros_like(efields)
        half_hann_window = _half_hann_window(efields.shape[0], half_percent=0.1)
        for pol in range(efields.shape[1]):
            smoothed_trace[:,pol] = efields[:,pol] * half_hann_window
        return smoothed_trace

def select_channels_per_station(det, station_id, requested_channel_ids):
    """
    Returns a defaultdict object containing the requested channel ids that are in the given station.
    This dict contains the channel group ids as keys with lists of channel ids as values.

    Parameters
    ----------
    det : DetectorBase
        The detector object that contains the station
    station_id : int
        The station id to select channels from
    requested_channel_ids : list
        List of requested channel ids
    """
    channel_ids = defaultdict(list)
    for channel_id in requested_channel_ids:
        if channel_id in det.get_channel_ids(station_id):
            channel_group_id = det.get_channel_group_id(station_id, channel_id)
            channel_ids[channel_group_id].append(channel_id)
    return channel_ids

class readCoREASDetector:
    """
    Use this as default when reading CoREAS files and combining them with a detector.

    This model reads the electric fields of a CoREAS file with a star shaped pattern and foldes them with it given detector. 
    The electric field of the star shaped pattern is interpolated at the detector positions. If the angle between magnetic field 
    and shower direction are below about 15 deg, the interpolation is no longer reliable and the closest observer is used instead.
    """

    def __init__(self):
        self.__t = 0
        self.__t_event_structure = 0
        self.__t_per_event = 0
        self.__input_file = None
        self.__interp_lowfreq = None
        self.__interp_highfreq = None
        self.logger = logging.getLogger('NuRadioReco.readCoREASDetector')

    def begin(self, input_file, interp_efield=True, interp_fluence=False, interp_lowfreq=30*units.MHz, interp_highfreq=1000*units.MHz, log_level=logging.INFO, debug=False):
            
        """
        begin method
        initialize readCoREAS module
        Parameters
        ----------
        input_files: input files
            list of coreas hdf5 files
        interp_lowfreq: float (default = 30)
            lower frequency for the bandpass filter in interpolation, should be broader than the sensetivity band of the detector
        interp_highfreq: float (default = 1000)
            higher frequency for the bandpass filter in interpolation,  should be broader than the sensetivity band of the detector
        """
        self.__input_file = input_file
        self.__corsika = h5py.File(input_file, "r")
        self.__interp_efield = interp_efield
        self.__interp_fluence = interp_fluence
        self.__interp_lowfreq = interp_lowfreq
        self.__interp_highfreq = interp_highfreq
        self.__sampling_rate = 1. / (self.__corsika['CoREAS'].attrs['TimeResolution'] * units.second)
        self.logger.setLevel(log_level)
        self.debug = debug

        self.initialize_efield_interpolator()

    def initialize_efield_interpolator(self):
        zenith, azimuth, magnetic_field_vector = coreas.get_angles(self.__corsika)
        geomagnetic_angle = coreas.get_geomagnetic_angle(zenith, azimuth, magnetic_field_vector)
        self.cs = cstrafo.cstrafo(zenith, azimuth, magnetic_field_vector)

        obs_positions = []
        electric_field_on_sky = []
        for j_obs, observer in enumerate(self.__corsika['CoREAS']['observers'].values()):
            obs_positions.append(np.array([-observer.attrs['position'][1], observer.attrs['position'][0], 0]) * units.cm)
            efield = np.array([observer[()][:,0]*units.second,
                                -observer[()][:,2]*conversion_fieldstrength_cgs_to_SI,
                                observer[()][:,1]*conversion_fieldstrength_cgs_to_SI,
                                observer[()][:,3]*conversion_fieldstrength_cgs_to_SI])
            efield_geo = self.cs.transform_from_magnetic_to_geographic(efield[1:,:])
            # convert coreas efield to NuRadio spherical coordinated eR, eTheta, ePhi (on sky)
            efield_on_sky = self.cs.transform_from_ground_to_onsky(efield_geo)
            # insert time column before efield values
            electric_field_on_sky.append(np.insert(efield_on_sky.T, 0, efield[0,:], axis = 1))

        # shape: (n_observers, n_samples, (time, eR, eTheta, ePhi))
        self.electric_field_on_sky = np.array(electric_field_on_sky)
        self.electric_field_r_theta_phi = self.electric_field_on_sky[:,:,1:]
        self.max_coreas_efield = np.max(np.abs(self.electric_field_r_theta_phi))
        self.empty_efield = np.zeros_like(self.electric_field_r_theta_phi[0,:,:])
        coreas_dt = (self.__corsika['CoREAS'].attrs['TimeResolution'] * units.second)

        obs_positions = np.array(obs_positions)
        # second to last dimension has to be 3 for the transformation
        self.obs_positions_geo = self.cs.transform_from_magnetic_to_geographic(obs_positions.T)
        # transforms the coreas observer positions into the vxB, vxvxB shower plane
        self.obs_positions_vBvvB = self.cs.transform_to_vxB_vxvxB(self.obs_positions_geo).T
        self.star_radius = np.max(np.linalg.norm(self.obs_positions_vBvvB[:, :-1], axis=-1))
        self.geo_star_radius = np.max(np.linalg.norm(self.obs_positions_geo[:-1, :], axis=0))

        if self.debug:
            max_efield = []
            for i in range(len(self.electric_field_on_sky[:,0,1])):
                max_efield.append(np.max(np.abs(self.electric_field_on_sky[i,:,1:4])))
            plt.scatter(self.obs_positions_vBvvB[:,0], self.obs_positions_vBvvB[:,1], c=max_efield, cmap='viridis', marker='o', edgecolors='k')
            cbar = plt.colorbar()
            cbar.set_label('max amplitude')
            plt.xlabel('v x B [m]')
            plt.ylabel('v x v x B [m]')
            plt.show()
            plt.close()

        if self.__interp_efield:
            if geomagnetic_angle < 15*units.deg:
                logging.warning(f'geomagnetic angle is {geomagnetic_angle/units.deg:.2f} deg, which is smaller than 15 deg, which is the lower limit for the signal interpolation. The closest obersever is used instead.')
                self.efield_interpolator = -1
            else:
                logging.info(f'initilize electric field interpolator with lowfreq {self.__interp_lowfreq/units.MHz} MHz and highfreq {self.__interp_highfreq/units.MHz} MHz')
                self.efield_interpolator = cr_pulse_interpolator.signal_interpolation_fourier.interp2d_signal(
                    self.obs_positions_vBvvB[:, 0],
                    self.obs_positions_vBvvB[:, 1],
                    self.electric_field_r_theta_phi,
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
        if self.__interp_fluence:
            #TODO use 10ns around the maximum for the fluence interpolation, then the sum of the square of the efield, get_electric_field_energy_fluence in utilities
            logging.info(f'initilize fluence interpolator')
            self.fluence_interpolator = cr_pulse_interpolator.interpolation_fourier.interp2d_fourier(
                self.obs_positions_vBvvB[:, 0],
                self.obs_positions_vBvvB[:, 1],
                np.max(np.max(np.abs(self.electric_field_r_theta_phi), axis=1),axis=1),
                fill_value="extrapolate"  # THIS OPTION IS UNUSED  IN fluinterp.interp2d_fourier.__init__
            )

    def get_efield_value(self, position, core, kind):
        """
        Accesses the interpolated electric field given the position of the detector on ground. For the interpolation, the pulse will be
        projected in the shower plane for geometrical resonst. Set pulse_centered to True to
        shift all pulses to the center of the trace and account for the physical time delay of the signal.
        """
        antenna_position = position
        z_plane = core[2]
        #core and antenna need to be in the same z plane
        antenna_position[2] = z_plane
        # transform antenna position into shower plane with respect to core position, core position is set to 0,0 in shower plane
        antenna_pos_vBvvB = self.cs.transform_to_vxB_vxvxB(antenna_position, core=core)

        # calculate distance between core position at(0,0) and antenna positions in shower plane
        dcore_vBvvB = np.linalg.norm(antenna_pos_vBvvB[:-1])
        # interpolate electric field at antenna position in shower plane which are inside star pattern
        if dcore_vBvvB > self.star_radius:
            efield_interp = self.empty_efield
            fluence_interp = 0
            logging.info(f'antenna position with distance {dcore_vBvvB:.2f} to core is outside of star pattern with radius {self.star_radius:.2f} on ground {self.geo_star_radius:.2f}, set efield and fluence to zero')
        else:
            if kind == 'efield':
                if self.efield_interpolator == -1:
                    efield = self.get_closest_observer_efield(antenna_pos_vBvvB)
                    efield_interp = efield
                else:
                    efield_interp = self.efield_interpolator(antenna_pos_vBvvB[0], antenna_pos_vBvvB[1],
                                                    lowfreq=self.__interp_lowfreq/units.MHz,
                                                    highfreq=self.__interp_highfreq/units.MHz,
                                                    filter_up_to_cutoff=False,
                                                    account_for_timing=False,
                                                    pulse_centered=True,
                                                    const_time_offset=20.0e-9,
                                                    full_output=False)
            elif kind == 'fluence':
                fluence_interp = self.fluence_interpolator(antenna_pos_vBvvB[0], antenna_pos_vBvvB[1])
            else:
                raise ValueError(f'kind {kind} not supported, please choose between efield and fluence')

        if kind == 'efield':
            if np.max(np.abs(efield_interp)) > self.max_coreas_efield:
                logging.warning(f'interpolated efield {np.max(np.abs(efield_interp)):.2f} is larger than the maximum coreas efield {self.max_coreas_efield:.2f}')
            return efield_interp
        elif kind == 'fluence':
            if np.max(np.abs(fluence_interp)) > self.max_coreas_efield:
                logging.warning(f'interpolated fluence {np.max(np.abs(fluence_interp)):.2f} is larger than the maximum coreas efield {self.max_coreas_efield:.2f}')
            return fluence_interp

    def plot_footprint_fluence(self, dist_scale=300, save_file_path=None):
        from matplotlib import cm
        import matplotlib.pyplot as plt
        print("plotting footprint")

        # Make color plot of f(x, y), using a meshgrid
        ti = np.linspace(-dist_scale, dist_scale, 500)
        XI, YI = np.meshgrid(ti, ti)

        ### Get interpolated values at each grid point, calling the instance of interp2d_fourier
        ZI =  self.fluence_interpolator(XI, YI)
        ZI_mV = ZI/units.mV
        ###
        # And plot it
        maxp = np.max(ZI_mV)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pcolor(XI, YI, ZI_mV, vmax=maxp, vmin=0, cmap=cm.gnuplot2_r)
        mm = cm.ScalarMappable(cmap=cm.gnuplot2_r)
        mm.set_array([0.0, maxp])
        cbar = plt.colorbar(mm, ax=ax)
        cbar.set_label(r'Efield strength [mV/m]', fontsize=14)
        ax.set_xlabel(r'$\vec{v} \times \vec{B} [m]', fontsize=16)
        ax.set_ylabel(r'$\vec{v} \times (\vec{v} \times \vec{B})$ [m]', fontsize=16)
        ax.set_xlim(-dist_scale, dist_scale)
        ax.set_ylim(-dist_scale, dist_scale)
        ax.set_aspect('equal')
        if save_file_path is not None:
            plt.savefig(save_file_path)
        plt.close()

    def get_closest_observer_efield(self, antenna_pos_vBvvB):
        distances = np.linalg.norm(antenna_pos_vBvvB[:2] - self.obs_positions_vBvvB[:,:2], axis=1)
        index = np.argmin(distances)
        distance = distances[index]
        efield = self.electric_field_r_theta_phi[index,:,:]
        logging.info(f'antenna position with distance {distance:.2f} to closest observer is used')
        return efield

    @register_run()
    def run(self, detector, core_position_list=[], selected_station_ids=[], selected_channel_ids=[]):
        """
        Parameters
        ----------
        detector: Detector object
            Detector description of the detector that shall be simulated
        """
        if len(selected_station_ids) == 0:
            selected_station_ids = detector.get_station_ids()
            logging.info(f"using all station ids in detector description: {selected_station_ids}")
        else:
            logging.info(f"using selected station ids: {selected_station_ids}")

        filesize = os.path.getsize(self.__input_file)
        if(filesize < 18456 * 2):  # based on the observation that a file with such a small filesize is corrupt
            self.logger.warning("file {} seems to be corrupt".format(self.__input_file))
        else:
            t = time.time()
            t_per_event = time.time()
            self.logger.info(
                "using coreas simulation {} with E={:2g} theta = {:.0f}".format(
                    self.__input_file,
                    self.__corsika['inputs'].attrs["ERANGE"][0] * units.GeV,
                    self.__corsika['inputs'].attrs["THETAP"][0]))
            
            self.__t_per_event += time.time() - t_per_event
            self.__t += time.time() - t

            for iCore, core in enumerate(core_position_list):
                t = time.time()
                evt = NuRadioReco.framework.event.Event(self.__input_file, iCore)  # create empty event
                sim_shower = coreas.make_sim_shower(self.__corsika)
                sim_shower.set_parameter(shp.core, core)
                evt.add_sim_shower(sim_shower)
                rd_shower = NuRadioReco.framework.radio_shower.RadioShower(station_ids=selected_station_ids)
                evt.add_shower(rd_shower)
                for station_id in selected_station_ids:
                    station = NuRadioReco.framework.station.Station(station_id)
                    sim_station = coreas.make_sim_station(station_id, self.__corsika, observer=None, channel_ids=None) # create empty station
                    det_station_position = detector.get_absolute_position(station_id)
                    channel_ids_in_station = detector.get_channel_ids(station_id)
                    if len(selected_channel_ids) == 0:
                        selected_channel_ids = channel_ids_in_station
                    channel_ids_dict = select_channels_per_station(detector, station_id, selected_channel_ids)
                    for ch_g_ids in channel_ids_dict.keys():
                        antenna_position_rel = detector.get_relative_position(station_id, ch_g_ids)
                        antenna_position = det_station_position + antenna_position_rel
                        if self.__interp_efield:
                            res_efield = self.get_efield_value(antenna_position, core, kind='efield')
                            smooth_res_efield = apply_hanning(res_efield)
                            if smooth_res_efield is None:
                                smooth_res_efield = self.empty_efield
                            efield_times = get_efield_times(smooth_res_efield, self.__sampling_rate)
                        if self.__interp_fluence:
                            res_fluence = self.get_efield_value(antenna_position, core, kind='fluence')
                        else:
                            res_fluence = None
                        channel_ids_for_group_id = channel_ids_dict[ch_g_ids]
                        coreas.add_electric_field(sim_station, channel_ids_for_group_id, smooth_res_efield.T, efield_times, self.__corsika, fluence=res_fluence)
                    station.set_sim_station(sim_station)
                    distance_to_core = np.linalg.norm(det_station_position[:-1] - core[:-1])
                    station.set_parameter(stnp.distance_to_core, distance_to_core)
                    evt.set_station(station)
                    t_event_structure = time.time()

                self.__t += time.time() - t
                yield evt
            self.__corsika.close()

    def end(self):
        self.logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        self.logger.info("total time used by this module is {}".format(dt))
        self.logger.info("\tcreate event structure {}".format(timedelta(seconds=self.__t_event_structure)))
        self.logger.info("per event {}".format(timedelta(seconds=self.__t_per_event)))
        return dt