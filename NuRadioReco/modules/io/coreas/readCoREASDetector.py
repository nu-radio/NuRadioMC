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
    efield_times = np.arange(0, len(efield[:,0])) / sampling_rate
    return efield_times

def half_hann_window(length, half_percent=None, hann_window_length=None):
    """
    Produce a half-Hann window. This is the Hann window from SciPY with ones inserted in the middle to make the window
    `length` long. Note that this is different from a Hamming window.
    Parameters
    ----------
    length : int
        The desired total length of the window
    half_percent : float, default=None
        The percentage of `length` at the beginning **and** end that should correspond to half of the Hann window
    hann_window_length : int, default=None
        The length of the half the Hann window. If `half_percent` is set, this value will be overwritten by it.
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

def apply_hanning(efields):
    """
    Apply a hann window to the electric field in the time domain

    Parameters
    ----------
    efield in time domain: array (n_samples, n_polarizations)

    Returns
    ----------
    smoothed_efield: array (n_samples, n_polarizations)
    """
    if efields is None:
        return None
    else:
        smoothed_trace = np.zeros_like(efields)
        half_hann_window = half_hann_window(efields.shape[0], half_percent=0.1)
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
    The electric field of the star shaped pattern is interpolated at the detector positions. 
    """

    def __init__(self):
        self.__t = 0
        self.__t_event_structure = 0
        self.__t_per_event = 0
        self.__input_file = None
        self.__random_generator = None
        self.__interp_lowfreq = None
        self.__interp_highfreq = None
        self.logger = logging.getLogger('NuRadioReco.readCoREASDetector')

    def begin(self, input_file, interp_lowfreq=30*units.MHz, interp_highfreq=1000*units.MHz, log_level=logging.INFO, debug=False):
            
        """
        begin method
        initialize readCoREAS module
        Parameters
        ----------
        input_files: input files
            list of coreas hdf5 files
        interp_lowfreq: float (default = 30)
            lower frequency for the bandpass filter in interpolation
        interp_highfreq: float (default = 1000)
            higher frequency for the bandpass filter in interpolation
        """
        self.__input_file = input_file
        self.__corsika = h5py.File(input_file, "r")
        self.__interp_lowfreq = interp_lowfreq
        self.__interp_highfreq = interp_highfreq
        self.__sampling_rate = 1. / (self.__corsika['CoREAS'].attrs['TimeResolution'] * units.second)
        self.logger.setLevel(log_level)
        self.debug = debug

        self.initialize_efield_interpolator()

    def initialize_efield_interpolator(self):
        zenith, azimuth, magnetic_field_vector = coreas.get_angles(self.__corsika)
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
        electric_field_on_sky = np.array(electric_field_on_sky)
        electric_field_r_theta_phi = electric_field_on_sky[:,:,1:]
        coreas_dt = (self.__corsika['CoREAS'].attrs['TimeResolution'] * units.second)

        obs_positions = np.array(obs_positions)
        # second to last dimension has to be 3 for the transformation
        obs_positions_geo = self.cs.transform_from_magnetic_to_geographic(obs_positions.T)
        # transforms the coreas observer positions into the vxB, vxvxB shower plane
        obs_positions_vBvvB = self.cs.transform_to_vxB_vxvxB(obs_positions_geo).T

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
        self.efield_interpolator = cr_pulse_interpolator.signal_interpolation_fourier.interp2d_signal(
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

    def get_interpolated_efield(self, position, core):
        """
        Accesses the interpolated electric field at the position of the detector on ground. Set pulse_centered to True to
        shift all pulses to the center of the trace and account for the physical time delay of the signal.
        """
        antenna_position = position
        antenna_position[2] = 0
        # transform antenna position into shower plane with respect to core position, core position is set to 0,0 in shower plane
        antenna_pos_vBvvB = self.cs.transform_to_vxB_vxvxB(antenna_position, core=core)
        # calculate distance between core position (0,0) and antenna positions in shower plane
        dcore_vBvvB = np.linalg.norm(antenna_pos_vBvvB) 
        # interpolate electric field at antenna position in shower plane which are inside star pattern
        if dcore_vBvvB > self.ddmax:
            efield_interp = None
        else:
            efield_interp = self.efield_interpolator(antenna_pos_vBvvB[0], antenna_pos_vBvvB[1],
                                                lowfreq=self.__interp_lowfreq/units.MHz,
                                                highfreq=self.__interp_highfreq/units.MHz,
                                                filter_up_to_cutoff=False,
                                                account_for_timing=False,
                                                pulse_centered=True,
                                                const_time_offset=20.0e-9,
                                                full_output=False)

        return efield_interp


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
                    sim_station = coreas.make_empty_sim_station(station_id, self.__corsika)
                    det_station_position = detector.get_absolute_position(station_id)
                    channel_ids_in_station = detector.get_channel_ids(station_id)

                    if len(selected_channel_ids) == 0:
                        selected_channel_ids = channel_ids_in_station

                    channel_ids_dict = select_channels_per_station(detector, station_id, selected_channel_ids)
                    print(channel_ids_dict)
                    for ch_g_ids in channel_group_ids:
                        antenna_position_rel = detector.get_relative_position(station_id, ch_g_ids)
                        antenna_position = det_station_position + antenna_position_rel
                        res_efield = self.get_interpolated_efield(antenna_position, core)
                        smooth_res_efield = apply_hanning(res_efield)
                        efield_times = get_efield_times(smooth_res_efield, self.__sampling_rate)
                        channel_ids_for_group_id = channel_ids_dict[ch_g_ids]
                        coreas.add_electric_field(sim_station, channel_ids_for_group_id, smooth_res_efield, efield_times, self.__corsika)
                    station.set_sim_station(sim_station)
                    evt.set_station(station)
                    t_event_structure = time.time()

                self.__t += time.time() - t
                yield evt
            self.__input_file += 1

    def end(self):
        self.logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        self.logger.info("total time used by this module is {}".format(dt))
        self.logger.info("\tcreate event structure {}".format(timedelta(seconds=self.__t_event_structure)))
        self.logger.info("per event {}".format(timedelta(seconds=self.__t_per_event)))
        return dt