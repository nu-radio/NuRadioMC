from datetime import timedelta
import logging
import os
import time
import copy
import numpy as np
from NuRadioReco.modules.base.module import register_run
import NuRadioReco.framework.event
import NuRadioReco.framework.station
import NuRadioReco.framework.radio_shower
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.modules.io.coreas import coreas, coreasInterpolator
from NuRadioReco.utilities import units
from NuRadioReco.utilities.signal_processing import half_hann_window
from collections import defaultdict

conversion_fieldstrength_cgs_to_SI = 2.99792458e10 * units.micro * units.volt / units.meter


def get_random_core_positions(xmin, xmax, ymin, ymax, n_cores, seed=None):
    """
    Generate random core positions within a rectangle

    Parameters
    ----------
    xmin: float
        minimum x value
    xmax: float
        maximum x value
    ymin: float
        minimum y value
    ymax: float
        maximum y value
    n_cores: int
        number of cores to generate
    seed: int
        seed for the random number generator

    Returns
    -------
    cores: np.ndarray
        array containing the core positions, shaped as (n_cores, 2)
    """
    random_generator = np.random.RandomState(seed)

    # generate core positions randomly within a rectangle
    cores = np.array([
        random_generator.uniform(xmin, xmax, n_cores),
        random_generator.uniform(ymin, ymax, n_cores),
    ]).T

    return cores


def apply_hanning(efield):
    """
    Apply a half-Hann window to the electric field in the time domain. This smoothens the edges
    to avoid ringing effects.

    Parameters
    ----------
    efield: np.ndarray
        The electric field in the time domain, shaped as (n_samples, n_polarizations)

    Returns
    -------
    smoothed_efield: np.ndarray
        The smoothed trace, shaped as (n_samples, n_polarizations)
    """
    smoothed_trace = np.zeros_like(efield)
    filter = half_hann_window(efield.shape[0], half_percent=0.1)
    for pol in range(efield.shape[1]):
        smoothed_trace[:, pol] = efield[:, pol] * filter
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
    requested_channel_ids : list of int
        List of requested channel ids

    Returns
    -------
    channel_ids : defaultdict
        Dictionary with channel group ids as keys and lists of channel ids as values
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

    This module reads the electric fields of a CoREAS file with a star shaped pattern of observers.
    The electric field is then interpolated at the positions of the antennas or stations of a detector.
    If the angle between magnetic field and shower direction are below about 15 deg,
    the interpolation is no longer reliable and the closest observer is used instead.
    """

    def __init__(self):
        self.__t = 0
        self.__t_event_structure = 0
        self.__t_per_event = 0
        self.__corsika_evt = None

        self.coreas_interpolator = None

        self.logger = logging.getLogger('NuRadioReco.readCoREASDetector')

    def begin(self, input_file, interp_lowfreq=30 * units.MHz, interp_highfreq=1000 * units.MHz,
              log_level=logging.NOTSET):
        """
        begin method, initialize readCoREASDetector module

        Parameters
        ----------
        input_file: str
            coreas hdf5 file
        interp_lowfreq: float, default=30 * units.MHz
            lower frequency for the bandpass filter in interpolation,
            should be broader than the sensitivity band of the detector
        interp_highfreq: float, default=1000 * units.MHz
            higher frequency for the bandpass filter in interpolation,
            should be broader than the sensitivity band of the detector
        log_level: default=logging.NOTSET
            log level for the logger
        """
        self.logger.setLevel(log_level)

        filesize = os.path.getsize(input_file)
        if filesize < 18456 * 2:  # based on the observation that a file with such a small filesize is corrupt
            self.logger.warning("file {} seems to be corrupt".format(input_file))

        self.__corsika_evt = coreas.read_CORSIKA7(input_file)
        self.logger.info(
            f"Using coreas simulation {input_file} with "
            f"E={self.__corsika_evt.get_first_sim_shower().get_parameter(shp.energy):.2g}eV, "
            f"zenith angle = {self.__corsika_evt.get_first_sim_shower().get_parameter(shp.zenith) / units.deg:.2f}deg and "
            f"azimuth angle = {self.__corsika_evt.get_first_sim_shower().get_parameter(shp.azimuth) / units.deg:.2f}deg"
        )

        self.coreas_interpolator = coreasInterpolator.coreasInterpolator(self.__corsika_evt)
        self.coreas_interpolator.initialize_efield_interpolator(interp_lowfreq, interp_highfreq)

    @register_run()
    def run(self, detector, core_position_list, selected_station_channel_ids=None):
        """
        run method, get interpolated electric fields for the given detector and core positions and set them in the event.
        The trace is smoothed with a half-Hann window to avoid ringing effects. When using short traces, this might have
        a significant effect on the result.

        Parameters
        ----------
        detector: `NuRadioReco.detector.detector_base.DetectorBase`
            Detector description of the detector that shall be simulated
        core_position_list: list of (list of float)
            list of core positions in the format [[x1, y1, z1], [x2, y2, z2], ...]
        selected_station_channel_ids: dict, default=None
            A dictionary containing the list of channels IDs to simulate per station.
            If None, all channels of all stations in the detector are simulated.
            To select a station and simulate all its channels, set its value to None.

        Yields
        ------
        evt : `NuRadioReco.framework.event.Event`
            An Event containing a Station object for every selected station, which holds a SimStation containing
            the interpolated ElectricField traces for the selected channels.
        """

        if selected_station_channel_ids is None:
            selected_station_ids = detector.get_station_ids()
            selected_station_channel_ids = {station_id: None for station_id in selected_station_ids}
            logging.info(f"Using all station ids in detector description: {selected_station_ids}")
        else:
            selected_station_ids = list(selected_station_channel_ids.keys())
            logging.info(f"Using selected station ids: {selected_station_ids}")

        t = time.time()
        t_per_event = time.time()
        self.__t_per_event += time.time() - t_per_event
        self.__t += time.time() - t

        # Loop over all cores
        for iCore, core in enumerate(core_position_list):
            t = time.time()

            # Create the Event and add the SimShower
            evt = NuRadioReco.framework.event.Event(self.__corsika_evt.get_run_number(), iCore)
            corsika_sim_stn = self.__corsika_evt.get_station(0).get_sim_station()
            sim_shower = copy.deepcopy(self.__corsika_evt.get_first_sim_shower())  # Don't modify the original shower
            sim_shower.set_parameter(shp.core, core)
            evt.add_sim_shower(sim_shower)

            # Loop over all selected stations
            for station_id in selected_station_ids:
                # Make the (Sim)Station objects to add to the Event
                station = NuRadioReco.framework.station.Station(station_id)
                sim_station = NuRadioReco.framework.sim_station.SimStation(station_id)

                # Copy relevant SimStation parameters over
                for key, value in corsika_sim_stn.get_parameters().items():
                    sim_station.set_parameter(key, value)
                sim_station.set_magnetic_field_vector(corsika_sim_stn.get_magnetic_field_vector())
                sim_station.set_is_cosmic_ray()

                # Find all the selected channels for this station
                det_station_position = detector.get_absolute_position(station_id)
                if selected_station_channel_ids[station_id] is None:
                    selected_channel_ids = detector.get_channel_ids(station_id)
                else:
                    selected_channel_ids = selected_station_channel_ids[station_id]

                # Get the channels in a dictionary with channel group as key and a list of channel ids as value
                channel_ids_dict = select_channels_per_station(detector, station_id, selected_channel_ids)
                for ch_g_ids, channel_ids_for_group_id in channel_ids_dict.items():
                    # Get the absolute antenna position
                    antenna_position_rel = detector.get_relative_position(station_id, channel_ids_for_group_id[0])
                    antenna_position = det_station_position + antenna_position_rel

                    # Get the interpolated electric field and smooth it
                    res_efield, res_trace_start_time = self.coreas_interpolator.get_interp_efield_value(
                        antenna_position[:len(core)] - core  # get the trace at the relative distance from the core
                    )
                    smooth_res_efield = apply_hanning(res_efield)

                    # Store the trace in an ElecticField object
                    electric_field = NuRadioReco.framework.electric_field.ElectricField(channel_ids_for_group_id)
                    electric_field.set_trace(smooth_res_efield.T, self.coreas_interpolator.sampling_rate)
                    electric_field.set_trace_start_time(res_trace_start_time)
                    electric_field.set_parameter(efp.ray_path_type, 'direct')
                    electric_field.set_parameter(efp.zenith, sim_shower[shp.zenith])
                    electric_field.set_parameter(efp.azimuth, sim_shower[shp.azimuth])
                    sim_station.add_electric_field(electric_field)

                sim_station.set_parameter(stnp.zenith, sim_shower[shp.zenith])
                sim_station.set_parameter(stnp.azimuth, sim_shower[shp.azimuth])
                station.set_sim_station(sim_station)

                evt.set_station(station)

            self.__t += time.time() - t
            yield evt

    def end(self):
        dt = timedelta(seconds=self.__t)
        self.logger.info("total time used by this module is {}".format(dt))
        self.logger.info("\tcreate event structure {}".format(timedelta(seconds=self.__t_event_structure)))
        self.logger.info("per event {}".format(timedelta(seconds=self.__t_per_event)))
        return dt
