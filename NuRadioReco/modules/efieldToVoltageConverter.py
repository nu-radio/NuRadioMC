import numpy as np
import time
import logging
import copy
import functools

import NuRadioReco.framework.channel
import NuRadioReco.framework.base_trace
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import stationParameters as stnp

from NuRadioReco.modules.base.module import register_run
from NuRadioReco.detector import antennapattern
from NuRadioReco.utilities import units, fft, ice, geometryUtilities as geo_utl

logger = logging.getLogger('NuRadioReco.efieldToVoltageConverter')


class efieldToVoltageConverter():
    """
    Convolves electric field with antenna response to get the voltage output of the antenna

    Module that should be used to convert simulations to data.
    It assumes that at least one efield is given per channel as input. It will
    convolve the electric field with the corresponding antenna response for the
    incoming direction specified in the channel object.
    The station id, defines antenna location and antenna type.

    """

    def __init__(self, log_level=logging.NOTSET):
        self.__t = 0
        self.__uncertainty = None
        self.__debug = None
        self.__pre_pulse_time = None
        self.__post_pulse_time = None
        self.__antenna_provider = None
        logger.setLevel(log_level)
        self.begin()


    def begin(self, debug=False, uncertainty=None, time_resolution=None,
              pre_pulse_time=200 * units.ns, post_pulse_time=400 * units.ns,
              caching=True
              ):
        """
        Begin method, sets general parameters of module

        Parameters
        ----------
        debug: bool
            enable/disable debug mode (default: False -> no debug output)
        uncertainty: dictionary (default: {})
            optional argument to specify systematic uncertainties. currently supported keys

             * 'sys_dx': systematic uncertainty of x position of antenna
             * 'sys_dy': systematic uncertainty of y position of antenna
             * 'sys_dz': systematic uncertainty of z position of antenna
             * 'sys_amp': systematic uncertainty of the amplifier aplification,
                specify value as relative difference of linear gain
             * 'amp': statistical uncertainty of the amplifier aplification,
                specify value as relative difference of linear gain
        time_resolution: float
            Deprecated.
        pre_pulse_time: float
            length of empty samples that is added before the first pulse
        post_pulse_time: float
            length of empty samples that is added after the simulated trace
        caching: bool
            enable/disable caching of antenna response to save loading times (default: True)
        """

        if time_resolution is not None:
            logger.warning("`time_resolution` is deprecated and will be removed in the future. "
                                "The argument is ignored.")
        self.__caching = caching
        self.__freqs = None
        self.__debug = debug
        self.__pre_pulse_time = pre_pulse_time
        self.__post_pulse_time = post_pulse_time

        # some uncertainties are systematic, fix them here
        self.__uncertainty = uncertainty or {}
        for key in ['sys_dx', 'sys_dy', 'sys_dz']:
            if key in self.__uncertainty:
                self.__uncertainty[key] = np.random.normal(0, self.__uncertainty[key])

        if 'sys_amp' in self.__uncertainty:
            for iCh in self.__uncertainty['sys_amp']:
                self.__uncertainty['sys_amp'][iCh] = np.random.normal(1, self.__uncertainty['sys_amp'][iCh])

        self.__antenna_provider = antennapattern.AntennaPatternProvider()

    @property
    def antenna_provider(self):
        logger.warning("Deprecation warning: `antenna_provider` is deprecated.")
        return self.__antenna_provider

    @functools.lru_cache(maxsize=1024)
    def _get_cached_antenna_response(self, ant_pattern, zen, azi, *ant_orient):
        """
        Returns the cached antenna reponse for a given antenna pattern, antenna orientation
        and signal arrival direction. This wrapper is necessary as arrays and list are not
        hashable (i.e., can not be used as arguments in functions one wants to cache).
        This module ensures that the cache is clearied if the vector `self.__freqs` changes.
        """
        return ant_pattern.get_antenna_response_vectorized(self.__freqs, zen, azi, *ant_orient)


    @register_run()
    def run(self, evt, station, det, channel_ids=None):
        t = time.time()

        # access simulated efield and high level parameters
        sim_station = station.get_sim_station()
        sim_station_id = sim_station.get_id()
        if len(sim_station.get_electric_fields()) == 0:
            raise LookupError(f"station {station.get_id()} has no efields")

        # first we determine the trace start time of all channels and correct
        times_min = []
        times_max = []
        if channel_ids is None:
            channel_ids = det.get_channel_ids(sim_station_id)

        for channel_id in channel_ids:

            for electric_field in sim_station.get_electric_fields_for_channels([channel_id]):
                cab_delay = det.get_cable_delay(sim_station_id, channel_id)
                t0 = electric_field.get_trace_start_time() + cab_delay

                # If the efield is not at the antenna (as possible for a simulated cosmic ray event),
                # the different signal travel time to the antennas has to be taken into account
                dist_channel_efield = np.linalg.norm(det.get_relative_position(sim_station_id, channel_id) - electric_field.get_position())
                if dist_channel_efield / units.mm > 0.01:
                    travel_time_shift = calculate_time_shift_for_cosmic_ray(det, sim_station, electric_field, channel_id)
                    t0 += travel_time_shift

                if not np.isnan(t0):
                    # trace start time is None if no ray tracing solution was found and channel contains only zeros
                    times_min.append(t0)
                    times_max.append(t0 + electric_field.get_number_of_samples() / electric_field.get_sampling_rate())
                    logger.debug("trace start time {}, cable delay {}, tracelength {}".format(
                        electric_field.get_trace_start_time(), cab_delay,
                        electric_field.get_number_of_samples() / electric_field.get_sampling_rate()))

        times_min = np.min(times_min)
        times_max = np.max(times_max)

        # Determine the maximum length of the "readout window"
        max_channel_trace_length = np.max([
            det.get_number_of_samples(station.get_id(), channel_id) / det.get_sampling_frequency(station.get_id(), channel_id)
            for channel_id in channel_ids])

        # pad event times by pre/post pulse time
        times_min -= self.__pre_pulse_time
        times_max += self.__post_pulse_time

        # Add post_pulse_time as long as we reach the minimum required trace length
        while times_max - times_min < max_channel_trace_length:
            times_max += self.__post_pulse_time

        # assumes that all electric fields have the same sampling rate
        time_resolution = 1. / electric_field.get_sampling_rate()

        trace_length = times_max - times_min
        trace_length_samples = int(round(trace_length / time_resolution))
        if trace_length_samples % 2 != 0:
            trace_length_samples += 1

        logger.debug(
            "smallest trace start time {:.1f}, largest trace time {:.1f} -> n_samples = {:d} {:.0f}ns)".format(
                times_min, times_max, trace_length_samples, trace_length / units.ns))

        # loop over all channels
        for channel_id in channel_ids:

            # one channel might contain multiple channels to store the signals from multiple ray paths,
            # so we loop over all simulated channels with the same id,
            # convolve each trace with the antenna response for the given angles
            # and everything up in the time domain
            logger.debug('channel id {}'.format(channel_id))
            channel = NuRadioReco.framework.channel.Channel(channel_id)

            if self.__debug:
                from matplotlib import pyplot as plt
                fig, axes = plt.subplots(2, 1)

            channel_spectrum = None
            trace_object = None
            for electric_field in sim_station.get_electric_fields_for_channels([channel_id]):

                # all simulated channels have a different trace start time
                # in a measurement, all channels have the same physical start time
                # so we need to create one long trace that can hold all the different channel times
                # to achieve a good time resolution, we upsample the trace first.
                new_trace = np.zeros((3, trace_length_samples))


                dist_channel_efield = np.linalg.norm(det.get_relative_position(sim_station_id, channel_id) - electric_field.get_position())
                efield_is_at_antenna = dist_channel_efield / units.mm < 0.01

                # calculate the start bin
                if not np.isnan(electric_field.get_trace_start_time()):
                    cab_delay = det.get_cable_delay(sim_station_id, channel_id)

                    dist_channel_efield = np.linalg.norm(det.get_relative_position(sim_station_id, channel_id) - electric_field.get_position())
                    if not efield_is_at_antenna:
                        travel_time_shift = calculate_time_shift_for_cosmic_ray(
                            det, sim_station, electric_field, channel_id)
                    else:
                        travel_time_shift = 0

                    start_time = electric_field.get_trace_start_time() - times_min + cab_delay + travel_time_shift
                    start_bin = int(round(start_time / time_resolution))

                    # calculate error by using discret bins
                    time_remainder = start_time - start_bin * time_resolution
                    logger.debug('channel {}, start time {:.1f} = bin {:d}, ray solution {}'.format(
                        channel_id, electric_field.get_trace_start_time() + cab_delay, start_bin, electric_field[efp.ray_path_type]))

                    new_efield = NuRadioReco.framework.base_trace.BaseTrace()  # create new data structure with new efield length
                    new_efield.set_trace(copy.copy(electric_field.get_trace()), electric_field.get_sampling_rate())
                    new_efield.apply_time_shift(time_remainder)

                    tr = new_efield.get_trace()
                    stop_bin = start_bin + new_efield.get_number_of_samples()

                    # if checks should never be true...
                    if stop_bin > np.shape(new_trace)[-1]:
                        # ensure new efield does not extend beyond end of trace although this should not happen
                        logger.warning("electric field trace extends beyond the end of the trace and will be cut.")
                        stop_bin = np.shape(new_trace)[-1]
                        tr = np.atleast_2d(tr)[:, :stop_bin-start_bin]

                    if start_bin < 0:
                        # ensure new efield does not extend beyond start of trace although this should not happen
                        logger.warning("electric field trace extends beyond the beginning of the trace and will be cut.")
                        tr = np.atleast_2d(tr)[:, -start_bin:]
                        start_bin = 0

                    new_trace[:, start_bin:stop_bin] = tr

                trace_object = NuRadioReco.framework.base_trace.BaseTrace()
                trace_object.set_trace(new_trace, 1. / time_resolution)

                if self.__debug:
                    axes[0].plot(trace_object.get_times(), new_trace[1], label="eTheta {}".format(electric_field[efp.ray_path_type]), c='C0')
                    axes[0].plot(trace_object.get_times(), new_trace[2], label="ePhi {}".format(electric_field[efp.ray_path_type]), c='C0', linestyle=':')
                    axes[0].plot(electric_field.get_times(), electric_field.get_trace()[1], c='C1', linestyle='-', alpha=.5)
                    axes[0].plot(electric_field.get_times(), electric_field.get_trace()[2], c='C1', linestyle=':', alpha=.5)

                ff = trace_object.get_frequencies()
                efield_fft = trace_object.get_frequency_spectrum()

                zenith = electric_field[efp.zenith]
                azimuth = electric_field[efp.azimuth]

                # If we cache the antenna pattern, we need to make sure that the frequencies have not changed
                # between stations. If they have, we need to clear the cache.
                if self.__caching:
                    if self.__freqs is None:
                        self.__freqs = ff
                    else:
                        if len(self.__freqs) != len(ff):
                            self.__freqs = ff
                            self._get_cached_antenna_response.cache_clear()
                            logger.warning(
                                "Frequencies have changed (array length). Clearing antenna response cache. "
                                "If you similate neutrinos/in-ice radio emission, this is not surprising. Please disable caching "
                                "By passing `caching==False` to the begin method. If you simulate air showers and this happens often, "
                                "something might be wrong...")
                        elif not np.allclose(self.__freqs, ff, rtol=0, atol=0.01 * units.MHz):
                            self.__freqs = ff
                            self._get_cached_antenna_response.cache_clear()
                            logger.warning(
                                "Frequencies have changed (values). Clearing antenna response cache. "
                                "If you similate neutrinos/in-ice radio emission, this is not surprising. Please disable caching "
                                "By passing `caching==False` to the begin method. If you simulate air showers and this happens often, "
                                "something might be wrong...")

                # If the electric field is not at the antenna, we may need to change the
                # signal arrival direction (due to refraction into the ice) and account for
                # missing power due to the Fresnel factors.
                if not efield_is_at_antenna:
                    zenith_antenna, t_theta, t_phi = geo_utl.fresnel_factors_and_signal_zenith(
                        det, sim_station, channel_id, zenith)
                else:
                    zenith_antenna = zenith
                    t_theta = 1
                    t_phi = 1

                # Get the antenna pattern and orientation for the current channel
                antenna_pattern, antenna_orientation = self.get_antenna_pattern_and_orientation(
                    det, sim_station, channel_id, zenith_antenna)

                # Get antenna sensitivity for the given direction
                if self.__caching:
                    vel = self._get_cached_antenna_response(
                        antenna_pattern, zenith_antenna, azimuth, *antenna_orientation)
                else:
                    vel = antenna_pattern.get_antenna_response_vectorized(
                        ff, zenith_antenna, azimuth, *antenna_orientation)

                if vel is None:  # this can happen if there is not signal path to the antenna
                    voltage_fft = np.zeros_like(efield_fft[1])  # set voltage trace to zeros
                else:
                    # Apply antenna response to electric field
                    vel = np.array([vel['theta'] * t_theta, vel['phi'] * t_phi])
                    voltage_fft = np.sum(vel * np.array([efield_fft[1], efield_fft[2]]), axis=0)

                # Remove DC offset
                voltage_fft[np.where(ff < 5 * units.MHz)] = 0.

                if self.__debug:
                    axes[1].plot(
                        trace_object.get_times(), fft.freq2time(voltage_fft, electric_field.get_sampling_rate()),
                        label="{}, zen = {:.0f}deg".format(electric_field[efp.ray_path_type], zenith / units.deg))

                if 'amp' in self.__uncertainty:
                    voltage_fft *= np.random.normal(1, self.__uncertainty['amp'][channel_id])

                if 'sys_amp' in self.__uncertainty:
                    voltage_fft *= self.__uncertainty['sys_amp'][channel_id]

                if channel_spectrum is None:
                    channel_spectrum = voltage_fft
                else:
                    channel_spectrum += voltage_fft

            if self.__debug:
                axes[0].legend(loc='upper left')
                axes[1].legend(loc='upper left')
                plt.show()

            if trace_object is None:  # this happens if don't have any efield for this channel
                # set the trace to zeros
                channel.set_trace(np.zeros(trace_length_samples), 1. / time_resolution)
            else:
                channel.set_frequency_spectrum(channel_spectrum, trace_object.get_sampling_rate())

            channel.set_trace_start_time(times_min)
            station.add_channel(channel)

        self.__t += time.time() - t

    def end(self):
        from datetime import timedelta
        logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        logger.info("total time used by this module is {}".format(dt))
        return dt

    def get_antenna_pattern_and_orientation(self, det, station, channel_id, zenith):
        """ Get the antenna pattern and orientation for a given channel and zenith angle.

        Parameters
        ----------
        det: Detector
            Detector object
        station: Station
            Station object
        channel_id: int
            Channel id of the channel
        zenith: float
            Zenith angle in radians. For some antenna models, the zenith angle is needed
            to get the correct antenna pattern.

        Returns
        -------
        antenna_pattern: AntennaPattern
            Antenna pattern object
        antenna_orientation: list
            Antenna orientation in radians
        """
        antenna_model = det.get_antenna_model(station.get_id(), channel_id, zenith)
        antenna_pattern = self.__antenna_provider.load_antenna_pattern(antenna_model)
        antenna_orientation = det.get_antenna_orientation(station.get_id(), channel_id)
        return antenna_pattern, antenna_orientation


def calculate_time_shift_for_cosmic_ray(det, sim_station, efield, channel_id):
    """
    Calculate the time shift for an electric field to reach a channel.

    For cosmic ray events, we often only have one electric field for all channels, so we have to account
    for the difference in signal travel between channels.

    Parameters
    ----------
    det: Detector
    sim_station: SimStation
    efield: ElectricField
    channel_id: int
        Channel id of the channel

    Returns
    -------
    travel_time_shift: float
        time shift in ns
    """
    station_id = sim_station.get_id()
    site = det.get_site(station_id)

    if sim_station.get_parameter(stnp.zenith) > 90 * units.deg:  # signal is coming from below, so we take IOR of ice
        index_of_refraction = ice.get_refractive_index(det.get_relative_position(station_id, channel_id)[2], site)
    else:  # signal is coming from above, so we take IOR of air
        index_of_refraction = ice.get_refractive_index(1, site)

    antenna_position_rel = det.get_relative_position(station_id, channel_id) - efield.get_position()

    if np.linalg.norm(antenna_position_rel) > 5 * units.m:
        logger.warning("Calculate an additional time shift for an electric field that is more than 5 meters "
                       "away from the antenna position.")

    travel_time_shift = geo_utl.get_time_delay_from_direction(
        efield.get_parameter(efp.zenith),
        efield.get_parameter(efp.azimuth),
        antenna_position_rel,
        index_of_refraction
    )

    return travel_time_shift
