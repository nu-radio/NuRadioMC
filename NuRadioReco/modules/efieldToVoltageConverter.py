import numpy as np
import time
import logging
import NuRadioReco.framework.channel
import NuRadioReco.framework.base_trace
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.detector import antennapattern
from NuRadioReco.utilities import geometryUtilities as geo_utl
from NuRadioReco.utilities import units, fft
from NuRadioReco.utilities import ice
from NuRadioReco.utilities import trace_utilities
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import stationParameters as stnp
import copy


class efieldToVoltageConverter():
    """
    Module that should be used to convert simulations to data.
    It assumes that at least one efield is given per channel as input. It will
    convolve the electric field with the corresponding antenna response for the
    incoming direction specified in the channel object.
    The station id, defines antenna location and antenna type.

    """

    def __init__(self, log_level=logging.WARNING):
        self.__t = 0
        self.__uncertainty = None
        self.__debug = None
        self.__time_resolution = None
        self.__pre_pulse_time = None
        self.__post_pulse_time = None
        self.__max_upsampling_factor = None
        self.antenna_provider = None
        self.begin()
        self.logger = logging.getLogger('NuRadioReco.efieldToVoltageConverter')
        self.logger.setLevel(log_level)

    def begin(self, debug=False, uncertainty=None,
              time_resolution=0.1 * units.ns,
              pre_pulse_time=200 * units.ns,
              post_pulse_time=200 * units.ns
              ):
        """
        Begin method, sets general parameters of module

        Parameters
        ---------------------
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
            time resolution of shifting pulse times
        pre_pulse_time: float
            length of empty samples that is added before the first pulse
        post_pulse_time: float
            length of empty samples that is added after the simulated trace
        """
        self.__debug = debug
        self.__time_resolution = time_resolution
        self.__pre_pulse_time = pre_pulse_time
        self.__post_pulse_time = post_pulse_time
        self.__max_upsampling_factor = 5000
        if uncertainty is None:
            self.__uncertainty = {}
        else:
            self.__uncertainty = uncertainty
        # some uncertainties are systematic, fix them here
        if('sys_dx' in self.__uncertainty):
            self.__uncertainty['sys_dx'] = np.random.normal(0, self.__uncertainty['sys_dx'])
        if('sys_dy' in self.__uncertainty):
            self.__uncertainty['sys_dy'] = np.random.normal(0, self.__uncertainty['sys_dy'])
        if('sys_dz' in self.__uncertainty):
            self.__uncertainty['sys_dz'] = np.random.normal(0, self.__uncertainty['sys_dz'])
        if('sys_amp' in self.__uncertainty):
            for iCh in self.__uncertainty['sys_amp'].keys():
                self.__uncertainty['sys_amp'][iCh] = np.random.normal(1, self.__uncertainty['sys_amp'][iCh])
        self.antenna_provider = antennapattern.AntennaPatternProvider()

    @register_run()
    def run(self, evt, station, det):
        t = time.time()

        # access simulated efield and high level parameters
        sim_station = station.get_sim_station()
        if(len(sim_station.get_electric_fields()) == 0):
            raise LookupError(f"station {station.get_id()} has no efields")
        sim_station_id = sim_station.get_id()

        # first we determine the trace start time of all channels and correct
        # for different cable delays
        times_min = []
        times_max = []
        for iCh in det.get_channel_ids(sim_station_id):
            for electric_field in sim_station.get_electric_fields_for_channels([iCh]):
                time_resolution = 1. / electric_field.get_sampling_rate()
                cab_delay = det.get_cable_delay(sim_station_id, iCh)
                t0 = electric_field.get_trace_start_time() + cab_delay
                # if we have a cosmic ray event, the different signal travel time to the antennas has to be taken into account
                if sim_station.is_cosmic_ray():
                    site = det.get_site(sim_station_id)
                    antenna_position = det.get_relative_position(sim_station_id, iCh) - electric_field.get_position()
                    if sim_station.get_parameter(stnp.zenith) > 90 * units.deg:  # signal is coming from below, so we take IOR of ice
                        index_of_refraction = ice.get_refractive_index(antenna_position[2], site)
                    else:  # signal is coming from above, so we take IOR of air
                        index_of_refraction = ice.get_refractive_index(1, site)
                    # For cosmic ray events, we only have one electric field for all channels, so we have to account
                    # for the difference in signal travel between channels. IMPORTANT: This is only accurate
                    # if all channels have the same z coordinate
                    travel_time_shift = geo_utl.get_time_delay_from_direction(
                        sim_station.get_parameter(stnp.zenith),
                        sim_station.get_parameter(stnp.azimuth),
                        antenna_position,
                        index_of_refraction
                    )
                    t0 += travel_time_shift
                if(not np.isnan(t0)):  # trace start time is None if no ray tracing solution was found and channel contains only zeros
                    times_min.append(t0)
                    times_max.append(t0 + electric_field.get_number_of_samples() / electric_field.get_sampling_rate())
                    self.logger.debug("trace start time {}, cab_delty {}, tracelength {}".format(electric_field.get_trace_start_time(), cab_delay, electric_field.get_number_of_samples() / electric_field.get_sampling_rate()))

        # pad event times by pre/post pulse time
        times_min = np.array(times_min) - self.__pre_pulse_time
        times_max = np.array(times_max) + self.__post_pulse_time

        trace_length = times_max.max() - times_min.min()
        trace_length_samples = int(round(trace_length / time_resolution))
        if trace_length_samples % 2 != 0:
            trace_length_samples += 1
        self.logger.debug("smallest trace start time {:.1f}, largest trace time {:.1f} -> n_samples = {:d} {:.0f}ns)".format(times_min.min(), times_max.max(), trace_length_samples, trace_length / units.ns))

        # loop over all channels
        for channel_id in det.get_channel_ids(station.get_id()):

            # one channel might contain multiple channels to store the signals from multiple ray paths,
            # so we loop over all simulated channels with the same id,
            # convolve each trace with the antenna response for the given angles
            # and everything up in the time domain
            self.logger.debug('channel id {}'.format(channel_id))
            channel = NuRadioReco.framework.channel.Channel(channel_id)
            channel_spectrum = None
            trace_object = None
            if(self.__debug):
                from matplotlib import pyplot as plt
                fig, axes = plt.subplots(2, 1)
            for electric_field in sim_station.get_electric_fields_for_channels([channel_id]):

                # all simulated channels have a different trace start time
                # in a measurement, all channels have the same physical start time
                # so we need to create one long trace that can hold all the different channel times
                # to achieve a good time resolution, we upsample the trace first.
                new_efield = NuRadioReco.framework.base_trace.BaseTrace()  # create new data structure with new efield length
                new_efield.set_trace(copy.copy(electric_field.get_trace()), electric_field.get_sampling_rate())
                new_trace = np.zeros((3, trace_length_samples))
                # calculate the start bin
                if(not np.isnan(electric_field.get_trace_start_time())):
                    cab_delay = det.get_cable_delay(sim_station_id, channel_id)
                    if sim_station.is_cosmic_ray():
                        site = det.get_site(sim_station_id)
                        antenna_position = det.get_relative_position(sim_station_id, channel_id) - electric_field.get_position()
                        if sim_station.get_parameter(stnp.zenith) > 90 * units.deg:  # signal is coming from below, so we take IOR of ice
                            index_of_refraction = ice.get_refractive_index(antenna_position[2], site)
                        else:  # signal is coming from above, so we take IOR of air
                            index_of_refraction = ice.get_refractive_index(1, site)
                        travel_time_shift = geo_utl.get_time_delay_from_direction(
                            sim_station.get_parameter(stnp.zenith),
                            sim_station.get_parameter(stnp.azimuth),
                            antenna_position,
                            index_of_refraction
                        )
                        start_time = electric_field.get_trace_start_time() + cab_delay - times_min.min() + travel_time_shift
                        start_bin = int(round(start_time / time_resolution))
                        time_remainder = start_time - start_bin * time_resolution
                    else:
                        start_time = electric_field.get_trace_start_time() + cab_delay - times_min.min()
                        start_bin = int(round(start_time / time_resolution))
                        time_remainder = start_time - start_bin * time_resolution
                    self.logger.debug('channel {}, start time {:.1f} = bin {:d}, ray solution {}'.format(channel_id, electric_field.get_trace_start_time() + cab_delay, start_bin, electric_field[efp.ray_path_type]))
                    new_efield.apply_time_shift(time_remainder)

                    tr = new_efield.get_trace()
                    stop_bin = start_bin + new_efield.get_number_of_samples()

                    # if checks should never be true...
                    if stop_bin > np.shape(new_trace)[-1]:
                        # ensure new efield does not extend beyond end of trace although this should not happen
                        self.logger.warning("electric field trace extends beyond the end of the trace and will be cut.")
                        stop_bin = np.shape(new_trace)[-1]
                        tr = np.atleast_2d(tr)[:,:stop_bin-start_bin]
                    if start_bin < 0:
                        # ensure new efield does not extend beyond start of trace although this should not happen
                        self.logger.warning("electric field trace extends beyond the beginning of the trace and will be cut.")
                        tr = np.atleast_2d(tr)[:,-start_bin:]
                        start_bin = 0
                    new_trace[:, start_bin:stop_bin] = tr
                trace_object = NuRadioReco.framework.base_trace.BaseTrace()
                trace_object.set_trace(new_trace, 1. / time_resolution)
                if(self.__debug):
                    axes[0].plot(trace_object.get_times(), new_trace[1], label="eTheta {}".format(electric_field[efp.ray_path_type]), c='C0')
                    axes[0].plot(trace_object.get_times(), new_trace[2], label="ePhi {}".format(electric_field[efp.ray_path_type]), c='C0', linestyle=':')
                    axes[0].plot(electric_field.get_times(), electric_field.get_trace()[1], c='C1', linestyle='-', alpha=.5)
                    axes[0].plot(electric_field.get_times(), electric_field.get_trace()[2], c='C1', linestyle=':', alpha=.5)
                ff = trace_object.get_frequencies()
                efield_fft = trace_object.get_frequency_spectrum()

                zenith = electric_field[efp.zenith]
                azimuth = electric_field[efp.azimuth]

                # get antenna pattern for current channel
                VEL = trace_utilities.get_efield_antenna_factor(sim_station, ff, [channel_id], det, zenith, azimuth, self.antenna_provider)

                if VEL is None:  # this can happen if there is not signal path to the antenna
                    voltage_fft = np.zeros_like(efield_fft[1])  # set voltage trace to zeros
                else:
                    # Apply antenna response to electric field
                    VEL = VEL[0]  # we only requested the VEL for one channel, so selecting it
                    voltage_fft = np.sum(VEL * np.array([efield_fft[1], efield_fft[2]]), axis=0)

                # Remove DC offset
                voltage_fft[np.where(ff < 5 * units.MHz)] = 0.

                if(self.__debug):
                    axes[1].plot(trace_object.get_times(), fft.freq2time(voltage_fft, electric_field.get_sampling_rate()), label="{}, zen = {:.0f}deg".format(electric_field[efp.ray_path_type], zenith / units.deg))

                if('amp' in self.__uncertainty):
                    voltage_fft *= np.random.normal(1, self.__uncertainty['amp'][channel_id])
                if('sys_amp' in self.__uncertainty):
                    voltage_fft *= self.__uncertainty['sys_amp'][channel_id]

                if(channel_spectrum is None):
                    channel_spectrum = voltage_fft
                else:
                    channel_spectrum += voltage_fft

            if(self.__debug):
                axes[0].legend(loc='upper left')
                axes[1].legend(loc='upper left')
                plt.show()
            if trace_object is None:  # this happens if don't have any efield for this channel
                # set the trace to zeros
                channel.set_trace(np.zeros(trace_length_samples), 1. / time_resolution)
            else:
                channel.set_frequency_spectrum(channel_spectrum, trace_object.get_sampling_rate())
            channel.set_trace_start_time(times_min.min())

            station.add_channel(channel)
        self.__t += time.time() - t

    def end(self):
        from datetime import timedelta
        self.logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        self.logger.info("total time used by this module is {}".format(dt))
        return dt
