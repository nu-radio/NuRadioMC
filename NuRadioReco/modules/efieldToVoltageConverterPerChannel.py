import numpy as np
from NuRadioReco.utilities import geometryUtilities as geo_utl
from NuRadioReco.utilities import units
from NuRadioReco.utilities import ice
# from detector import antennamodel
from NuRadioReco.detector import antennapattern
from radiotools import coordinatesystems
import copy
import time
import logging
import fractions
from scipy import signal
from decimal import Decimal
import NuRadioReco.framework.channel
logger = logging.getLogger('efieldToVoltageConverterPerChannel')


class efieldToVoltageConverterPerChannel:
    """
    Module that should be used to convert simulations to data.
    It assumes that at least one efield is given per channel as input. It will
    convolve the electric field with the corresponding antenna response for the
    incoming direction specified in the channel object.
    The station id, defines antenna location and antenna type.
    """

    def __init__(self):
        self.__t = 0
        self.begin()

    def begin(self, debug=False, uncertainty={},
              time_resolution=0.1 * units.ns,
              pre_pulse_time=200 * units.ns,
              post_pulse_time=200 * units.ns
              ):
        """
        begin method, sets general parameters of module

        Parameters
        ------------
        debug: bool
            enable/disable debug mode (default: False -> no debug output)
        uncertainty: dictionary
            optional argument to specify systematic uncertainties. currently supported keys
#             * 'sys_dx': systematic uncertainty of x position of antenna
#             * 'sys_dy': systematic uncertainty of y position of antenna
#             * 'sys_dz': systematic uncertainty of z position of antenna
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
        self.__uncertainty = uncertainty
        # some uncertainties are systematic, fix them here
        if('sys_dx' in self.__uncertainty):
            self.__uncertainty['sys_dx'] = np.random.normal(0, self.__uncertainty['sys_dx'])
        if('sys_dy' in self.__uncertainty):
            self.__uncertainty['sys_dy'] = np.random.normal(0, self.__uncertainty['sys_dy'])
        if('sys_dz' in self.__uncertainty):
            self.__uncertainty['sys_dz'] = np.random.normal(0, self.__uncertainty['sys_dz'])
        if('sys_amp'in self.__uncertainty):
            for iCh in self.__uncertainty['sys_amp'].keys():
                self.__uncertainty['sys_amp'][iCh] = np.random.normal(1, self.__uncertainty['sys_amp'][iCh])
        self.antenna_provider = antennapattern.AntennaPatternProvider()

    def run(self, evt, station, det):
        t = time.time()

        # access simulated efield and high level parameters
        sim_station = station.get_sim_station()
        sim_station_id = sim_station.get_id()
        event_time = sim_station.get_station_time()

        # first we determine the trace start time of all channels and correct
        # for different cable delays
        times_min = []
        times_max = []
        for iCh, sim_channel in enumerate(sim_station.iter_channels()):
            for sim_channel2 in sim_channel:
                cab_delay = det.get_cable_delay(sim_station_id, sim_channel2.get_id())
                t0 = sim_channel2.get_trace_start_time() + cab_delay
                times_min.append(t0)
                times_max.append(t0 + sim_channel2.get_number_of_samples() / sim_channel2.get_sampling_rate())
        times_min = np.array(times_min) - self.__pre_pulse_time
        times_max = np.array(times_max) + self.__post_pulse_time
        trace_length = times_max.max() - times_min.min()
        trace_length_samples = int(round(trace_length / self.__time_resolution))
        if trace_length_samples % 2 != 0:
            trace_length_samples += 1
        logger.debug("smallest trace start time {:.1f}, largest trace time {:.1f} -> n_samples = {:d}".format(times_min.min(), times_max.max(), trace_length_samples))

        # loop over all channels
        for sim_channel in sim_station.iter_channels():

            # one channel might contain multiple channels to store the signals from multiple ray paths,
            # so we loop over all simulated channels with the same id,
            # convolve each trace with the antenna response for the given angles
            # and everything up in the time domain
            channel_id = sim_channel[0].get_id()
            channel = NuRadioReco.framework.channel.Channel(channel_id)
            channel_spectrum = None
            for sim_channel2 in sim_channel:
                channel_id = sim_channel2.get_id()
                if(self.__debug):
                    from matplotlib import pyplot as plt
                    fig, axes = plt.subplots(4, 1)

                # all simulated channels have a different trace start time
                # in a measurement, all channels have the same physical start time
                # so we need to create one long trace that can hold all the different channel times
                # to achieve a good time resolution, we upsample the trace first.
                orig_binning = 1. / sim_channel2.get_sampling_rate()  # assume that all channels have the same sampling rate
                target_binning = self.__time_resolution
                resampling_factor = fractions.Fraction(Decimal(orig_binning / target_binning)).limit_denominator(self.__max_upsampling_factor)
                efield = sim_channel2.get_trace()
                new_length = int(efield.shape[1] * resampling_factor)
                resampled_efield = np.zeros((3, new_length))  # create new data structure with new efield length
                for iE in range(len(efield)):
                    trace = efield[iE]
                    if(self.__debug):
                        axes[0].plot(sim_channel2.get_times(), trace)
                    if(resampling_factor.numerator != 1):
                        trace = signal.resample(trace, resampling_factor.numerator * len(trace))
                    if(resampling_factor.denominator != 1):
                        trace = signal.resample(trace, len(trace) / resampling_factor.denominator)
                    resampled_efield[iE] = trace

                if(self.__debug):
                    axes[1].plot(sim_channel2.get_frequencies(), np.abs(sim_channel2.get_frequency_spectrum()[1]))
                    axes[1].plot(sim_channel2.get_frequencies(), np.abs(sim_channel2.get_frequency_spectrum()[2]))

                logger.debug(resampled_efield.shape)
                new_trace = np.zeros((3, trace_length_samples))
                # calculate the start bin
                start_bin = int(round((sim_channel2.get_trace_start_time() - times_min.min()) / self.__time_resolution))
                new_trace[:, start_bin:(start_bin + len(trace))] = resampled_efield
                trace_object = NuRadioReco.framework.base_trace.BaseTrace()
                trace_object.set_trace(new_trace, 1. / self.__time_resolution)
                trace_object.set_trace_start_time(times_min.min())
                if(self.__debug):
                    axes[0].plot(trace_object.get_times(), new_trace[1], "--")
                    axes[0].plot(trace_object.get_times(), new_trace[2], "--")

                ff = trace_object.get_frequencies()
                efield_fft = trace_object.get_frequency_spectrum()
                if(self.__debug):
                    axes[1].plot(ff, np.abs(efield_fft[1]))
                    axes[1].plot(ff, np.abs(efield_fft[2]))

                zenith = sim_channel2['zenith']
                azimuth = sim_channel2['azimuth']

                # get antenna pattern for current channel
                antenna_model = det.get_antenna_model(sim_station_id, channel_id, zenith)
                antenna_pattern = self.antenna_provider.load_antenna_pattern(antenna_model)
                ori = det.get_antanna_orientation(sim_station_id, channel_id)
                logger.debug(ori)
                logger.debug("zen {:.0f}, az {:.0f}".format(zenith / units.deg, azimuth / units.deg))
                VEL = antenna_pattern.get_antenna_response_vectorized(ff, zenith, azimuth, *ori)

                # Apply antenna response to electric field
                voltage_fft = efield_fft[2] * VEL['phi'] + efield_fft[1] * VEL['theta']

                # Remove DC offset
                voltage_fft[np.where(ff < 5 * units.MHz)] = 0.
                if('amp' in self.__uncertainty):
                    voltage_fft *= np.random.normal(1, self.__uncertainty['amp'][channel_id])
                if('sys_amp' in self.__uncertainty):
                    voltage_fft *= self.__uncertainty['sys_amp'][channel_id]

                if(channel_spectrum is None):
                    channel_spectrum = voltage_fft
                else:
                    channel_spectrum += voltage_fft

                if(self.__debug):
                    axes[2].plot(np.abs(voltage_fft))
                    axes[3].plot(np.abs(VEL['phi']))
                    axes[3].plot(np.abs(VEL['theta']))

                if(self.__debug):
                    plt.show()
            channel.set_frequency_spectrum(channel_spectrum, trace_object.get_sampling_rate())

            station.add_channel(channel)
        self.__t += time.time() - t

    def end(self):
        from datetime import timedelta
        logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        logger.info("total time used by this module is {}".format(dt))
        return dt
