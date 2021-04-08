import numpy as np
import numpy.random
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import channelParameters as chp
import scipy.ndimage


class channelPulseFinderSimulator:
    """
    A class to simulate the behavior of a pulse finder, but using the
    simChannel waveforms. It assumes that the pulse finder will find
    all pulses above a certain signal to noise ratio, with maybe some
    uncertainty on the exact timing.
    """
    def __init__(self):
        self.__min_snr = None
        self.__signal_window_limits = None
        self.__pulse_width = None
        self.__noise_level = None
        self.__signal_window_uncertainty = None
        self.__random = None

    def begin(
            self,
            noise_level,
            min_snr=3,
            signal_window_limits=None,
            pulse_width=50 * units.ns,
            signal_window_uncertainty=0,
            random_seed=None
    ):
        """
        Set up module configuration

        Parameters
        ------------
        noise_level: float
            Root mean square of the noise on the voltage waveforms.
        min_snr: float
            Minimum signal to noise ratio for a pulse to be found.
            The module assumes that all pulses above this SNR are
            found while all the ones below it are missed. Note the
            SNR is defined as half the peak-to-peak amplitude of the
            simChannel waveforms difided by the noise_level parameter
        signal_window_limits: tupel of floats
            Lower and upper limits of the noise window relative to
            the pulse maximum. Note that the first element has to be
            negative and the second positive, otherwise the pulse would
            be outside the window and an error is raised.
        pulse_width: float
            Width of the window in which the peak-to-peak amplitude is
            calculated.
        signal_window_uncertainty: float
            Used to simulate imprecisions when determining the exact pulse
            position. If signal_window_uncertainty > 0, the calculated
            pulse time is shifted by a random variable drawn from a normal
            distribution with standard deviation of signal_window_uncertainty.
        random_seed: float or None
            If signal_window_uncertainty > 0, this parameter can be used to
            set a seed for the random number generator.
        """
        self.__min_snr = min_snr
        if signal_window_limits is None:
            self.__signal_window_limits = (-50 * units.ns, 50 * units.ns)
        else:
            if signal_window_limits[0] > 0 or signal_window_limits[1] < 0:
                raise ValueError('Pulse is outside the signal window for windown limits of ({}, {}).'.format(signal_window_limits[0], signal_window_limits[1]))
            self.__signal_window_limits = signal_window_limits
        self.__pulse_width = pulse_width
        self.__noise_level = noise_level
        self.__signal_window_uncertainty = signal_window_uncertainty
        self.__random = numpy.random.Generator(numpy.random.Philox(random_seed))

    @register_run()
    def run(
            self,
            event,
            station,
            detector
    ):
        for channel in station.iter_channels():
            signal_windows = []
            pulse_times = []
            snrs = []
            sim_channel_sum = None
            for sim_channel in station.get_sim_station().get_channels_by_channel_id(channel.get_id()):
                if sim_channel_sum is None:
                    sim_channel_sum = sim_channel
                else:
                    sim_channel_sum += sim_channel
            if sim_channel_sum is not None:
                channel_maxima = scipy.ndimage.maximum_filter(
                    sim_channel_sum.get_trace(),
                    size=int(self.__pulse_width * sim_channel_sum.get_sampling_rate())
                )
                channel_minima = scipy.ndimage.minimum_filter(
                    sim_channel_sum.get_trace(),
                    size=int(self.__pulse_width * sim_channel_sum.get_sampling_rate())
                )

                for i in range(100):
                    signal_window, pulse_time, snr = self.__get_signal_window(
                        sim_channel_sum.get_times(),
                        channel_maxima,
                        channel_minima,
                        sim_channel_sum.get_trace(),
                        signal_windows
                    )

                    if signal_window is None:
                        break
                    else:
                        signal_windows.append(signal_window)
                        pulse_times.append(pulse_time)
                        snrs.append(snr)
            channel.set_parameter(chp.signal_regions, np.array(signal_windows))
            channel.set_parameter(chp.pulse_times, np.array(pulse_times))
            channel.set_parameter(chp.signal_region_snrs, np.array(snrs))

    def __get_signal_window(
            self,
            times,
            maxima,
            minima,
            trace,
            signal_windows
    ):
        threshold_mask = (maxima - minima > 2. * self.__min_snr * self.__noise_level)
        for signal_window in signal_windows:
            threshold_mask[(times >= signal_window[0] - self.__pulse_width) & (times <= signal_window[1] + self.__pulse_width)] = 0
        if np.max(threshold_mask) == 0:
            return None, None, None
        signal_window_center = times[threshold_mask][np.argmax(np.abs(trace[threshold_mask]))]
        if self.__signal_window_uncertainty > 0:
            signal_window_center += self.__random.normal(0, self.__signal_window_uncertainty)
        snr = (np.max(
            maxima[(times > signal_window_center + self.__signal_window_limits[0]) & (times < signal_window_center + self.__signal_window_limits[1])]
        ) - np.min(
            minima[(times > signal_window_center + self.__signal_window_limits[0]) & (times < signal_window_center + self.__signal_window_limits[1])]
        )) / 2. / self.__noise_level
        return (signal_window_center + self.__signal_window_limits[0], signal_window_center + self.__signal_window_limits[1]), signal_window_center, snr

