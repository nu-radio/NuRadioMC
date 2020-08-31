from NuRadioReco.utilities import units
import numpy as np
import logging
import scipy.signal
from NuRadioReco.modules.channelGenericNoiseAdder import channelGenericNoiseAdder
from NuRadioReco.utilities.fft import time2freq, freq2time
from scipy.signal import butter, freqs
import NuRadioReco.framework.channel

logger = logging.getLogger('diodeSimulator')


class diodeSimulator:

    def __init__(self, output_passband=(None, None)):
        """
        Calculate a signal as processed by the tunnel diode.
        The given signal is convolved with the tunnel diodde response as in
        AraSim.

        Parameters
        ----------
        output_passband: (float, float) tuple
            Frequencies for a 6th-order Butterworth filter to be applied after
            the diode filtering. If a lowpass filter is needed, please pass a
            (None, float) tuple as parameter.
        """

        logger.info("This module does not contain cutting the trace to ARA specific parameters.")
        logger.info("The user should take care of the appropriate noise rate and trace window.")

        self._output_passband = output_passband

    # Tunnel diode response functions pulled from arasim
    # RL (Robert Lahmann) Sept 3, 2018: this is not documented in the arasim code, but it seems most
    # logical to assume that the units of the two middle parameters are seconds and that the
    # other parameters are unitless
    _td_args = {
        'down1': (-0.8, 15e-9 * units.s, 2.3e-9 * units.s, 0),
        'down2': (-0.2, 15e-9 * units.s, 4e-9 * units.s, 0),
        'up': (1, 18e-9 * units.s, 7e-9 * units.s, 1e9)
    }
    # Set td_args['up'][0] based on the other args, like in arasim
    _td_args['up'] = (-np.sqrt(2 * np.pi) * (_td_args['down1'][0] * _td_args['down1'][2] + _td_args['down2'][0] * _td_args['down2'][2]) / (2e18 * _td_args['up'][2] ** 3),) + _td_args['up'][1:]

    # Set "down" and "up" functions as in arasim
    @classmethod
    def _td_fdown1(cls, x):
        return (cls._td_args['down1'][3] + cls._td_args['down1'][0] * np.exp(-(x - cls._td_args['down1'][1]) ** 2 / (2 * cls._td_args['down1'][2] ** 2)))

    @classmethod
    def _td_fdown2(cls, x):
        return (cls._td_args['down2'][3] + cls._td_args['down2'][0] * np.exp(-(x - cls._td_args['down2'][1]) ** 2 / (2 * cls._td_args['down2'][2] ** 2)))

    @classmethod
    def _td_fup(cls, x):
        return (cls._td_args['up'][0] * (cls._td_args['up'][3] * (x - cls._td_args['up'][1])) ** 2 * np.exp(-(x - cls._td_args['up'][1]) / cls._td_args['up'][2]))

    def tunnel_diode(self, channel):
        """
        Calculate a signal as processed by the tunnel diode.
        The given signal is convolved with the tunnel diode response as in
        AraSim.

        The diode model used in this module returns a dimensionless power trace,
        where the antenna resistance is only a normalisation for the final
        voltage. That's why the antenna resistance has been fixed to a value
        of 8.5 ohms.

        Parameters
        ----------
        channel: Channel
            Signal to be processed by the tunnel diode.

        Returns
        -------
        trace_after_tunnel_diode: array
            Signal output of the tunnel diode for the input `channel`.
            Careful! This trace is dimensionless and comes from a convolution
            of the power with the diode response.
        """
        t_max = 1e-7 * units.s
        antenna_resistance = 8.5 * units.ohm
        n_pts = int(t_max * channel.get_sampling_rate())
        times = np.linspace(0, t_max, n_pts + 1)
        diode_resp = self._td_fdown1(times) + self._td_fdown2(times)
        t_slice = times > self._td_args['up'][1]
        diode_resp[t_slice] += self._td_fup(times[t_slice])
        conv = scipy.signal.convolve(channel.get_trace() ** 2 / antenna_resistance,
                                     diode_resp, mode='full')
        # conv multiplied by dt so that the amplitude stays constant for
        # varying dts (determined emperically, see ARVZAskaryanSignal comments)
        # Setting output
        trace_after_tunnel_diode = conv / channel.get_sampling_rate()
        trace_after_tunnel_diode = trace_after_tunnel_diode[:channel.get_trace().shape[0]]

        # We filter the output if the band is specified
        if self._output_passband != (None, None):

            sampling_rate = channel.get_sampling_rate()
            trace_spectrum = time2freq(trace_after_tunnel_diode, sampling_rate)
            frequencies = np.linspace(0, sampling_rate / 2, len(trace_spectrum))
            if self._output_passband[0] is None:
                b, a = butter(6, self._output_passband[1], 'lowpass', analog=True)
            else:
                b, a = butter(6, self._output_passband, 'bandpass', analog=True)
            w, h = freqs(b, a, frequencies)
            trace_after_tunnel_diode = freq2time(h * trace_spectrum, sampling_rate)

        return trace_after_tunnel_diode

    def calculate_noise_parameters(self,
                                   sampling_rate=1 * units.GHz,
                                   min_freq=50 * units.MHz,
                                   max_freq=1 * units.GHz,
                                   amplitude=10 * units.microvolt,
                                   type='rayleigh',
                                   n_tries=10000,
                                   n_samples=10000):
        """
        Calculates the mean and the standard deviation for the diode-filtered noise.

        Parameters
        ----------
        sampling_rate: float
            Sampling rate
        min_freq: float
            Minimum frequency of the bandwidth
        max_freq: float
            Maximum frequency of the bandwidth
        amplitude: float
            Voltage amplitude (RMS) for the noise
        type: string
            Noise type
        n_tries: int
            Number of times the calculation is carried out, to get proper
            averages for the mean and the standard deviation.
        n_samples: int
            Number of samples for each individual noise trace

        Returns
        -------
        power_mean: float
            Mean of the diode-filtered noise
        power_std: float
            Standard deviation of the diode-filtered noise
        """

        power_mean_list = []
        power_std_list = []

        for i_try in range(n_tries):

            noise = NuRadioReco.framework.channel.Channel(0)

            long_noise = channelGenericNoiseAdder().bandlimited_noise(min_freq=min_freq,
                                                                      max_freq=max_freq,
                                                                      n_samples=n_samples,
                                                                      sampling_rate=sampling_rate,
                                                                      amplitude=amplitude,
                                                                      type=type)

            noise.set_trace(long_noise, sampling_rate)
            power_noise = self.tunnel_diode(noise)

            power_mean_list.append(np.mean(power_noise))
            power_std_list.append(np.std(power_noise))

        power_mean = np.mean(power_mean_list)
        power_std = np.mean(power_std_list)

        return power_mean, power_std

    def end(self):

        pass
