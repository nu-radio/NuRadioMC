from NuRadioReco.utilities import units
import numpy as np
import time
import logging
import scipy
from NuRadioReco.modules.channelGenericNoiseAdder import channelGenericNoiseAdder
import NuRadioReco.framework.channel

logger = logging.getLogger('triggerSimulator')


class triggerSimulator:
    """
    Calculates the trigger of an event.
    Uses the ARA trigger logic of a tunnel diode.
    Implementation as in PyRex by Ben Hokanson-Fasig/
    """

    def __init__(self):
        self.__t = 0

    def begin(self, power_threshold=4,
                    antenna_resistance=8.5*units.ohm,
                    power_mean=None,
                    power_rms=None):

        self.power_threshold = power_threshold
        self.antenna_resistance = antenna_resistance
        self._power_mean = power_mean
        self._power_rms = power_rms

    # Tunnel diode response functions pulled from arasim
    _td_args = {
        'down1': (-0.8, 15e-9, 2.3e-9, 0),
        'down2': (-0.2, 15e-9, 4e-9, 0),
        'up': (1, 18e-9, 7e-9, 1e9)
    }
    # Set td_args['up'][0] based on the other args, like in arasim
    _td_args['up'] = (-np.sqrt(2*np.pi) *
                      (_td_args['down1'][0]*_td_args['down1'][2] +
                       _td_args['down2'][0]*_td_args['down2'][2]) /
                      (2e18*_td_args['up'][2]**3),) + _td_args['up'][1:]

    # Set "down" and "up" functions as in arasim
    @classmethod
    def _td_fdown1(cls, x):
        return (cls._td_args['down1'][3] + cls._td_args['down1'][0] *
                np.exp(-(x-cls._td_args['down1'][1])**2 /
                       (2*cls._td_args['down1'][2]**2)))

    @classmethod
    def _td_fdown2(cls, x):
        return (cls._td_args['down2'][3] + cls._td_args['down2'][0] *
                np.exp(-(x-cls._td_args['down2'][1])**2 /
                       (2*cls._td_args['down2'][2]**2)))

    @classmethod
    def _td_fup(cls, x):
        return (cls._td_args['up'][0] *
                (cls._td_args['up'][3] * (x-cls._td_args['up'][1]))**2 *
np.exp(-(x-cls._td_args['up'][1])/cls._td_args['up'][2]))


    def tunnel_diode(self, channel):
        """
        Calculate a signal as processed by the tunnel diode.
        The given signal is convolved with the tunnel diodde response as in
        AraSim.
        Parameters
        ----------
        signal : Signal
            Signal to be processed by the tunnel diode.
        Returns
        -------
        Signal
            Signal output of the tunnel diode for the input `signal`.

        """
        t_max = 1e-7 * units.s
        n_pts = int(t_max * channel.get_sampling_rate())
        times = np.linspace(0, t_max, n_pts+1)
        diode_resp = self._td_fdown1(times) + self._td_fdown2(times)
        t_slice = times>self._td_args['up'][1]
        diode_resp[t_slice] += self._td_fup(times[t_slice])
        conv = scipy.signal.convolve(channel.get_trace()**2 / self.antenna_resistance,
                                     diode_resp, mode='full')

        # Signal class will automatically only take the first part of conv,
        # which is what we want.
        # conv multiplied by dt so that the amplitude stays constant for
        # varying dts (determined emperically, see ARVZAskaryanSignal comments)

        #Setting output
        trace_after_tunnel_diode = conv/channel.get_sampling_rate()

        return trace_after_tunnel_diode


    def has_triggered(self, channel):
        """
        Check if the detector system triggers on a given channel.
        Passes the signal through the tunnel diode. Then compares the maximum
        and minimum values to a tunnel diode noise signal. Triggers if one of
        the maximum or minimum values exceed the noise mean +/- the noise rms
        times the power threshold.
        Parameters
        ----------
        signal : Signal
            ``Signal`` object on which to test the trigger condition.
        Returns
        -------
        boolean
            Whether or not the antenna triggers on `signal`.
        """
        if self._power_mean is None or self._power_rms is None:
            # Prepare for antenna trigger by finding rms of noise waveform
            # (1 microsecond) convolved with tunnel diode response

            # This is not fully true yet, since we don't have ARA frontend implemeted
            # long_noise is therefore just set to a certain value rather
            # than taken the full ARA signal chain

            noise = NuRadioReco.framework.channel.Channel(0)

            long_noise = channelGenericNoiseAdder().bandlimited_noise(min_freq=50*units.MHz,
                                            max_freq=1000*units.MHz,
                                            n_samples=10001,
                                            sampling_rate=channel.get_sampling_rate(),
                                            amplitude=15*units.mV,
                                            type='perfect_white')

            noise.set_trace(long_noise, channel.get_sampling_rate())

            power_noise = self.tunnel_diode(noise)

            self._power_mean = np.mean(power_noise)
            self._power_rms = np.sqrt(np.mean(power_noise**2))


        # Send signal through tunnel_diode
        after_tunnel_diode = self.tunnel_diode(channel)

        low_trigger = (self._power_mean -
                       self._power_rms*np.abs(self.power_threshold))
        high_trigger = (self._power_mean +
                        self._power_rms*np.abs(self.power_threshold))

        return (np.min(after_tunnel_diode)<low_trigger or
np.max(after_tunnel_diode)>high_trigger)


    def run(self, evt, station, det,
                ):

        channels = station.get_channels()

        # No coincidence requirement yet
        station.set_triggered(False)
        for channel in channels:
            trigger = self.has_triggered(channel)
            if trigger:
                station.set_triggered(True)


    def end(self):
        from datetime import timedelta
        logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        logger.info("total time used by this module is {}".format(dt))
        return dt
