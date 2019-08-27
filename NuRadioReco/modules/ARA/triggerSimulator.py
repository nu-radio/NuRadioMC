from NuRadioReco.utilities import units
import numpy as np
import time
import logging
import scipy.signal
from NuRadioReco.modules.channelGenericNoiseAdder import channelGenericNoiseAdder
from NuRadioReco.framework.trigger import IntegratedPowerTrigger
import NuRadioReco.framework.channel

logger = logging.getLogger('ARAtriggerSimulator')


class triggerSimulator:
    """
    Calculates the trigger of an event.
    Uses the ARA trigger logic of a tunnel diode.
    Implementation as in PyRex by Ben Hokanson-Fasig/
    """

    def __init__(self):
        self.__t = 0
        self.begin()
        logger.warning("This module does not contain cutting the trace to ARA specific parameters.")

    def begin(self, antenna_resistance=8.5 * units.ohm,
              power_mean=None,
              power_std=None):
        """
        Calculate a signal as processed by the tunnel diode.
        The given signal is convolved with the tunnel diodde response as in
        AraSim.

        Parameters
        ----------
        antenna_resistance : float
            Value of the resistance of the ARA antennas
        """

        self.antenna_resistance = antenna_resistance
        self._power_mean = power_mean
        self._power_std = power_std

    # Tunnel diode response functions pulled from arasim
    # RL (Robert Lahmann) Sept 3, 2018: this is not documented in the arasim code, but it seems most
    # logical to assume that the units of the two middle parameters are seconds and that the
    # other parameters are unitless
    _td_args = {
        'down1': (-0.8, 15e-9*units.s, 2.3e-9*units.s, 0),
        'down2': (-0.2, 15e-9*units.s, 4e-9*units.s, 0),
        'up': (1, 18e-9*units.s, 7e-9*units.s, 1e9)
    }
    # Set td_args['up'][0] based on the other args, like in arasim
    _td_args['up'] = (-np.sqrt(2 * np.pi) *
                      (_td_args['down1'][0] * _td_args['down1'][2] +
                       _td_args['down2'][0] * _td_args['down2'][2]) /
                      (2e18 * _td_args['up'][2] ** 3),) + _td_args['up'][1:]

    # Set "down" and "up" functions as in arasim
    @classmethod
    def _td_fdown1(cls, x):
        return (cls._td_args['down1'][3] + cls._td_args['down1'][0] *
                np.exp(-(x - cls._td_args['down1'][1]) ** 2 /
                       (2 * cls._td_args['down1'][2] ** 2)))

    @classmethod
    def _td_fdown2(cls, x):
        return (cls._td_args['down2'][3] + cls._td_args['down2'][0] *
                np.exp(-(x - cls._td_args['down2'][1]) ** 2 /
                       (2 * cls._td_args['down2'][2] ** 2)))

    @classmethod
    def _td_fup(cls, x):
        return (cls._td_args['up'][0] *
                (cls._td_args['up'][3] * (x - cls._td_args['up'][1])) ** 2 *
                np.exp(-(x - cls._td_args['up'][1]) / cls._td_args['up'][2]))

    def tunnel_diode(self, channel):
        """
        Calculate a signal as processed by the tunnel diode.
        The given signal is convolved with the tunnel diodde response as in
        AraSim.
        Parameters
        ----------
        channel : Channel
            Signal to be processed by the tunnel diode.
        power_mean : float
            Parameter extracted in ARA from noise.
            If not given, it is calculated from generic noise
        power_std : float
            Parameter extracted in ARA from noise.
            If not given, it is calculated from generic noise

        Returns
        -------
        trace_after_tunnel_diode: array
            Signal output of the tunnel diode for the input `channel`.

        """
        t_max = 1e-7 * units.s
        n_pts = int(t_max * channel.get_sampling_rate())
        times = np.linspace(0, t_max, n_pts + 1)
        diode_resp = self._td_fdown1(times) + self._td_fdown2(times)
        t_slice = times > self._td_args['up'][1]
        diode_resp[t_slice] += self._td_fup(times[t_slice])
        conv = scipy.signal.convolve(channel.get_trace() ** 2 / self.antenna_resistance,
                                     diode_resp, mode='full')
        # conv multiplied by dt so that the amplitude stays constant for
        # varying dts (determined emperically, see ARVZAskaryanSignal comments)
        # Setting output
        trace_after_tunnel_diode = conv / channel.get_sampling_rate()
        trace_after_tunnel_diode = trace_after_tunnel_diode[:channel.get_trace().shape[0]]

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
        channel : Channel
            ``Channel`` object on which to test the trigger condition.
        Returns
        -------
        boolean
            Whether or not the antenna triggers on `channel`.
        """
        if self._power_mean is None or self._power_std is None:
            # Prepare for antenna trigger by finding rms of noise waveform
            # (1 microsecond) convolved with tunnel diode response

            # This is not fully true yet, since we don't have ARA frontend implemeted
            # long_noise is therefore just set to a certain value rather
            # than taken the full ARA signal chain
            noise = NuRadioReco.framework.channel.Channel(0)

            long_noise = channelGenericNoiseAdder().bandlimited_noise(min_freq=50 * units.MHz,
                                            max_freq=1000 * units.MHz,
                                            n_samples=10000,
                                            sampling_rate=channel.get_sampling_rate(),
                                            amplitude=20.*units.mV,
                                            type='perfect_white')

            noise.set_trace(long_noise, channel.get_sampling_rate())

            self.__power_noise = self.tunnel_diode(noise)

            self._power_mean = np.mean(self.__power_noise)
            self._power_std = np.std(self.__power_noise)

        # Send signal through tunnel_diode
        after_tunnel_diode = self.tunnel_diode(channel)
        low_trigger = (self._power_mean -
                       self._power_std * np.abs(self.power_threshold))
        return np.min(after_tunnel_diode)<low_trigger

    def run(self, evt, station, det,
            power_threshold=6.5,
            coinc_window=110 * units.ns,
            number_concidences=3,
            triggered_channels=[0, 1, 2, 3, 4, 5, 6, 7],
            power_mean=None,
            power_std=None,
            trigger_name='default_integrated_power'):
        """
        simulate ARA trigger logic

        Parameters
        ----------
        power_threshold: float
            The factor of sigma that the signal needs to exceed the noise
        coinc_window: float
            time window in which number_concidences channels need to trigger
        number_concidences: int
            number of channels that are requried in coincidence to trigger a station
        triggered_channels: array of ints
            channels ids that are triggered on
        power_mean : float
            Parameter extracted in ARA from noise.
            If not given, it is calculated from generic noise
        power_std : float
            Parameter extracted in ARA from noise.
            If not given, it is calculated from generic noise
        trigger_name: string
            a unique name of this particular trigger
        """
        # if the run method specifies power mean and rms we use these values,
        # if the parameters are None, the power mean and rms gets calculated for
        # some standard assumptions on the noise RMS and it needs to be done only once
        if(power_mean is not None and power_std is not None):
            self._power_mean = power_mean
            self._power_std = power_std

        self.power_threshold = power_threshold

        # No coincidence requirement yet
        trigger = {}
        for channel in station.iter_channels():
            channel_id = channel.get_id()
            if channel_id not in triggered_channels:
                continue
            trigger[channel_id] = self.has_triggered(channel)

        has_triggered = False
        trigger_time_sample = None
        # loop over the trace with a sliding window of "coinc_window"
        coinc_window_samples = np.int(np.round(coinc_window * channel.get_sampling_rate()))
        trace_length = len(station.get_channel(0).get_trace())
        sampling_rate = station.get_channel(0).get_sampling_rate()

        for i in range(0, trace_length - coinc_window_samples):
            istop = i + coinc_window_samples
            coinc = 0
            trigger_times = []
            for iCh, tr in trigger.items():  # loops through triggers of each channel
                tr = np.array(tr)
                mask_trigger_in_coind_window = (tr >= i) & (tr < istop)
                if(np.sum(mask_trigger_in_coind_window)):
                    coinc += 1
                    trigger_times.append(tr[mask_trigger_in_coind_window][0])  # save time/sample of first trigger in

            if coinc >= number_concidences:
                has_triggered = True
                trigger_time_sample = min(trigger_times)
                break

        trigger = IntegratedPowerTrigger(trigger_name, power_threshold,
                                         coinc_window, channels=triggered_channels,
                                         number_of_coincidences=number_concidences,
                                         power_mean=self._power_mean, power_std=self._power_std)
        if not has_triggered:
            trigger.set_triggered(False)
            logger.info("Station has NOT passed trigger")
            trigger_time_sample = 0
            station.get_trigger().set_trigger_time(trigger_time_sample / sampling_rate)
        else:
            trigger.set_triggered(True)
            trigger.set_trigger_time(trigger_time_sample / sampling_rate)
            logger.info("Station has passed trigger, trigger time is {:.1f} ns (sample {})".format(
                trigger.get_trigger_time() / units.ns, trigger_time_sample))

        station.set_trigger(trigger)

    def end(self):
        from datetime import timedelta
        logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        logger.info("total time used by this module is {}".format(dt))
        return dt
