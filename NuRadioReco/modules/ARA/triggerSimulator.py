from NuRadioReco.utilities import units
from NuRadioReco.modules.base.module import register_run
import numpy as np
import time
import logging
import scipy.signal
from NuRadioReco.modules.channelGenericNoiseAdder import channelGenericNoiseAdder
from NuRadioReco.framework.trigger import IntegratedPowerTrigger
from NuRadioReco.utilities.diodeSimulator import diodeSimulator
import NuRadioReco.framework.channel

logger = logging.getLogger('ARAtriggerSimulator')


class triggerSimulator:
    """
    Calculates the trigger of an event.
    Uses the ARA trigger logic of a tunnel diode.
    Implementation similar to PyRex by Ben Hokanson-Fasig/
    """

    def __init__(self):
        self.__t = 0
        self._power_mean = None
        self._power_std = None
        self._diode = diodeSimulator()
        logger.warning("This module does not contain cutting the trace to ARA specific parameters.")

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

        # Send signal through tunnel_diode
        after_tunnel_diode = self._diode.tunnel_diode(channel)
        low_trigger = (self._power_mean -
                       self._power_std * np.abs(self.power_threshold))

        return np.min(after_tunnel_diode) < low_trigger

    @register_run()
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
        else:
            error_msg  = 'The power_mean and power_std parameters are not defined. '
            error_msg += 'Please define them. You can use the calculate_noise_parameters '
            error_msg += 'function in utilities.diodeSimulator to do so.'
            raise ValueError(error_msg)

        self.power_threshold = power_threshold

        # No coincidence requirement yet
        trigger = {}
        trigger_times = []
        times_min = []
        times_max = []
        sampling_rates = []
        number_triggered_channels = 0

        for channel in station.iter_channels():
            channel_id = channel.get_id()
            if channel_id not in triggered_channels:
                continue
            trigger[channel_id] = self.has_triggered(channel)
            if trigger[channel_id]:
                number_triggered_channels += 1
                times = channel.get_times()
                trace_after_diode = self._diode.tunnel_diode(channel)
                arg_trigger = np.argmin(trace_after_diode)
                trigger_times.append(times[arg_trigger])
                times_min.append(np.min(times))
                times_max.append(np.max(times))
                sampling_rates.append(channel.get_sampling_rate())

        has_triggered = False
        trigger_time = None

        if (number_triggered_channels >= number_concidences):

            trace_times = np.arange(np.min(times_min), np.max(times_max),
                                    1/np.min(sampling_rates))

            trigger_times = np.array(trigger_times)
            slice_left = int(coinc_window/2/(trace_times[1]-trace_times[0]))
            slice_right = len(trace_times)-slice_left
            for trace_time in trace_times[slice_left:slice_right]:
                if ( np.sum( np.abs(trace_time-trigger_times) <= coinc_window/2 ) >= number_concidences ):
                    has_triggered = True
                    trigger_time = np.min(trigger_times)
                    break

        trigger = IntegratedPowerTrigger(trigger_name, power_threshold,
                                         coinc_window, channels=triggered_channels,
                                         number_of_coincidences=number_concidences,
                                         power_mean=self._power_mean, power_std=self._power_std)

        if not has_triggered:
            trigger.set_triggered(False)
            logger.info("Station has NOT passed trigger")
            trigger_time = 0
            trigger.set_trigger_time(trigger_time)
        else:
            trigger.set_triggered(True)
            trigger.set_trigger_time(trigger_time)
            logger.info("Station has passed trigger, trigger time is {:.1f} ns (sample {})".format(
                trigger.get_trigger_time() / units.ns, trigger_time))

        station.set_trigger(trigger)

    def end(self):
        from datetime import timedelta
        logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        logger.info("total time used by this module is {}".format(dt))
        return dt
