from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.trigger import IntegratedPowerTrigger
from NuRadioReco.modules.trigger.highLowThreshold import get_majority_logic
import numpy as np
import time
import logging
logger = logging.getLogger('powerIntegrationTrigger')


def get_power_int_triggers(trace, threshold, window=10 * units.ns, dt=1 * units.ns, full_output=False):
    """
    calculats a power integration trigger

    Parameters
    ----------
    trace: array of floats
        the signal trace
    threshold: float
        the threshold
    window: float
        the integration window
    dt: float
        the time binning of the trace
    full_output: bool (default False)
        if True, the integrated power is returned as second argument
    Returns
    -------
    triggered bins: array of bools
        the bins where the trigger condition is satisfied
    """
    i_window = int(window / dt)
    power = trace ** 2
    int_power = np.convolve(power, np.ones(i_window, dtype=int), 'valid') * dt

    if full_output:
        return threshold < int_power, int_power
    return threshold < int_power


class triggerSimulator:
    """
    Calculates a power integration trigger.
    """

    def __init__(self):
        self.__t = 0
        self.begin()

    def begin(self, debug=False):
        self.__debug = debug

    @register_run()
    def run(self, evt, station, det,
            threshold,
            integration_window,
            number_concidences=1,
            triggered_channels=None,
            coinc_window=200 * units.ns,
            trigger_name='default_powerint'):
        """
        simulates a power integration trigger. The squared voltages are integrated over a sliding window

        Parameters
        ----------
        number_concidences: int
            number of channels that are requried in coincidence to trigger a station
        threshold: float
            threshold in units of integrated power (V^2*time)
        integration_window: float
            the integration window
        triggered_channels: array of ints or None
            channels ids that are triggered on, if None trigger will run on all channels
        coinc_window: float
            time window in which number_concidences channels need to trigger
        trigger_name: string
            a unique name of this particular trigger
        """
        t = time.time()

        sampling_rate = station.get_channel(0).get_sampling_rate()
        dt = 1. / sampling_rate
        triggerd_bins_channels = []
        if triggered_channels is None:
            for channel in station.iter_channels():
                channel_trace_start_time = channel.get_trace_start_time()
                break
        else:
            channel_trace_start_time = station.get_channel(triggered_channels[0]).get_trace_start_time()
        channels_that_passed_trigger = []
        for channel in station.iter_channels():
            channel_id = channel.get_id()
            if triggered_channels is not None and channel_id not in triggered_channels:
                logger.debug("skipping channel {}".format(channel_id))
                continue
            if channel.get_trace_start_time() != channel_trace_start_time:
                logger.warning('Channel has a trace_start_time that differs from the other channels. The trigger simulator may not work properly')
            trace = channel.get_trace()
            triggerd_bins = get_power_int_triggers(trace, threshold, integration_window, dt=dt)
            triggerd_bins_channels.append(triggerd_bins)
            if True in triggerd_bins:
                channels_that_passed_trigger.append(channel.get_id())

        has_triggered, triggered_bins, triggered_times = get_majority_logic(
            triggerd_bins_channels, number_concidences, coinc_window, dt)
        # set maximum signal aplitude
        max_signal = 0
        if(has_triggered):
            for channel in station.iter_channels():
                max_signal = max(max_signal, np.abs(channel.get_trace()[triggered_bins]).max())
            station.set_parameter(stnp.channels_max_amplitude, max_signal)
        trigger = IntegratedPowerTrigger(trigger_name, threshold, triggered_channels,
                                         number_concidences, integration_window=integration_window)
        trigger.set_triggered_channels(channels_that_passed_trigger)
        if has_triggered:
            trigger.set_triggered(True)
            trigger.set_trigger_time(triggered_times.min() + channel_trace_start_time)
            logger.debug("station has triggered")
        else:
            trigger.set_triggered(False)
            trigger.set_trigger_time(0)
            logger.debug("station has NOT triggered")
        station.set_trigger(trigger)

        self.__t += time.time() - t

    def end(self):
        from datetime import timedelta
        logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        logger.info("total time used by this module is {}".format(dt))
        return dt
