import numpy as np
import time
import logging
from datetime import timedelta

from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.trigger import SimpleThresholdTrigger
from NuRadioReco.modules.trigger.highLowThreshold import get_majority_logic



def get_threshold_triggers(trace, threshold):
    """
    Calculats a simple threshold trigger.

    Parameters
    ----------
    trace: array of floats
        The signal trace
    threshold: float
        The threshold

    Returns
    -------
    triggered bins: array of bools
        The bins where the trigger condition is satisfied
    """

    return np.abs(trace) >= threshold


class triggerSimulator:
    """
    Calculate a very simple amplitude trigger.
    """

    def __init__(self):
        self.__t = 0
        self.begin()
        self.logger = logging.getLogger('NuRadioReco.simpleThresholdTrigger')

    def begin(self):
        return

    @register_run()
    def run(self, evt, station, det,
            threshold=60 * units.mV,
            number_concidences=1,
            triggered_channels=None,
            coinc_window=200 * units.ns,
            trigger_name='default_simple_threshold',
            pre_trigger_time=None):
        """
        Simulate simple trigger logic, no time window, just threshold in all channels

        Parameters
        ----------
        evt: Event
            The event to run the module on
        station: Station
            The station to run the module on
        det: Detector
            The detector description
        number_concidences: int
            number of channels that are requried in coincidence to trigger a station
        threshold: float or dict of floats
            threshold above (or below) a trigger is issued, absolute amplitude
            a dict can be used to specify a different threshold per channel where the key is the channel id
        triggered_channels: array of ints or None
            channels ids that are triggered on, if None trigger will run on all channels
        coinc_window: float
            time window in which number_concidences channels need to trigger
        trigger_name: string
            a unique name of this particular trigger
        pre_trigger_time: float or dict of floats
            Defines the amount of trace recorded before the trigger time. This module does not cut the traces,
            but this trigger property is later used to trim traces accordingly.
            if a dict is given, the keys are the channel_ids, and the value is the pre_trigger_time between the
            start of the trace and the trigger time.
            if only a float is given, the same pre_trigger_time is used for all channels
            If none, the default value of the Trigger class is used, which is currently 55ns.
        """
        t = time.time()

        if triggered_channels is None:
            tmp_channel = station.get_trigger_channel(station.get_channel_ids()[0])
        else:
            tmp_channel = station.get_trigger_channel(triggered_channels[0])

        channel_trace_start_time = tmp_channel.get_trace_start_time()
        sampling_rate = tmp_channel.get_sampling_rate()

        dt = 1. / sampling_rate

        triggerd_bins_channels = []
        channels_that_passed_trigger = []
        for channel in station.iter_trigger_channels():
            channel_id = channel.get_id()
            if triggered_channels is not None and channel_id not in triggered_channels:
                self.logger.debug("skipping channel {}".format(channel_id))
                continue

            if channel.get_trace_start_time() != channel_trace_start_time:
                self.logger.warning('Channel has a trace_start_time that differs from the other channels. The trigger simulator may not work properly')

            trace = channel.get_trace()

            if isinstance(threshold, dict):
                threshold_tmp = threshold[channel_id]
            else:
                threshold_tmp = threshold

            triggerd_bins = get_threshold_triggers(trace, threshold_tmp)
            triggerd_bins_channels.append(triggerd_bins)

            if np.any(triggerd_bins):
                channels_that_passed_trigger.append(channel.get_id())

        has_triggered, triggered_bins, triggered_times = get_majority_logic(
            triggerd_bins_channels, number_concidences, coinc_window, dt)

        # set maximum signal aplitude
        max_signal = 0
        if has_triggered:
            for channel in station.iter_trigger_channels():
                max_signal = max(max_signal, np.abs(channel.get_trace()[triggered_bins]).max())
            station.set_parameter(stnp.channels_max_amplitude, max_signal)

        kwargs = {}
        if pre_trigger_time is not None:
            kwargs['pre_trigger_times'] = pre_trigger_time
        trigger = SimpleThresholdTrigger(trigger_name, threshold, triggered_channels, number_concidences,
                                         **kwargs)
        trigger.set_triggered_channels(channels_that_passed_trigger)

        if has_triggered:
            trigger.set_triggered(True)
            trigger.set_trigger_time(triggered_times.min() + channel_trace_start_time)
            self.logger.debug("station has triggered")
        else:
            trigger.set_triggered(False)
            self.logger.debug("station has NOT triggered")

        station.set_trigger(trigger)

        self.__t += time.time() - t

    def end(self):
        dt = timedelta(seconds=self.__t)
        self.logger.info("total time used by this module is {}".format(dt))
        return dt
