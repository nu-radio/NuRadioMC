from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units
from NuRadioReco.modules.trigger.highLowThreshold import get_majority_logic
from NuRadioReco.framework.trigger import EnvelopeTrigger
from scipy.signal import hilbert
import numpy as np
import time
import logging

logger = logging.getLogger('envelopeTrigger')


def get_envelope_triggers(trace, threshold):  # define trigger constraint for each channel

    """
    calculates a Hilbert-envelope based trigger

    Parameters
    ----------
    trace: array of floats
        the signal trace
    threshold: float
        the threshold
    Returns
    -------
    triggered bins: array of bools
        the bins where the trigger condition is satisfied
    """

    return np.abs(hilbert(trace)) > threshold


class triggerSimulator:
    """
    Calculate a simple amplitude trigger depending on the Hilbert-envelope.
    """

    def __init__(self):
        self.__t = 0
        self.begin()

    def begin(self, debug=False):
        self.__debug = debug

    @register_run()
    def run(self, evt, station, det,
            threshold=60 * units.mV,
            number_coincidences=2,
            triggered_channels=None,
            coinc_window=500 * units.ns,
            trigger_name='default_envelope_trigger'):
        """
        simulate simple trigger logic, no time window, just threshold in all channels

        Parameters
        ----------
        evt:
            event
        station:
        det:
        threshold: float
            threshold above (or below) a trigger is issued, absolute amplitude
        number_coincidences: int
            number of channels that are required in coincidence to trigger a station
        triggered_channels: array of ints or None
            channels ids that are triggered on, if None trigger will run on all channels
        coinc_window: float
            time window in which number_coincidences channels need to trigger
        trigger_name: string
            a unique name of this particular trigger
        """
        t = time.time()  # absolute time of system

        sampling_rate = station.get_channel(0).get_sampling_rate()
        dt = 1. / sampling_rate

        triggered_bins_channels = []
        channels_that_passed_trigger = []
        if triggered_channels is None:  # caveat: all channels start at the same time
            for channel in station.iter_channels():
                channel_trace_start_time = channel.get_trace_start_time()
                break
        else:
            channel_trace_start_time = station.get_channel(triggered_channels[0]).get_trace_start_time()

        event_id = evt.get_id()
        for channel in station.iter_channels():  # apply envelope trigger to each channel
            channel_id = channel.get_id()
            trace = channel.get_trace()
            if triggered_channels is not None and channel_id not in triggered_channels:
                logger.debug("skipping channel {}".format(channel_id))
                continue
            if channel.get_trace_start_time() != channel_trace_start_time:
                logger.warning('Channel has a trace_start_time that differs from '
                               '        the other channels. The trigger simulator may not work properly')

            triggered_bins = get_envelope_triggers(trace, threshold)
            triggered_bins_channels.append(triggered_bins)

            if True in triggered_bins:
                channels_that_passed_trigger.append(channel.get_id())

        # check for coincidences with get_majority_logic(tts, number_of_coincidences=2,
        # time_coincidence=32 * units.ns, dt=1 * units.ns)
        # returns:
        # triggered: bool; returns True if majority logic is fulfilled --> has_triggered
        # triggered_bins: array of ints; the bins that fulfilled the trigger --> triggered_bins
        # triggered_times = triggered_bins * dt: array of floats;
        # the trigger times relative to the trace --> triggered_times

        has_triggered, triggered_bins, triggered_times = get_majority_logic(triggered_bins_channels,
                                                                            number_coincidences, coinc_window, dt)

        # set maximum signal amplitude
        max_signal = 0

        trigger = EnvelopeTrigger(trigger_name, threshold, triggered_channels, number_coincidences, coinc_window)
        trigger.set_triggered_channels(channels_that_passed_trigger)

        if has_triggered:
            trigger.set_triggered(True)
            trigger.set_trigger_time(triggered_times.min() + channel_trace_start_time)  # trigger_time = time from the beginning of the trace
            logger.debug("station has triggered")

        else:
            trigger.set_triggered(False)
            trigger.set_trigger_time(None)
            logger.debug("station has NOT triggered")

        station.set_trigger(trigger)
        self.__t += time.time() - t

    def end(self):
        from datetime import timedelta
        logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        logger.info("total time used by this module is {}".format(dt))
        return dt
