from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.trigger import HighLowTrigger
from NuRadioReco.modules.trigger.highLowThreshold import get_majority_logic
import numpy as np
import time
import logging
logger = logging.getLogger('multiHighLowTrigger')


def get_high_triggers(trace, threshold):
    c2 = np.array([1, -1])
    m1 = trace > threshold
    return np.convolve(m1, c2, mode='full')[:len(m1)] > 0


def get_low_triggers(trace, threshold):
    c2 = np.array([1, -1])
    m1 = trace < threshold
    return np.convolve(m1, c2, mode='full')[:len(m1)] > 0


def get_multiple_high_low_trigger(trace, high_threshold, low_threashold, n_high_lows, time_coincidence=10 * units.ns, dt=1 * units.ns):
    """
    calculats a multiple high low threshold crossings in a time coincidence window

    Parameters
    ----------
    trace: array of floats
        the signal trace
    high_threshold: float
        the high threshold
    low_threshold: float
        the low threshold
    n_high_lows: int
        the required number of high or low crossings in a given coincidence window
    time_coincidence: float
        the time coincidence window between the high or low threshold crossings
    dt: float
        the width of a time bin (inverse of sampling rate)

    Returns
    -------
    triggered bins: array of bools
        the bins where the trigger condition is satisfied
    """
    trig_up = get_high_triggers(trace, high_threshold)
    trig_low = get_low_triggers(trace, low_threashold)
    nc = int(time_coincidence / dt)
    c1 = np.ones(nc)

    tsum_high = np.convolve(trig_up, c1, mode='full')[:-(nc - 1)]
    tsum_low = np.convolve(trig_low, c1, mode='full')[:-(nc - 1)]

    c2 = np.array([1, -1])
    tsumtot = np.convolve((tsum_high + tsum_low) >= n_high_lows, c2, mode='same')
    return tsumtot > 0


class triggerSimulator:
    """
    Calculates the trigger of an event.
    Uses the ARIANNA trigger logic, that a single antenna needs to cross a high and a low threshold value,
    and then coincidences between antennas can be required.
    """

    def __init__(self):
        self.__t = 0
        self.begin()

    def begin(self):
        return

    @register_run()
    def run(self, evt, station, det,
            threshold_high=60 * units.mV,
            threshold_low=-60 * units.mV,
            high_low_window=5 * units.ns,
            n_high_lows=5,
            coinc_window=200 * units.ns,
            number_concidences=2,
            triggered_channels=[0, 1, 2, 3],
            trigger_name="default_high_low",
            set_not_triggered=False):
        """
        simulate ARIANNA trigger logic

        Parameters
        ----------
        threshold_high: float
            the threshold voltage that needs to be crossed on a single channel on the high side
        threshold_low: float
            the threshold voltage that needs to be crossed on a single channel on the low side
        high_low_window: float
           time window in which a high+low crossing needs to occur to trigger a channel
        n_high_lows: int
            the required number of high or low crossings in a given coincidence window
        coinc_window: float
            time window in which number_concidences channels need to trigger
        number_concidences: int
            number of channels that are requried in coincidence to trigger a station
        triggered_channels: array of ints or None
            channels ids that are triggered on, if None trigger will run on all channels
        cut_trace: bool
            if true, trace is cut to the correct length (50ns before the trigger,
            max trace length is set according to detector description)
        trigger_name: string
            a unique name of this particular trigger
        set_not_triggered: bool (default: False)
            if True not trigger simulation will be performed and this trigger will be set to not_triggered

        """
        t = time.time()
        if threshold_low >= threshold_high:
            logger.error("Impossible trigger configuration, high {0} low {1}.".format(threshold_high, threshold_low))
            raise NotImplementedError

        sampling_rate = station.get_channel(0).get_sampling_rate()
        if not set_not_triggered:
            max_signal = 0

            triggerd_bins_channels = []
            dt = 1. / sampling_rate
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
                    continue
                if channel.get_trace_start_time() != channel_trace_start_time:
                    logger.warning('Channel has a trace_start_time that differs from the other channels. The trigger simulator may not work properly')
                trace = channel.get_trace()
                triggerd_bins = get_multiple_high_low_trigger(trace, threshold_high, threshold_low, n_high_lows, high_low_window, dt)
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
        else:
            logger.info("set_not_triggered flag True, setting triggered to False.")
            has_triggered = False

        trigger = HighLowTrigger(trigger_name, threshold_high, threshold_low, high_low_window,
                                 coinc_window, channels=triggered_channels, number_of_coincidences=number_concidences)
        trigger.set_triggered_channels(channels_that_passed_trigger)

        if not has_triggered:
            trigger.set_triggered(False)
            logger.info("Station has NOT passed trigger")
            trigger.set_trigger_time(0)
        else:
            trigger.set_triggered(True)
            trigger.set_trigger_time(triggered_times.min()+channel_trace_start_time)
            logger.info("Station has passed trigger, trigger time is {:.1f} ns".format(
                trigger.get_trigger_time() / units.ns))

        station.set_trigger(trigger)

        self.__t += time.time() - t

    def end(self):
        from datetime import timedelta
        logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        logger.info("total time used by this module is {}".format(dt))
        return dt
