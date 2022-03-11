from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.trigger import HighLowTrigger
import numpy as np
import time
import logging
logger = logging.getLogger('HighLowTriggerSimulator')


def get_high_low_triggers(trace, high_threshold, low_threshold,
                          time_coincidence=5 * units.ns, dt=1 * units.ns):
    """
    calculats a high low trigger in a time coincidence window

    Parameters
    ----------
    trace: array of floats
        the signal trace
    high_threshold: float
        the high threshold
    low_threshold: float
        the low threshold
    time_coincidence: float
        the time coincidence window between a high + low
    dt: float
        the width of a time bin (inverse of sampling rate)

    Returns
    -------
    triggered bins: array of bools
        the bins where the trigger condition is satisfied
    """
    n_bins_coincidence = int(np.round(time_coincidence / dt)) + 1
    c = np.ones(n_bins_coincidence, dtype=np.bool)
    logger.debug("length of trace {} bins, coincidence window {} bins".format(len(trace), len(c)))

    c2 = np.array([1, -1])
    m1 = np.convolve(trace > high_threshold, c, mode='full')[:-(n_bins_coincidence - 1)]
    m2 = np.convolve(trace < low_threshold, c, mode='full')[:-(n_bins_coincidence - 1)]
    return np.convolve(m1 & m2, c2, mode='same') > 0


def get_majority_logic(tts, number_of_coincidences=2, time_coincidence=32 * units.ns, dt=1 * units.ns):
    """
    calculates a majority logic trigger

    Parameters
    ----------
    tts: array/list of array of bools
        an array of bools that indicate a single channel trigger per channel
    number_of_coincidences: int (default: 2)
        the number of coincidences between channels
    time_coincidence: float
        the time coincidence window between channels
    dt: float
        the width of a time bin (inverse of sampling rate)

    Returns
    --------
    triggerd: bool
        returns True if majority logic is fulfilled
    triggerd_bins: array of ints
        the bins that fulfilled the trigger
    triggered_times: array of floats
        the trigger times
    """
    n = len(tts[0])
    n_bins_coincidence = int(np.round(time_coincidence / dt)) + 1
    if(n_bins_coincidence > n):  # reduce coincidence window to maximum trace length
        n_bins_coincidence = n
        logger.debug("specified coincidence window longer than tracelenght, reducing coincidence window to trace length")
    c = np.ones(n_bins_coincidence, dtype=np.bool)

    for i in range(len(tts)):
        logger.debug("get_majority_logic() length of trace {} bins, coincidence window {} bins".format(len(tts[i]), len(c)))
        tts[i] = np.convolve(tts[i], c, mode='full')[:-(n_bins_coincidence - 1)]
    tts = np.sum(tts, axis=0)
    ttt = tts >= number_of_coincidences
    triggered_bins = np.atleast_1d(np.squeeze(np.argwhere(tts >= number_of_coincidences)))
    return np.any(ttt), triggered_bins, triggered_bins * dt


class triggerSimulator:
    """
    Calculates the trigger of an event.
    Uses the ARIANNA trigger logic, that a single antenna needs to cross a high and a low threshold value,
    and then coincidences between antennas can be required.
    """

    def __init__(self):
        self.__t = 0
        self.begin()

    def begin(self, log_level=None):
        if(log_level is not None):
            logger.setLevel(log_level)
        return

    @register_run()
    def run(self, evt, station, det,
            threshold_high=60 * units.mV,
            threshold_low=-60 * units.mV,
            high_low_window=5 * units.ns,
            coinc_window=200 * units.ns,
            number_concidences=2,
            triggered_channels=None,
            trigger_name="default_high_low",
            set_not_triggered=False):
        """
        simulate ARIANNA trigger logic

        Parameters
        ----------
        evt: Event
            The event to run the module on
        station: Station
            The station to run the module on
        det: Detector
            The detector description
        threshold_high: float or dict of floats
            the threshold voltage that needs to be crossed on a single channel on the high side
            a dict can be used to specify a different threshold per channel where the key is the channel id
        threshold_low: float or dict of floats
            the threshold voltage that needs to be crossed on a single channel on the low side
            a dict can be used to specify a different threshold per channel where the key is the channel id
        high_low_window: float
           time window in which a high+low crossing needs to occur to trigger a channel
        coinc_window: float
            time window in which number_concidences channels need to trigger
        number_concidences: int
            number of channels that are requried in coincidence to trigger a station
        triggered_channels: array of ints or None
            channels ids that are triggered on, if None trigger will run on all channels
        trigger_name: string
            a unique name of this particular trigger
        set_not_triggered: bool (default: False)
            if True not trigger simulation will be performed and this trigger will be set to not_triggered

        """
        t = time.time()
        sampling_rate = station.get_channel(station.get_channel_ids()[0]).get_sampling_rate()
        channels_that_passed_trigger = []
        if not set_not_triggered:
            triggerd_bins_channels = []
            dt = 1. / sampling_rate
            if triggered_channels is None:
                for channel in station.iter_channels():
                    channel_trace_start_time = channel.get_trace_start_time()
                    break
            else:
                channel_trace_start_time = station.get_channel(triggered_channels[0]).get_trace_start_time()
            for channel in station.iter_channels():
                channel_id = channel.get_id()
                if triggered_channels is not None and channel_id not in triggered_channels:
                    continue
                if channel.get_trace_start_time() != channel_trace_start_time:
                    logger.warning('Channel has a trace_start_time that differs from the other channels. The trigger simulator may not work properly')
                trace = channel.get_trace()
                if(isinstance(threshold_high, dict)):
                    threshold_high_tmp = threshold_high[channel_id]
                else:
                    threshold_high_tmp = threshold_high
                if(isinstance(threshold_low, dict)):
                    threshold_low_tmp = threshold_low[channel_id]
                else:
                    threshold_low_tmp = threshold_low
                triggerd_bins = get_high_low_triggers(trace, threshold_high_tmp, threshold_low_tmp,
                                                      high_low_window, dt)
                if True in triggerd_bins:
                    channels_that_passed_trigger.append(channel.get_id())
                triggerd_bins_channels.append(triggerd_bins)
                logger.debug("channel {}, len(triggerd_bins) = {}".format(channel_id, len(triggerd_bins)))

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
        else:
            trigger.set_triggered(True)
            trigger.set_trigger_time(triggered_times.min() + channel_trace_start_time)
            trigger.set_trigger_times(triggered_times + channel_trace_start_time)
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
