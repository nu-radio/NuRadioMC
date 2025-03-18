from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.trigger import HighLowTrigger
from NuRadioReco.modules.analogToDigitalConverter import analogToDigitalConverter
import numpy as np
import time
import logging


logger = logging.getLogger('NuRadioReco.HighLowTriggerSimulator')


def get_high_low_triggers(trace, high_threshold, low_threshold,
                          time_coincidence=5 * units.ns, dt=1 * units.ns):
    """
    calculates a high low trigger in a time coincidence window

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
    c = np.ones(n_bins_coincidence, dtype=bool)
    logger.debug("length of trace {} bins, coincidence window {} bins".format(len(trace), len(c)))

    if trace.dtype != type(high_threshold):
        logger.error(f"The trace ({trace.dtype}) and the threshold ({type(high_threshold)}) must have the same type")
        raise TypeError(f"The trace ({trace.dtype}) and the threshold ({type(high_threshold)}) must have the same type")

    c2 = np.array([1, -1])
    m1 = np.convolve(trace >= high_threshold, c, mode='full')[:-(n_bins_coincidence - 1)]
    m2 = np.convolve(trace <= low_threshold, c, mode='full')[:-(n_bins_coincidence - 1)]
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
    -------
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
    c = np.ones(n_bins_coincidence, dtype=bool)

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

    def begin(self, log_level=logging.NOTSET):
        logger.setLevel(log_level)

    @register_run()
    def run(self, evt, station, det,
            use_digitization=False,  # Only active if use_digitization is set to True
            threshold_high=60 * units.mV,
            threshold_low=-60 * units.mV,
            high_low_window=5 * units.ns,
            coinc_window=200 * units.ns,
            number_concidences=2,
            triggered_channels=None,
            trigger_name="default_high_low",
            set_not_triggered=False,
            Vrms=None,
            trigger_adc=True,
            clock_offset=0,
            adc_output='voltage'):
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
        use_digitization: bool
            If True, traces will be digitized
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
        Vrms: float
            If supplied, overrides adc_voltage_range as supplied in the detector description file
        trigger_adc: bool
            If True, the relevant ADC parameters in the config file are the ones
            that start with `'trigger_'`
        clock_offset: float

        adc_output: string
            Options:
            * 'voltage' to store the ADC output as discretised voltage trace
            * 'counts' to store the ADC output in ADC counts
        """
        t = time.time()

        if use_digitization:
            adcConverter = analogToDigitalConverter()

        channels_that_passed_trigger = []
        if not set_not_triggered:
            triggerd_bins_channels = []

            if triggered_channels is None:
                for channel in station.iter_trigger_channels():
                    channel_trace_start_time = channel.get_trace_start_time()
                    break
            else:
                channel_trace_start_time = station.get_trigger_channel(triggered_channels[0]).get_trace_start_time()

            for channel in station.iter_trigger_channels():
                channel_id = channel.get_id()
                if triggered_channels is not None and channel_id not in triggered_channels:
                    continue

                if channel.get_trace_start_time() != channel_trace_start_time:
                    logger.warning('Channel has a trace_start_time that differs from the other channels. The trigger simulator may not work properly')

                dt = 1. / channel.get_sampling_rate()
                trace = np.array(channel.get_trace())

                if use_digitization:
                    trace, trigger_sampling_rate = adcConverter.get_digital_trace(
                        station, det, channel,
                        Vrms=Vrms,
                        trigger_adc=trigger_adc,
                        clock_offset=clock_offset,
                        return_sampling_frequency=True,
                        adc_type='perfect_floor_comparator',
                        adc_output=adc_output,
                        trigger_filter=None
                    )

                    # overwrite the dt defined for the original trace by the digitized one
                    dt = 1. / trigger_sampling_rate

                triggerd_bins = get_high_low_triggers(
                    trace,
                    _get_threshold_channel(threshold_high, channel_id),
                    _get_threshold_channel(threshold_low, channel_id),
                    high_low_window, dt)

                if np.any(triggerd_bins):
                    channels_that_passed_trigger.append(channel.get_id())

                triggerd_bins_channels.append(triggerd_bins)
                logger.debug("channel {}, len(triggerd_bins) = {}".format(channel_id, len(triggerd_bins)))

            if len(triggerd_bins_channels):
                has_triggered, triggered_bins, triggered_times = get_majority_logic(
                    triggerd_bins_channels, number_concidences, coinc_window, dt)
            else:
                has_triggered = False
            # set maximum signal aplitude
            max_signal = 0

            if has_triggered:
                for channel in station.iter_trigger_channels():
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
            # trigger_time= time from start of trace + start time of trace with respect to moment of first interaction = trigger time from moment of first interaction
            trigger.set_trigger_time(triggered_times.min() + channel_trace_start_time)
            trigger.set_trigger_times(triggered_times + channel_trace_start_time)
            logger.info("Station has passed trigger, trigger time is {:.1f} ns".format(
                trigger.get_trigger_time() / units.ns))

        station.set_trigger(trigger)

        self.__t += time.time() - t

    def end(self):
        from datetime import timedelta
        dt = timedelta(seconds=self.__t)
        logger.info("total time used by this module is {}".format(dt))
        return dt


def _get_threshold_channel(threshold, channel_id):
    """ Returns channel specific threshold if threshold is a dict, otherwise returns threshold """
    if isinstance(threshold, dict):
        return threshold[channel_id]
    elif isinstance(threshold, (int, float)):
        return threshold
    else:
        raise TypeError(f"Threshold must be a int/float or dict, not {type(threshold)}")