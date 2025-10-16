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
                          time_coincidence=5 * units.ns, dt=1 * units.ns,
                          step=1, align_strides_to_start=False):
    """
    Calculates a high low trigger in a time coincidence window

    Parameters
    ----------
    trace: array of floats
        the signal trace
    high_threshold: float
        the high threshold
    low_threshold: float
        the low threshold
    time_coincidence: float (default: 5 ns)
        the time coincidence window between a high + low
    dt: float (default: 1 ns)
        the width of a time bin (inverse of sampling rate)
    step: int (default: 1_
        stride length for sampling rate and clock rate mismatch in trigger logic
    align_strides_to_start: bool (default: False)
        If true, the trace represents real detector data and will force the striding
        to start at the beginning of the trace without padding. If false, the traces
        will be zero-padded at the beginning of the trace. This allows a trigger at
        beginning of the trace to be associated with the correct trigger time.

    Returns
    -------
    triggered bins: array of bools
        the bins where the trigger condition is satisfied
    """

    if trace.dtype != type(high_threshold):
        logger.error(f"The trace ({trace.dtype}) and the threshold ({type(high_threshold)}) must have the same type")
        raise TypeError(f"The trace ({trace.dtype}) and the threshold ({type(high_threshold)}) must have the same type")

    n_bins_coincidence = int(np.round(time_coincidence / dt))

    if not align_strides_to_start:
        # Pad trace so trigger bin matches with sample index
        padded_trace = np.pad(trace, (n_bins_coincidence - 1, 0), "constant")
    else:
        padded_trace = trace

    num_frames = int((len(padded_trace) - n_bins_coincidence) / step)
    num_real_frames = int(len(trace) / step)

    logger.debug("length of trace {} samples, coincidence window {}, num window bins {}".format(
        len(padded_trace), n_bins_coincidence, num_frames))

    # This transforms the trace (n samples) to an array with shape (num_frames, window) where each frame
    # is extracted from the trace in steps of sample intervals
    # Ex. step=2, window=4, trace=[1, 2, 3, 4, 5, 6, 7, 8], trace_windowed=[[1, 2, 3, 4], [3, 4, 5, 6], ...]
    trace_windowed = np.lib.stride_tricks.as_strided(
        padded_trace, (num_frames, n_bins_coincidence),
        (padded_trace.strides[0] * step, padded_trace.strides[0]),
        writeable=False)

    # Find high and low triggering windows
    trace_high = np.any(trace_windowed >= high_threshold, axis=1)
    trace_low = np.any(trace_windowed <= low_threshold, axis=1)
    trace_high_low = trace_high & trace_low

    # Keep as many samples as the original trace or cut short to keep triggers in the trace length.
    trace_high_low = trace_high_low[:num_real_frames]

    return trace_high_low


def get_majority_logic(tts, number_of_coincidences=2, time_coincidence=32 * units.ns, dt=1 * units.ns,
                       step=1, align_strides_to_start=False):
    """
    Calculates a majority logic trigger

    Parameters
    ----------
    tts: array/list of array of bools
        an array of bools that indicate a single channel trigger per channel
    number_of_coincidences: int (default: 2)
        the number of coincidences between channels
    time_coincidence: float (default: 32 ns)
        the time coincidence window between channels
    dt: float (default: 1ns)
        the width of a time bin (inverse of sampling rate)
    step: int (default: 1)
        stride length for sampling rate and clock rate mismatch in trigger logic
    align_strides_to_start: bool (default: False)
        If true, the trace represents real detector data and will force the striding
        to start at the beginning of the trace without padding. If false, the traces
        will be zero-padded at the beginning of the trace. This allows a trigger at
        beginning of the trace to be associated with the correct trigger time.

    Returns
    -------
    triggerd: bool
        returns True if majority logic is fulfilled
    triggerd_bins: array of ints
        the bins that fulfilled the trigger
    triggered_times: array of floats
        the trigger times
    """

    n_bins_coincidence = int(np.round(time_coincidence / dt))
    n = len(tts[0])

    if n_bins_coincidence > n:  # reduce coincidence window to maximum trace length
        n_bins_coincidence = n
        logger.debug("specified coincidence window longer than tracelength, reducing coincidence window to trace length")

    for i in range(len(tts)):
        if not align_strides_to_start:
            trace = np.pad(tts[i], (n_bins_coincidence - 1, 0), "constant")
        else:
            trace = tts[i]

        num_frames = int((len(trace) - n_bins_coincidence) / step)

        # Use the stride trick again.
        trace_windowed = np.lib.stride_tricks.as_strided(
            trace, (num_frames, n_bins_coincidence),
            (trace.strides[0] * step, trace.strides[0]),
            writeable=False)

        tts[i] = np.any(trace_windowed, axis=1)

    tt = np.array(tts)
    ttt = np.sum(np.array(tt), axis=0) >= number_of_coincidences
    triggered_bins = np.atleast_1d(np.squeeze(np.argwhere(ttt))) * step

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
            adc_output='voltage',
            step=1,
            align_strides_to_start=False,
            pre_trigger_time=None):
        """
        Calculates the high-low coincidence trigger for an event.

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

        step: int
            stride length for sampling rate and clock rate mismatch in trigger logic
        align_strides_to_start: bool (default: False)
            If true, the trace represents real detector data and will force the striding
            to start at the beginning of the trace without padding. If false, the traces
            will be zero-padded at the beginning of the trace. This allows a trigger at
            beginning of the trace to be associated with the correct trigger time.
        pre_trigger_time: float or dict of floats
            Defines the amount of trace recorded before the trigger time. This module does not cut the traces,
            but this trigger property is later used to trim traces accordingly.
            if a dict is given, the keys are the channel_ids, and the value is the pre_trigger_time between the
            start of the trace and the trigger time.
            if only a float is given, the same pre_trigger_time is used for all channels
            If none, the default value of the HighLowTrigger class is used, which is currently 55ns.
        Returns
        -------
        has_triggered: bool
            True if the trigger condition was met
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
                    high_low_window, dt, step, align_strides_to_start)

                if np.any(triggerd_bins):
                    channels_that_passed_trigger.append(channel.get_id())

                triggerd_bins_channels.append(triggerd_bins)
                logger.debug("channel {}, len(triggerd_bins) = {}".format(channel_id, len(triggerd_bins)))

            if len(triggerd_bins_channels):
                has_triggered, triggered_bins, triggered_times = get_majority_logic(
                    triggerd_bins_channels, number_concidences, coinc_window, dt * step, 1, align_strides_to_start)
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

        kwargs = {}
        if pre_trigger_time is not None:
            kwargs['pre_trigger_times'] = pre_trigger_time
        trigger = HighLowTrigger(trigger_name, threshold_high, threshold_low, high_low_window,
                                 coinc_window, channels=triggered_channels, number_of_coincidences=number_concidences,
                                 **kwargs)

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

        return has_triggered

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
