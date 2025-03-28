from NuRadioReco.modules.base.module import register_run
import NuRadioReco.framework.channel
from NuRadioReco.utilities import units, signal_processing

import numpy as np
import functools
import logging
logger = logging.getLogger('NuRadioReco.channelReadoutWindowCutter')


class channelReadoutWindowCutter:
    """
    Modifies channel traces to simulate the effects of the trigger

    The trace is cut to the length defined in the detector description relative to the trigger time.
    If no trigger exists, nothing is done.
    """

    def __init__(self, log_level=logging.NOTSET):
        logger.setLevel(log_level)
        self.begin()

    def begin(self):
        self.__sampling_rate_error_issued = False
        pass

    @register_run()
    def run(self, event, station, detector):
        """
        Cuts the traces to the readout window defined in the trigger.

        If multiple triggers exist, the primary trigger is used. If multiple
        primary triggers exist, an error is raised.
        If no primary trigger exists, the trigger with the earliest trigger time
        is defined as the primary trigger and used to set the readout windows.

        Parameters
        ----------
        event: `NuRadioReco.framework.event.Event`

        station: `NuRadioReco.framework.base_station.Station`

        detector: `NuRadioReco.detector.detector.Detector`
        """
        counter = 0
        for i, (name, instance, kwargs) in enumerate(event.iter_modules(station.get_id())):
            if name == 'channelReadoutWindowCutter':
                counter += 1
        if counter > 1:
            logger.warning('channelReadoutWindowCutter was called twice. '
                           'This is likely a mistake. The module will not be applied again.')
            return 0

        # determine which trigger to use
        # if no primary trigger exists, use the trigger with the earliest trigger time
        trigger = station.get_primary_trigger()
        if trigger is None: # no primary trigger found
            logger.debug('No primary trigger found. Using the trigger with the earliest trigger time.')
            trigger = station.get_first_trigger()
            if trigger is not None:
                logger.info(f"setting trigger {trigger.get_name()} primary because it triggered first")
                trigger.set_primary(True)

        if trigger is None or not trigger.has_triggered():
            logger.info('No trigger found (which triggered)! Channel timings will not be changed.')
            return

        trigger_time = trigger.get_trigger_time()
        for channel in station.iter_channels():

            detector_sampling_rate = detector.get_sampling_frequency(station.get_id(), channel.get_id())
            sampling_rate = channel.get_sampling_rate()
            detector_n_samples = detector.get_number_of_samples(station.get_id(), channel.get_id())

            number_of_samples, valid_sampling_rate = _get_number_of_samples(
                sampling_rate, detector_sampling_rate, detector_n_samples,
                issue_error=not self.__sampling_rate_error_issued
                )

            if not self.__sampling_rate_error_issued:
                self.__sampling_rate_error_issued = not valid_sampling_rate # this ensures the warning is printed only once

            trace = channel.get_trace()
            if number_of_samples > trace.shape[0]:
                logger.error((
                    "Input has fewer samples than desired output. "
                    "Channels has only {} samples but {} samples are requested.").format(
                    trace.shape[0], number_of_samples))
                raise AttributeError

            channel_id = channel.get_id()
            pre_trigger_time = trigger.get_pre_trigger_time_channel(channel_id)

            pre_trigger_time_channel = trigger_time - pre_trigger_time - channel.get_trace_start_time()

            # throw error if the trigger time is outside the trace or warnings if the readout window is partially outside the trace
            trace_length = len(trace)
            if (trigger_time < channel.get_trace_start_time() or
                trigger_time > channel.get_trace_start_time() + trace_length / sampling_rate):
                msg = ("Trigger time outside trace for station.channel {}.{} (trigger time = {:.2f}ns, "
                       "start of trace {:.2f}ns, end of trace {:.2f}ns, this would result in rolling over "
                       "the edge of the trace and is not the intended use of this function").format(
                        station.get_id(), channel_id, trigger_time, channel.get_trace_start_time(),
                        channel.get_trace_start_time() + trace_length / sampling_rate)
                logger.error(msg)
                raise AttributeError(msg)
            elif pre_trigger_time_channel < 0:
                msg = ("Start of the readout window is before the start of the trace for station.channel {}.{}. "
                       "This can happen with an accidental noise trigger but should not happen otherwise. "
                       "(trigger time = {:.2f}ns, pre-trigger time = {:.2f}ns, start of trace {:.2f}ns, "
                       "requested time before trace = {:.2f}ns), the trace will be rolled over the edge to "
                       "fit in the readout window").format(
                        station.get_id(), channel_id, trigger_time, pre_trigger_time,
                        channel.get_trace_start_time(), pre_trigger_time_channel)
                logger.warning(msg)
            elif pre_trigger_time_channel + number_of_samples / sampling_rate > trace_length / sampling_rate:
                msg = ("End of the readout window is outside the end of the trace for station.channel {}.{}. "
                       "(trigger time = {:.2f}ns, pre-trigger time = {:.2f}ns, start of sim. trace = {:.2f}ns, "
                       "end of sim. trace {:.2f}, length of readout window {:.2f}ns, requested time after trace = "
                       "{:.2f}ns), the trace will be rolled over the edge to fit in the readout window").format(
                        station.get_id(), channel_id, trigger_time, pre_trigger_time, channel.get_trace_start_time(),
                        channel.get_trace_start_time() + trace_length / sampling_rate, number_of_samples / sampling_rate,
                        pre_trigger_time_channel + number_of_samples / sampling_rate - trace_length / sampling_rate)
                logger.warning(msg)

            # "roll" the start of the readout window to the start of the trace
            channel.apply_time_shift(-pre_trigger_time_channel, silent=True)

            # cut the trace
            trace = channel.get_trace()
            trace = trace[:number_of_samples]

            channel.set_trace(trace, channel.get_sampling_rate())
            channel.set_trace_start_time(trigger_time - pre_trigger_time)


def _get_number_of_samples(sampling_rate, detector_sampling_rate, detector_n_samples, issue_error=True):
    """
    Calculate the number of samples that will result in the correct number of samples after resampling.

    Parameters
    ----------
    sampling_rate : float
        The current sampling rate
    detector_sampling_rate : float
        The target sampling rate
    detector_n_samples : int
        The target number of samples after resampling to `detector_sampling_rate`
    issue_error : bool, optional (default: True)
        Print an error (but does not raise one) if, after resampling, the desired
        number of samples can not be achieved.

    Returns
    -------
    number_of_samples : int
        The number of samples at sampling rate `sampling_rate`
        that will result in the desired number of samples after resampling.
    valid_sampling_rate : bool
        `True` if the sampling rate is an integer multiple of the target sampling rate,
        `False` otherwise.
    """
    # Check that the current sampling rate is an integer multiple of the target
    # sampling rate. If this is not the case, we may not be able to guarantee
    # the number of samples after resampling will be correct
    valid_sampling_rate = sampling_rate % detector_sampling_rate < 1e-8

    # this should ensure that 1) the number of samples is even and
    # 2) resampling to the detector sampling rate results in the correct number of samples
    # (note that 2) can only be guaranteed if the detector sampling rate is lower than the
    # current sampling rate)
    number_of_samples = int(
        2 * np.ceil(detector_n_samples / 2 * sampling_rate / detector_sampling_rate))

    if not valid_sampling_rate:
        # actually check if the number of samples is correct after resampling
        final_number_of_samples = _get_resampled_number_of_samples(number_of_samples, sampling_rate, detector_sampling_rate)
        valid_sampling_rate = final_number_of_samples == detector_n_samples

        if issue_error and not valid_sampling_rate:
            logger.error(
                'The current sampling rate '
                f'({sampling_rate/units.GHz:.3f} GHz) is not a multiple of '
                f'the target detector sampling rate ({detector_sampling_rate/units.GHz:.3f} GHz). '
                f'Traces will not have the correct trace length after resampling. Desired number of samples: {detector_n_samples}, '
                f'expected number of samples after resampling: {final_number_of_samples}.'
            )

    return number_of_samples, valid_sampling_rate


def get_empty_channel(station_id, channel_id, detector, trigger, sampling_rate):
    """
    Returns a channel with a trace containing zeros.

    The trace start time is given by the trigger,
    the duration of the trace is determined by the detector description, and the number of samples
    determined by the duration and the given sampling rate.

    Parameters
    ----------
    station_id: int
        The station id

    channel_id: int
        The channel id

    detector: `NuRadioReco.detector.detector.Detector`
        The detector description

    trigger: `NuRadioReco.framework.trigger.Trigger`
        The trigger that triggered the station

    sampling_rate: float
        The sampling rate of the channel
    """
    channel = NuRadioReco.framework.channel.Channel(channel_id)

    detector_n_samples = detector.get_number_of_samples(station_id, channel_id)
    detector_sampling_rate = detector.get_sampling_frequency(station_id, channel_id)

    n_samples, _ = _get_number_of_samples(
        sampling_rate, detector_sampling_rate, detector_n_samples, issue_error=False)

    # get the correct trace start time taking into account different `pre_trigger_times`
    channel_trace_start_time = trigger.get_trigger_time() - trigger.get_pre_trigger_time_channel(channel_id)

    channel.set_trace(np.zeros(n_samples), sampling_rate)
    channel.set_trace_start_time(channel_trace_start_time)

    return channel


@functools.lru_cache(maxsize=1024)
def _get_resampled_number_of_samples(number_of_samples, sampling_rate, detector_sampling_rate):
    """
    Calculate the number of samples after resampling.

    Parameters
    ----------
    number_of_samples : int
        The number of samples at the current sampling rate
    sampling_rate : float
        The current sampling rate
    detector_sampling_rate : float
        The target sampling rate

    Returns
    -------
    final_number_of_samples : int
        The number of samples after resampling
    """
    return len(signal_processing.resample(np.zeros(number_of_samples), detector_sampling_rate / sampling_rate))