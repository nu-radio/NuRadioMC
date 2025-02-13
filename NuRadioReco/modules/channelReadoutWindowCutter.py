from NuRadioReco.modules.base.module import register_run
import NuRadioReco.framework.channel
from NuRadioReco.utilities import units

import numpy as np
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
        self.__sampling_rate_warning_issued = False
        self.begin()

    def begin(self):
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
            n_samples = detector.get_number_of_samples(station.get_id(), channel.get_id())
            self.__check_sampling_rates(channel, detector_sampling_rate, sampling_rate, n_samples)


            # this should ensure that 1) the number of samples is even and
            # 2) resampling to the detector sampling rate results in the correct number of samples
            # (note that 2) can only be guaranteed if the detector sampling rate is lower than the
            # current sampling rate)
            number_of_samples = int(
                2 * np.ceil(n_samples / 2 * sampling_rate / detector_sampling_rate))

            trace = channel.get_trace()
            if number_of_samples > trace.shape[0]:
                logger.error((
                    "Input has fewer samples than desired output. "
                    "Channels has only {} samples but {} samples are requested.").format(
                    trace.shape[0], number_of_samples))
                raise AttributeError

            # windowed_channel = get_empty_channel(
            #     station.get_id(), channel.get_id(), detector, trigger, sampling_rate)
            # windowed_channel.add_to_trace(channel)

            # channel.set_trace(windowed_channel.get_trace(), windowed_channel.get_sampling_rate())
            # channel.set_trace_start_time(windowed_channel.get_trace_start_time())

            channel_id = channel.get_id()
            pre_trigger_time = trigger.get_pre_trigger_time_channel(channel_id)

            pre_trigger_time_channel = trigger_time - pre_trigger_time - channel.get_trace_start_time()
            
            # throw warinings or errors depending on if the readout window is outside or too far outside the trace
            trace_length = len(trace)
            if pre_trigger_time_channel < -pre_trigger_time:
                msg = ("Start of the readout window is more than pre_trigger_time before the start of the "
                       "trace (trigger time = {}ns, pre-trigger time = {}ns, start of trace {}ns, requested "
                       "time before trace = {}ns), this would result in rolling over the edge of the trace "
                       "and is not the intended use of this function").format(
                            trigger_time, pre_trigger_time, channel.get_trace_start_time(), pre_trigger_time_channel)
                logger.error(msg)
                raise AttributeError(msg)
            elif pre_trigger_time_channel < 0:
                msg = ("Start of the readout window is before the start of the trace (trigger time = {}ns, "
                       "pre-trigger time = {}ns, start of trace {}ns, requested time before trace = {}ns), "
                       "the trace will be rolled over the edge to fit in the readout window").format(
                            trigger_time, pre_trigger_time, channel.get_trace_start_time(), pre_trigger_time_channel)
                logger.warning(msg)
            elif trigger_time > channel.get_trace_start_time() + trace_length / sampling_rate:
                msg = ("Trigger time is after the end of the trace (trigger time = {}ns, end of trace {}ns), "
                       "this would result in rolling over the edge of the trace and is not the intended use "
                       "of this function").format(
                            trigger_time, channel.get_trace_start_time() + trace_length / sampling_rate,
                            pre_trigger_time_channel + number_of_samples / sampling_rate - trace_length / sampling_rate)
                logger.error(msg)
                raise AttributeError(msg)
            elif pre_trigger_time_channel + number_of_samples / sampling_rate > trace_length / sampling_rate:
                msg = ("End of the readout window is outside the end of the trace (trigger time = {}ns, "
                       "pre-trigger time = {}ns, length of readout window {}ns, end of trace {}, requested time after trace = {}ns), "
                       "the trace will be rolled over the edge to fit in the readout window").format(
                            trigger_time, pre_trigger_time, number_of_samples/sampling_rate, channel.get_trace_start_time() + trace_length/sampling_rate,
                            pre_trigger_time_channel + number_of_samples / sampling_rate - trace_length / sampling_rate)
                logger.warning(msg)

            # "roll" the start of the readout window to the start of the trace
            channel.apply_time_shift(-pre_trigger_time_channel, silent=True)

            # cut the trace
            trace = channel.get_trace()
            trace = trace[:number_of_samples]

            channel.set_trace(trace, channel.get_sampling_rate())
            channel.set_trace_start_time(trigger_time - pre_trigger_time)


    def __check_sampling_rates(self, channel, detector_sampling_rate, channel_sampling_rate, n_samples):
        if not self.__sampling_rate_warning_issued: # we only issue this warning once
            if not np.isclose(detector_sampling_rate, channel_sampling_rate):
                logger.warning(
                    'channelReadoutWindowCutter was called, but the channel sampling rate '
                    f'({channel_sampling_rate/units.GHz:.3f} GHz) is not equal to '
                    f'the target detector sampling rate ({detector_sampling_rate/units.GHz:.3f} GHz). '
                    'Traces may not have the correct trace length after resampling.'
                )
                self.__sampling_rate_warning_issued = True


def get_empty_channel(station_id, channel_id, detector, trigger, sampling_rate):
    channel = NuRadioReco.framework.channel.Channel(channel_id)

    # Get the correct number of sample for the final sampling rate
    sampling_rate_ratio = sampling_rate / detector.get_sampling_frequency(station_id, channel_id)
    n_samples = int(round(detector.get_number_of_samples(station_id, channel_id) * sampling_rate_ratio))

    # get the correct trace start time taking into account different `pre_trigger_times`
    channel_trace_start_time = trigger.get_trigger_time() - trigger.get_pre_trigger_time_channel(channel_id)

    channel.set_trace(np.zeros(n_samples), sampling_rate)
    channel.set_trace_start_time(channel_trace_start_time)

    return channel
