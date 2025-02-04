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

            trigger_time_channel = trigger_time - channel.get_trace_start_time()
            trigger_time_sample = int(np.round(trigger_time_channel * sampling_rate))

            channel_id = channel.get_id()
            pre_trigger_time = trigger.get_pre_trigger_time_channel(channel_id)
            samples_before_trigger = int(pre_trigger_time * sampling_rate)

            trace_length = len(trace)
            cut_samples_beginning = 0
            if samples_before_trigger <= trigger_time_sample:
                cut_samples_beginning = trigger_time_sample - samples_before_trigger
                if cut_samples_beginning + number_of_samples > trace_length:
                    logger.warning(("trigger time is sample {} but total trace length is only {} "
                                    "samples (requested trace length is {} with an offest of {} "
                                    "before trigger). To achieve desired configuration, trace "
                                    "will be rolled").format(trigger_time_sample, trace_length,
                                                             number_of_samples, samples_before_trigger))

                    roll_by = cut_samples_beginning + number_of_samples - trace_length  # roll_by is positive
                    trace = np.roll(trace, -1 * roll_by)
                    cut_samples_beginning -= roll_by

            else:
                roll_by = samples_before_trigger - trigger_time_sample
                logger.warning(("trigger time is before 'trigger offset window' (requested samples before trigger = {},"
                                "trigger time sample = {}), the trace needs to be rolled by {} samples first"
                                " = {}ns").format(samples_before_trigger, trigger_time_sample, roll_by,
                                                  round(roll_by / sampling_rate/units.ns, 2)))

                trace = np.roll(trace, roll_by)

            # shift trace to be in the correct location for cutting
            trace = trace[cut_samples_beginning:(number_of_samples + cut_samples_beginning)]
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
