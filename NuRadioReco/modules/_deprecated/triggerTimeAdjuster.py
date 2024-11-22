from NuRadioReco.modules.base.module import register_run
import numpy as np
import logging
import warnings
from NuRadioReco.utilities import units

logger = logging.getLogger('NuRadioReco.triggerTimeAdjuster')


class triggerTimeAdjuster:
    """
    Modifies channel traces to simulate the effects of the trigger

    The trace is cut to the length defined in the detector description relative to the trigger time.
    If no trigger exists, nothing is done.
    """

    def __init__(self, log_level=logging.NOTSET):
        logger.setLevel(log_level)
        self.__sampling_rate_warning_issued = False
        self.begin()
        warnings.warn("triggerTimeAdjuster is deprecated and will be removed soon. In most cased you can safely delete the application "
                       "of this module as it is automatically applied in NuRadioMC simulations. If you really need to use this module, "
                       "please use the channelReadoutWindowCutter module instead.", DeprecationWarning)

    def begin(self):
        pass


    def get_pre_trigger_time(self, trigger_name, channel_id):
        """ Get the pre_trigger_time for a given trigger_name and channel_id """
        if isinstance(self.__pre_trigger_time, float):
            return self.__pre_trigger_time

        pre_trigger_time = self.__pre_trigger_time
        while isinstance(pre_trigger_time, dict):
            if trigger_name in pre_trigger_time: # keys are different triggers
                pre_trigger_time = pre_trigger_time[trigger_name]
            elif channel_id in pre_trigger_time: # keys are channel_ids
                pre_trigger_time = pre_trigger_time[channel_id]
            else:
                logger.error(
                    'pre_trigger_time was specified as a dictionary, '
                    f'but neither the trigger_name {trigger_name} '
                    f'nor the channel id {channel_id} are present as keys'
                    )
                raise KeyError

        return pre_trigger_time

    @register_run()
    def run(self, event, station, detector, mode='sim_to_data'):
        """
        Run the trigger time adjuster.

        This module can be used either to 'cut' the simulated traces into
        the appropriate readout windows, or to adjust the trace start times
        of simulated / real data to account for the different trigger readout
        delays.

        If multiple triggers exist, the primary trigger is used. If multiple
        primary triggers exist, an error is raised.
        If no primary trigger exists, the trigger with the earliest trigger time
        is defined as the primary trigger and used to set the readout windows.

        Parameters
        ----------
        event: `NuRadioReco.framework.event.Event`

        station: `NuRadioReco.framework.base_station.Station`

        detector: `NuRadioReco.detector.detector.Detector`

        mode: 'sim_to_data' (default) | 'data_to_sim'
            If 'sim_to_data', cuts the (arbitrary-length) simulated traces
            to the appropriate readout windows.
            If 'data_to_sim', looks through all triggers in the station and adjusts the
            trace_start_time according to the different readout delays

            If the ``trigger_name`` was specified in the ``begin`` function,
            only this trigger is considered.
        """
        warnings.warn("triggerTimeAdjuster is deprecated and will be removed soon. In most cased you can safely delete the application "
                       "of this module as it is automatically applied in NuRadioMC simulations. If you really need to use this module, "
                       "please use the channelReadoutWindowCutter module instead.", DeprecationWarning)
        counter = 0
        for i, (name, instance, kwargs) in enumerate(event.iter_modules(station.get_id())):
            if name == 'triggerTimeAdjuster':
                if(kwargs['mode'] == mode):
                    counter += 1
        if counter > 1:
            logger.warning('triggerTimeAdjuster was called twice with the same mode. '
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

        if trigger is None:
            logger.info('No trigger found! Channel timings will not be changed.')
            return

        if mode == 'sim_to_data':
            if trigger.has_triggered():
                trigger_time = trigger.get_trigger_time()
                for channel in station.iter_channels():
                    trigger_time_channel = trigger_time - channel.get_trace_start_time()
                    # if trigger_time_channel == 0:
                    #     logger.warning(f"the trigger time is equal to the trace start time for channel {channel.get_id()}. This is likely because this module was already run on this station. The trace will not be changed.")
                    #     continue

                    trace = channel.get_trace()
                    trace_length = len(trace)
                    detector_sampling_rate = detector.get_sampling_frequency(station.get_id(), channel.get_id())
                    sampling_rate = channel.get_sampling_rate()
                    self.__check_sampling_rates(detector_sampling_rate, sampling_rate)

                    # this should ensure that 1) the number of samples is even and
                    # 2) resampling to the detector sampling rate results in the correct number of samples
                    # (note that 2) can only be guaranteed if the detector sampling rate is lower than the
                    # current sampling rate)
                    number_of_samples = int(
                        2 * np.ceil(
                            detector.get_number_of_samples(station.get_id(), channel.get_id()) / 2
                            * sampling_rate / detector_sampling_rate
                        ))

                    if number_of_samples > trace.shape[0]:
                        logger.error("Input has fewer samples than desired output. Channels has only {} samples but {} samples are requested.".format(
                            trace.shape[0], number_of_samples))
                        raise AttributeError
                    else:
                        trigger_time_sample = int(np.round(trigger_time_channel * sampling_rate))
                        # logger.debug(f"channel {channel.get_id()}: trace_start_time = {channel.get_trace_start_time():.1f}ns, trigger time channel {trigger_time_channel/units.ns:.1f}ns,  trigger time sample = {trigger_time_sample}")
                        channel_id = channel.get_id()
                        pre_trigger_time = trigger.get_pre_trigger_time_channel(channel_id)
                        samples_before_trigger = int(pre_trigger_time * sampling_rate)
                        cut_samples_beginning = 0
                        if(samples_before_trigger <= trigger_time_sample):
                            cut_samples_beginning = trigger_time_sample - samples_before_trigger
                            roll_by = 0
                            if(cut_samples_beginning + number_of_samples > trace_length):
                                logger.warning("trigger time is sample {} but total trace length is only {} samples (requested trace length is {} with an offest of {} before trigger). To achieve desired configuration, trace will be rolled".format(
                                    trigger_time_sample, trace_length, number_of_samples, samples_before_trigger))
                                roll_by = cut_samples_beginning + number_of_samples - trace_length  # roll_by is positive
                                trace = np.roll(trace, -1 * roll_by)
                                cut_samples_beginning -= roll_by
                            rel_station_time_samples = cut_samples_beginning + roll_by
                        elif(samples_before_trigger > trigger_time_sample):
                            roll_by = -trigger_time_sample + samples_before_trigger
                            logger.warning(f"trigger time is before 'trigger offset window' (requested samples before trigger = {samples_before_trigger}," \
                                           f"trigger time sample = {trigger_time_sample}), the trace needs to be rolled by {roll_by} samples first" \
                                            f" = {roll_by / sampling_rate/units.ns:.2f}ns")
                            trace = np.roll(trace, roll_by)

                        # shift trace to be in the correct location for cutting
                        trace = trace[cut_samples_beginning:(number_of_samples + cut_samples_beginning)]
                        channel.set_trace(trace, channel.get_sampling_rate())
                        channel.set_trace_start_time(trigger_time)
                        # channel.set_trace_start_time(channel.get_trace_start_time() + rel_station_time_samples / channel.get_sampling_rate())
                        # logger.debug(f"setting trace start time to {channel.get_trace_start_time() + rel_station_time_samples / channel.get_sampling_rate():.0f} = {channel.get_trace_start_time():.0f} + {rel_station_time_samples / channel.get_sampling_rate():.0f}")

        elif mode == 'data_to_sim':
            for channel in station.iter_channels():
                pre_trigger_time = trigger.get_pre_trigger_time_channel(channel.get_id())
                channel.set_trace_start_time(channel.get_trace_start_time()-pre_trigger_time)
        else:
            raise ValueError(f"Argument '{mode}' for mode is not valid. Options are 'sim_to_data' or 'data_to_sim'.")

    def __check_sampling_rates(self, detector_sampling_rate, channel_sampling_rate):
        if not self.__sampling_rate_warning_issued: # we only issue this warning once
            if not np.isclose(detector_sampling_rate, channel_sampling_rate):
                logger.warning(
                    'triggerTimeAdjuster was called, but the channel sampling rate '
                    f'({channel_sampling_rate/units.GHz:.3f} GHz) is not equal to '
                    f'the target detector sampling rate ({detector_sampling_rate/units.GHz:.3f} GHz). '
                    'Traces may not have the correct trace length after resampling.'
                )
                self.__sampling_rate_warning_issued = True
