from NuRadioReco.modules.base.module import register_run
import numpy as np
import logging
from NuRadioReco.utilities import units

logger = logging.getLogger('NuRadioReco.triggerTimeAdjuster')


class triggerTimeAdjuster:
    """
    Modifies channel traces to simulate the effects of the trigger

    The trace is cut to the length defined in the detector description relative to the trigger time.
    If no trigger exists, nothing is done.
    """

    def __init__(self, log_level=logging.WARNING):
        logger.setLevel(log_level)
        self.__trigger_name = None
        self.__pre_trigger_time = None
        self.begin()

    def begin(self, trigger_name=None, pre_trigger_time=55. * units.ns):
        """
        Parameters
        ----------
        trigger_name: string or None
            name of the trigger that should be used.
            If trigger_name is None, the trigger with the smallest trigger_time will be used.
            If a name is given, corresponding trigger module must be run beforehand.
            If the trigger does not exist or did not trigger, this module will do nothing
        pre_trigger_time: float or dict
            Amount of time that should be stored in the channel trace before the trigger. 
            If the channel trace is long enough, it will be cut accordingly. 
            Otherwise, it will be rolled.

            If given as a float, the same ``pre_trigger_time`` will be used for all channels.
            If a dict, the keys should be ``channel_id``, and the values the ``pre_trigger_time`` 
            to use for each channel. Alternatively, the keys should be the ``trigger_name``,
            and the values either a float or a dictionary with (``channel_id``, ``pre_trigger_time``)
            pairs.
        """
        self.__trigger_name = trigger_name
        self.__pre_trigger_time = pre_trigger_time
        self.__sampling_rate_warning_issued = False

    @register_run()
    def run(self, event, station, detector, mode='sim_to_data'):
        """
        Run the trigger time adjuster.

        This module can be used either to 'cut' the simulated traces into
        the appropriate readout windows, or to adjust the trace start times 
        of simulated / real data to account for the different trigger readout
        delays.

        Parameters
        ----------
        event: NuRadioReco.framework.event.Event

        station: NuRadioReco.framework.base_station.Station

        detector: NuRadioReco.detector.detector.Detector

        mode: 'sim_to_data' (default) | 'data_to_sim'
            If 'sim_to_data', cuts the (arbitrary-length) simulated traces
            to the appropriate readout windows. If 'data_to_sim',
            looks through all triggers in the station and adjusts the
            trace_start_time according to the different readout delays

            If the ``trigger_name`` was specified in the ``begin`` function,
            only this trigger is considered.
        
        """
        if mode == 'sim_to_data':
            trigger = None
            if self.__trigger_name is not None:
                trigger = station.get_trigger(self.__trigger_name)
            else:
                min_trigger_time = None
                for trig in station.get_triggers().values():
                    if(trig.has_triggered()):
                        if min_trigger_time is None or trig.get_trigger_time() < min_trigger_time:
                            min_trigger_time = trig.get_trigger_time()
                            trigger = trig
                if(min_trigger_time is not None):
                    logger.info(f"minimum trigger time is {min_trigger_time/units.ns:.2f}ns")
            if trigger is None:
                logger.info('No trigger found! Channel timings will not be changed.')
                return
            if trigger.has_triggered():
                trigger_time = trigger.get_trigger_time()
                store_pre_trigger_time = {} # we also want to save the used pre_trigger_time
                for channel in station.iter_channels():
                    trigger_time_channel = trigger_time - channel.get_trace_start_time()

                    trace = channel.get_trace()
                    trace_length = len(trace)
                    detector_sampling_rate = detector.get_sampling_frequency(station.get_id(), channel.get_id())
                    number_of_samples = int(
                        2 * np.ceil( # this should ensure that 1) the number of samples is even and 2) resampling to the detector sampling rate results in the correct number of samples (note that 2) can only be guaranteed if the detector sampling rate is lower than the current sampling rate)
                            detector.get_number_of_samples(station.get_id(), channel.get_id()) / 2
                            * channel.get_sampling_rate() / detector_sampling_rate
                        ))
                    if number_of_samples > trace.shape[0]:
                        logger.error("Input has fewer samples than desired output. Channels has only {} samples but {} samples are requested.".format(
                            trace.shape[0], number_of_samples))
                        raise AttributeError
                    else:
                        sampling_rate = channel.get_sampling_rate()
                        self.__check_sampling_rates(detector_sampling_rate, sampling_rate)
                        trigger_time_sample = int(np.round(trigger_time_channel * sampling_rate))
                        # logger.debug(f"channel {channel.get_id()}: trace_start_time = {channel.get_trace_start_time():.1f}ns, trigger time channel {trigger_time_channel/units.ns:.1f}ns,  trigger time sample = {trigger_time_sample}")
                        pre_trigger_time = self.__pre_trigger_time
                        channel_id = channel.get_id()
                        trigger_name = trigger.get_name()
                        while isinstance(pre_trigger_time, dict):
                            if trigger_name in pre_trigger_time.keys(): # keys are different triggers
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
                        samples_before_trigger = int(pre_trigger_time * sampling_rate)
                        rel_station_time_samples = 0
                        cut_samples_beginning = 0
                        if(samples_before_trigger < trigger_time_sample):
                            cut_samples_beginning = trigger_time_sample - samples_before_trigger
                            roll_by = 0
                            if(cut_samples_beginning + number_of_samples > trace_length):
                                logger.info("trigger time is sample {} but total trace length is only {} samples (requested trace length is {} with an offest of {} before trigger). To achieve desired configuration, trace will be rolled".format(
                                    trigger_time_sample, trace_length, number_of_samples, samples_before_trigger))
                                roll_by = cut_samples_beginning + number_of_samples - trace_length  # roll_by is positive
                                trace = np.roll(trace, -1 * roll_by)
                                cut_samples_beginning -= roll_by
                            rel_station_time_samples = cut_samples_beginning + roll_by
                        elif(samples_before_trigger > trigger_time_sample):
                            roll_by = -trigger_time_sample + samples_before_trigger
                            logger.info(
                                "trigger time is before 'trigger offset window', the trace needs to be rolled by {} samples first".format(roll_by))
                            trace = np.roll(trace, roll_by)
                            trigger_time_sample -= roll_by
                            rel_station_time_samples = -roll_by

                        # shift trace to be in the correct location for cutting
                        # logger.debug(f"cutting trace to {cut_samples_beginning}-{number_of_samples + cut_samples_beginning} samples")
                        trace = trace[cut_samples_beginning:(number_of_samples + cut_samples_beginning)]
                        channel.set_trace(trace, channel.get_sampling_rate())
                        channel.set_trace_start_time(trigger_time)
                        store_pre_trigger_time[channel_id] = pre_trigger_time
                        # channel.set_trace_start_time(channel.get_trace_start_time() + rel_station_time_samples / channel.get_sampling_rate())
                        # logger.debug(f"setting trace start time to {channel.get_trace_start_time() + rel_station_time_samples / channel.get_sampling_rate():.0f} = {channel.get_trace_start_time():.0f} + {rel_station_time_samples / channel.get_sampling_rate():.0f}")
                
                # store the used pre_trigger_times
                trigger.set_pre_trigger_times(store_pre_trigger_time)
            
            else:
                logger.debug('Trigger {} has not triggered. Channel timings will not be changed.'.format(self.__trigger_name))
        elif mode == 'data_to_sim':
            if self.__trigger_name is not None:
                triggers = [station.get_trigger(self.__trigger_name)]
            else:
                triggers = station.get_triggers().values()

            pre_trigger_times = [trigger.get_pre_trigger_times() for trigger in triggers]
            if np.sum([dt is not None for dt in pre_trigger_times]) > 1:
                logger.warning(
                    'More than one trigger claims to have adjusted the pre_trigger_times. '
                    'Normally, only one trigger should set pre_trigger_times. '
                    )

            for pre_trigger_time in pre_trigger_times:
                if pre_trigger_time is not None:
                    for channel in station.iter_channels():
                        channel.set_trace_start_time(channel.get_trace_start_time()-pre_trigger_time[channel.get_id()])
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
