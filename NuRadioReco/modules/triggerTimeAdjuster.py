import numpy as np
import logging
from NuRadioReco.utilities import units

logger = logging.getLogger('triggerTimeAdjuster')

class triggerTimeAdjuster:
    """
    Modifies channel traces to simulate the effects of the trigger
    """
    def __init__(self, trigger_name):
        self.__trigger_name = trigger_name
        """
        trigger_name: string
            name of the trigger that should be used. The corresponding trigger module must be run beforehand.
            If the trigger does not exist or did not trigger, this module will do nothing
        """
    
    def begin(self, pre_trigger_time = 50.*units.ns):
        """
        Setup
        ----
        pre_trigger_time: float
            Amount of time that should be stored in the channel trace before the trigger. If the channel trace is long
            enough, it will be cut accordingly. Otherwise, it will be rolled.
        cut_trace: bool
            If true, the trace will be cut to the length specified in the detector description
        """
        self.__pre_trigger_time = pre_trigger_time
    
    def run(self, event, station, detector):
        trigger = station.get_trigger(self.__trigger_name)
        if trigger.has_triggered():
            trigger_time = trigger.get_trigger_time()
            for channel in station.iter_channels():
                trace = channel.get_trace()
                trace_length = len(trace)
                number_of_samples = int(detector.get_number_of_samples(station.get_id(), channel.get_id()) * channel.get_sampling_rate() / detector.get_sampling_frequency(station.get_id(), channel.get_id()))
                if number_of_samples > trace.shape[0]:
                    logger.error("Input has fewer samples than desired output. Channels has only {} samples but {} samples are requested.".format(
                        trace.shape[0], number_of_samples))
                    raise StandardError
                else:
                    sampling_rate = channel.get_sampling_rate()
                    trigger_time_sample = int(np.round(trigger_time * sampling_rate))
                    samples_before_trigger = int(self.__pre_trigger_time * sampling_rate)
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
                    trace = trace[cut_samples_beginning:(number_of_samples + cut_samples_beginning)]
                    channel.set_trace(trace, channel.get_sampling_rate())
                    channel.set_trace_start_time(channel.get_trace_start_time() + rel_station_time_samples / channel.get_sampling_rate())
            trigger.set_trigger_time(self.__pre_trigger_time)
        else:
            logger.debug('Trigger {} has not triggered. Channel timings will not be changed.'.format(self.__trigger_name))


