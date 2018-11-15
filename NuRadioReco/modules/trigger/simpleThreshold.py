from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.trigger import SimpleThresholdTrigger
import numpy as np
import time
import logging
logger = logging.getLogger('simpleThresholdTrigger')


class triggerSimulator:
    """
    Calculate a very simple amplitude trigger.
    """

    def __init__(self):
        self.__t = 0
        self.begin()

    def begin(self, debug=False, pre_trigger_time=100 * units.ns):
        self.__pre_trigger_time = pre_trigger_time
        self.__debug = debug

    def run(self, evt, station, det,
            threshold=60 * units.mV,
            number_concidences=1,
            triggered_channels=None,
            trigger_name='default_simple_threshold',
            cut_trace=False):
        """
        simulate simple trigger logic, no time window, just threshold in all channels

        Parameters
        ----------
        number_concidences: int
            number of channels that are requried in coincidence to trigger a station
        threshold: float
            threshold above (or below) a trigger is issued, absolute amplitude
        triggered_channels: array of ints or None
            channels ids that are triggered on, if None trigger will run on all channels
        trigger_name: string
            a unique name of this particular trigger
        """
        t = time.time()
        coincidences = 0
        max_signal = 0
        trigger_time_sample = 99999999999

        for channel in station.iter_channels():
            channel_id = channel.get_id()
            if triggered_channels is not None and channel_id not in triggered_channels:
                logger.debug("skipping channel {}".format(channel_id))
                continue
            trace = channel.get_trace()
            index = np.argmax(np.abs(trace))
            trigger_time_sample = min(index, trigger_time_sample)
            maximum = np.abs(trace)[index]
            max_signal = max(max_signal, maximum)
            if maximum > threshold:
                coincidences += 1
            if self.__debug:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.plot(trace)
                plt.axhline(threshold)
                plt.show()

        station.set_parameter(stnp.channels_max_amplitude, max_signal)

        trigger = SimpleThresholdTrigger(trigger_name, threshold, triggered_channels,
                                         number_concidences)
        if coincidences >= number_concidences:
            trigger.set_triggered(True)
            logger.debug("station has triggered")
        else:
            trigger.set_triggered(False)
            logger.debug("station has NOT triggered")
        station.set_trigger(trigger)

        if not cut_trace:
            self.__t += time.time() - t
            return

        # now cut trace to the correct number of samples
        # assuming that all channels have the same trace length
        for channel in station.iter_channels():
            trace = channel.get_trace()
            trace_length = len(trace)
            number_of_samples = int(det.get_number_of_samples(station.get_id(), channel.get_id()) * channel.get_sampling_rate() / det.get_sampling_frequency(station.get_id(), channel.get_id()))
            if number_of_samples > trace.shape[0]:
                logger.error("Input has fewer samples than desired output. Channels has only {} samples but {} samples are requested.".format(
                    trace.shape[0], number_of_samples))
#                 new_trace = np.zeros(self.number_of_samples)
#                 new_trace[:trace.shape[0]] = trace
#                 change_time = 0
                raise StandardError
#             elif number_of_samples == trace.shape[0]:
#                 logger.info("Channel {} already at desired length, nothing done.".format(channel.get_id()))
            else:
                sampling_rate = channel.get_sampling_rate()
                samples_before_trigger = int(self.__pre_trigger_time * sampling_rate)
                rel_station_time_samples = 0
                cut_samples_beginning = 0
                if(samples_before_trigger < trigger_time_sample):
                    cut_samples_beginning = trigger_time_sample - samples_before_trigger
                    if(cut_samples_beginning + number_of_samples > trace_length):
                        logger.warning("trigger time is sample {} but total trace length is only {} samples (requested trace length is {} with an offest of {} before trigger). To achieve desired configuration, trace will be rolled".format(
                            trigger_time_sample, trace_length, number_of_samples, samples_before_trigger))
                        roll_by = cut_samples_beginning + number_of_samples - trace_length  # roll_by is positive
                        trace = np.roll(trace, -1 * roll_by)
                        cut_samples_beginning -= roll_by
                    rel_station_time_samples = cut_samples_beginning
                elif(samples_before_trigger > trigger_time_sample):
                    roll_by = trigger_time_sample - samples_before_trigger
                    logger.warning(
                        "trigger time is before 'trigger offset window', the trace needs to be rolled by {} samples first".format(roll_by))
                    trace = np.roll(trace, roll_by)
                    trigger_time_sample -= roll_by
                    rel_station_time_samples = -roll_by

                # shift trace to be in the correct location for cutting
                trace = trace[cut_samples_beginning:(number_of_samples + cut_samples_beginning)]
                channel.set_trace(trace, channel.get_sampling_rate())
        try:
            logger.debug('setting ssim tation start time to {:.1f} + {:.1f}ns'.format(
                station.get_sim_station().get_trace_start_time(), (rel_station_time_samples / sampling_rate)))
            # here we assumed that all channels had the same length
            station.get_sim_station().add_trace_start_time(-rel_station_time_samples / sampling_rate)
        except:
            logger.warning("No simulation information in event, trace start time will not be added")

        self.__t += time.time() - t

    def end(self):
        from datetime import timedelta
        logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        logger.info("total time used by this module is {}".format(dt))
        return dt
