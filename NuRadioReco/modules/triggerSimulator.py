from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.trigger import SimpleThresholdTrigger
import numpy as np
import time
import logging
logger = logging.getLogger('triggerSimulator')


class triggerSimulator:
    """
    Calculate a very simple amplitude trigger.
    """

    def __init__(self):
        self.__t = 0
        self.begin()
        self.warning("the usage of this module is deprecated, plase use 'modules/trigger/simpleThreshold.py instead'")

    def begin(self, debug=False):
        self.__debug = debug

    def run(self, evt, station, det,
            threshold=60 * units.mV,
            number_concidences=1,
            triggered_channels=None,
            trigger_name='default_simple_threshold'):
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

        for channel in station.iter_channels():
            channel_id = channel.get_id()
            if triggered_channels is not None and channel_id not in triggered_channels:
                logger.debug("skipping channel {}".format(channel_id))
                continue
            trace = channel.get_trace()
            maximum = np.max(np.abs(trace))
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

        trigger = SimpleThresholdTrigger(trigger_name, threshold, triggered_channels, number_concidences)
        if coincidences >= number_concidences:
            trigger.set_triggered(True)
            logger.debug("station has triggered")
        else:
            trigger.set_triggered(False)
            logger.debug("station has NOT triggered")
        station.set_trigger(trigger)

        self.__t += time.time() - t

    def end(self):
        from datetime import timedelta
        logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        logger.info("total time used by this module is {}".format(dt))
        return dt
