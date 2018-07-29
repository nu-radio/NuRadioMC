from NuRadioReco.utilities import units
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

    def begin(self, debug=False):
        self.__debug = debug

    def run(self, evt, station, det,
            threshold=60 * units.mV,
            number_concidences=1,
            triggered_channels=None):
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
        """
        t = time.time()

        coincidences = 0

        for channel in station.get_channels():
            channel_id = channel.get_id()
            if triggered_channels is not None and channel_id not in triggered_channels:
                continue
            trace = channel.get_trace()
            if np.max(np.abs(trace)) > threshold:
                coincidences += 1
            if self.__debug:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.plot(trace)
                print np.std(trace)
                plt.axhline(threshold)
                plt.show()

        if coincidences >= number_concidences:
            station.set_triggered(True)
        else:
            station.set_triggered(False)

        self.__t += time.time() - t

    def end(self):
        from datetime import timedelta
        logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        logger.info("total time used by this module is {}".format(dt))
        return dt
