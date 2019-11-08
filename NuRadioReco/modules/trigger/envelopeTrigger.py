from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.trigger import EnvelopeTrigger
import numpy as np
import time
import logging
from scipy.signal import hilbert

logger = logging.getLogger('envelopeTrigger')


class triggerSimulator:
    """
    Calculate a simple amplitude trigger depending on the Hilbert-envelope.
    """

    def __init__(self):
        self.__t = 0
        self.begin()

    def begin(self, debug=False):
        self.__debug = debug

    @register_run()
    def run(self, evt, station, det,
            threshold=60 * units.mV,
            triggered_channels=None,
            trigger_name='default_envelope_trigger'):
        """
        simulate simple trigger logic based on a hilbert envelope of the trace, no time window, just threshold in all channels

        Parameters
        ----------
        threshold: float
            threshold above (or below) a trigger is issued, absolute amplitude
        triggered_channels: array of ints or None
            channels ids that are triggered on, if None trigger will run on all channels
        trigger_name: string
            a unique name of this particular trigger
        """

        # set maximum signal amplitude
        max_signal = 0
        event_id = evt.get_id()
        for channel in station.iter_channels():
            trace = channel.get_trace()
            amplitude_envelope = np.abs(hilbert(trace))

            max_signal = max(max_signal, amplitude_envelope.max()) # which channels had the highest amplitude

        trigger = EnvelopeTrigger(trigger_name, threshold, triggered_channels)

        if max_signal < threshold:
            trigger.set_triggered(False)
            logger.debug("station has NOT triggered")
        else:
            trigger.set_triggered(True)
            logger.debug("station has triggered")

        station.set_trigger(trigger)

    def end(self):
        return
