from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import channelParameters as chp



import numpy as np
import logging

logger = logging.getLogger('NuRadioReco.RNO_G.channelGlitchDetector')

class channelGlitchDetector:
    """
    This module detects "glitches" in the channels.

    RNO-G data (in particular the RNO-G data recorded with the first generation radiants)
    is known to have "glitches" in the channels. When a glitch is present in a channel,
    the 64-sample readout blocks were scrambled by the readout electronics which results
    in sudden unphyiscal jumps in the signal. These jumps are detected with this module.
    """

    def __init__(self):
        pass

    def begin(self, max_deviation=2000 * units.mV):
        """
        Parameters
        ----------
        max_deviation : float (default: 2000 * units.mV)
            The maximum deviation in the signal that is considered a glitch.
        """
        self.max_deviation = max_deviation

    def end(self):
        pass

    @register_run()
    def run(self, event, station, det=None):
        """ Run over channel traces and sets `channelParameter.glitch`.

        Parameters
        ----------
        event : `NuRadioReco.framework.event.Event`
            The event object
        station : `NuRadioReco.framework.station.Station`
            The station object
        det : `NuRadioReco.detector.detector.Detector` (default: None)
            Detector object, not used!
        """

        for ch in station.iter_channels():
            trace = ch.get_trace()
            diff = np.diff(trace)
            ch.set_parameter(chp.glitch, np.any(np.abs(diff) > self.max_deviation))
