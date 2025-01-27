from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import channelParametersRNOG as chp



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

    def begin(self):
        pass
        
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

        def diff_sq(eventdata):
            """
            Returns sum of squared differences of samples across seams of 128-sample chunks.
            
            `eventdata`: channel waveform
            """
            runsum = 0.0
            deltaN = 64
            for chunk in range(len(eventdata) // 128 - 1):
                runsum += (eventdata[chunk * 128 + deltaN - 1] - eventdata[chunk * 128 + deltaN]) ** 2
            return np.sum(runsum)

        def unscramble(trace):
            """ script to fix scrambled traces (Note: first and last 64 samples are unusable and hence masked with zeros) """
            new_trace = np.zeros_like(trace)
            for i_section in range(len(trace) // 64):
                section_start = i_section * 64
                section_end = i_section * 64 + 64
                if i_section % 2 == 0:
                    new_trace[(section_start + 128) % 2048:(section_end + 128) % 2048] = trace[section_start:section_end]
                elif i_section > 1:
                    new_trace[(section_start - 128) % 2048:(section_end - 128) % 2048] = trace[section_start:section_end]
                    new_trace[0:64] = 0
            return new_trace    
        
        for ch in station.iter_channels():
            trace = ch.get_trace()
            trace_us = unscramble(trace)

            # glitching test statistic
            ts = diff_sq(trace) - diff_sq(trace_us)        
                
            ch.set_parameter(chp.glitch, ts)
