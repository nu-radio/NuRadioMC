from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import channelParametersRNOG as chp
from NuRadioReco.framework.event import Event

import numpy as np
import logging
import collections

class channelGlitchDetector:
    """
    This module detects scrambled data (digitizer "glitches") in the channels.

    RNO-G data (in particular the RNO-G data recorded with the LAB4D digitizers on the
    first-generation radiants) is known to have "glitches" in the channels.
    When a glitch is present in a channel, the 64-sample readout blocks were scrambled
    by the readout electronics which results in sudden unphyiscal jumps in the signal.
    These jumps are detected with this module (but not corrected).
    """

    def __init__(self, cut_value=0.0, glitch_fraction_warn_level=0.1, log_level=logging.NOTSET):
        """
        Parameters
        ----------
        cut_value : float
             This module calculates a test statistic that is sensitive to the presence
             of digitizer glitches. This parameter marks the critical value at which the test is
             said to have detected a glitch. This is a free parameter; increasing its value results
             in a lower false-positive rate (i.e. healthy events are incorrectly marked as glitching),
             but also a lower true-positive rate (i.e. glitching events are correctly marked as such).
             The default value of 0.0 is a good starting point.

        glitch_fraction_warn_level : float
            Print warning messages at the end of a run if a channel shows glitching in more than a fraction of
            `glitch_fraction_warn_level` of all events.

        log_level: enum
            Set verbosity level of logger. If logging.DEBUG, set mattak to verbose (unless specified in mattak_kwargs).
            (Default: logging.NOTSET, ie adhere to general log level)

        """

        self.logger = logging.getLogger('NuRadioReco.RNO_G.channelGlitchDetector')

        self.ts_cut_value = cut_value
        self.glitch_fraction_warn_level = glitch_fraction_warn_level

        # Total sampling buffer of the LAB4D: 2048 samples
        self._lab4d_readout_size = 2048

        # Length of a sampling block in the LAB4D: 64 samples
        self._lab4d_sampling_blocksize = 64

    def begin(self):
        # Per-run glitching statistics
        self.events_checked = 0
        self.events_glitching_per_channel = collections.defaultdict(int)

    def end(self):
        # Print glitch statistic summary
        for ch_id, events_glitching in self.events_glitching_per_channel.items():
            glitching_fraction = events_glitching / self.events_checked
            if glitching_fraction > self.glitch_fraction_warn_level:
                self.logger.warning(f"Channel {ch_id} shows glitches in {events_glitching} "
                                    f"/ {self.events_checked} = {100*glitching_fraction:.2f}% of events!")
            else:
                self.logger.info(f"Channel {ch_id} shows glitches in {events_glitching} "
                                 f"/ {self.events_checked} = {100*glitching_fraction:.2f}% of events!")

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

            block_size = self._lab4d_sampling_blocksize
            twice_block_size = 2 * block_size

            runsum = 0.0
            for chunk in range(len(eventdata) // twice_block_size - 1):
                runsum += (eventdata[chunk * twice_block_size + block_size - 1] - eventdata[chunk * twice_block_size + block_size]) ** 2
            return np.sum(runsum)

        def unscramble(trace):
            """
            Applies an unscrambling operation to the passed `trace`.
            Note: the first and last sampling block are unusable and hence replaced by zeros in the returned waveform.

            Parameters
            ----------
            `trace`: channel waveform
            """

            readout_size = self._lab4d_readout_size
            block_size = self._lab4d_sampling_blocksize
            twice_block_size = 2 * block_size

            new_trace = np.zeros_like(trace)

            for i_section in range(len(trace) // block_size):
                section_start = i_section * block_size
                section_end = i_section * block_size + block_size
                if i_section % 2 == 0:
                    new_trace[(section_start + twice_block_size) % readout_size :\
                              (section_end + twice_block_size) % readout_size] = trace[section_start:section_end]
                elif i_section > 1:
                    new_trace[(section_start - twice_block_size) % readout_size :\
                              (section_end - twice_block_size) % readout_size] = trace[section_start:section_end]
                    new_trace[0:block_size] = 0

            return new_trace

        # update event counter
        self.events_checked += 1

        for ch in station.iter_channels():
            ch_id = ch.get_id()

            trace = ch.get_trace()
            trace_us = unscramble(trace)

            # glitching test statistic and boolean discriminate
            glitch_ts = (diff_sq(trace) - diff_sq(trace_us)) / np.var(trace)
            glitch_disc = glitch_ts > self.ts_cut_value

            ch.set_parameter(chp.glitch, glitch_disc)
            ch.set_parameter(chp.glitch_test_statistic, glitch_ts)

            # update glitching statistics
            self.events_glitching_per_channel[ch_id] += glitch_disc


def has_glitch(event_or_station):
    """
    Returns true if any channel in any station has a "glitch".

    Requires the RNO_G.channelGlitchDetector module to have ran on the event.

    Parameters
    ----------
    event_or_station : Event or Station
        The event or station to check for glitches. If an event is given, the first station is used
        (if multiple stations exist in the event an error is raised).

    Returns
    -------
    has_glitch : bool
        True if any channel has a glitch, False otherwise.

    See Also
    --------
    NuRadioReco.modules.RNO_G.channelGlitchDetector
    """
    if isinstance(event_or_station, Event):
        # This will throw an error if the event has more than one station
        station = event_or_station.get_station()
    else:
        station = event_or_station

    for channel in station.iter_channels():
        if channel.has_parameter(chp.glitch) and channel.get_parameter(chp.glitch):
            return True

    return False