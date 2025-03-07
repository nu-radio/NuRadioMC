from NuRadioReco.modules.base.module import register_run
import numpy as np
import logging
from NuRadioReco.utilities import units
import scipy.signal.windows
logger = logging.getLogger('NuRadioReco.channelStopFilter')


class channelStopFilter:
    """
    Apply tapering towards zero to channel traces

    This class applies a Tukey window to the ends of the traces
    to gradually taper them towards zero, and appends zeros at either end.
    This prevents unphysical features that can result from applying transformations
    to the data in the frequency domain (see :doc:`additional explanation here </NuRadioReco/pages/times>`)

    For ARIANNA data, this module should always be used to remove the glitch around
    the beginning and end of the trace (around the 'stop').
    """

    def begin(self):
        pass

    @register_run()
    def run(self, evt, station, det, filter_size=0.1, prepend=128 * units.ns, append=128 * units.ns):
        """
        Parameters
        ----------
        evt: Event
            The event to run the module on
        station: Station
            The station to run the module on
        det: Detector
            The detector description
        filter_size: size of tukey window (float)
            specifies the percentage of the trace where the filter is active.
            default is 0.1, i.e. the first 5% and the last 5% of the trace are filtered
        prepend: time interval to prepend
            the time span that is filled with zeros and prepended to the trace
        append: time interval to append
            the time span that is filled with zeros and appended to the trace
        """
        for channel in station.iter_channels():
            trace = channel.get_trace()
            sampling_rate = channel.get_sampling_rate()
            window = scipy.signal.windows.tukey(len(trace), filter_size)
            trace *= window
            prepend_samples = int(np.round(prepend * sampling_rate))
            trace = np.append(np.zeros(prepend_samples), trace)
            trace = np.append(trace, np.zeros(int(np.round(append * sampling_rate))))
            channel.set_trace(trace, sampling_rate)
            channel.add_trace_start_time(-prepend_samples / sampling_rate)

    def end(self):
        pass
