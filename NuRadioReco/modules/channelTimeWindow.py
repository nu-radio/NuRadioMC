import numpy as np
from NuRadioReco.utilities import units


class channelTimeWindow:
    """
    Set trace outside time window to zero.
    """
    def begin(self):
        pass

    def run(self, evt, station, det, window=[0 * units.ns, 100 * units.ns], debug=False):

        channels = station.get_channels()
        for channel in channels:
            times = channel.get_times()
            trace = channel.get_trace()
            trace[np.where(times < window[0])] = 0.
            trace[np.where(times > window[1])] = 0.
            channel.set_trace(trace, channel.get_sampling_rate())

    def end(self):
        pass
