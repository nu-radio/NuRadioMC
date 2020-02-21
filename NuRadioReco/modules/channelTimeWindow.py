import numpy as np
import logging
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.modules.base.module import register_run


class channelTimeWindow:
    """
    Set trace outside time window to zero.
    """
    def __init__(self):
        self.logger = logging.getLogger("NuRadioReco.channelTimeWindow")
        self.begin()

    def begin(self, debug=False):
        self.__debug = debug
        if(debug):
            logger.setLevel(logging.DEBUG)

    @register_run()
    def run(self, evt, station, det, window=None,
            window_function='rectangular',
            around_pulse=True,
            window_width=50 * units.ns,
            window_rise_time=20 * units.ns
            ):
        """
        Parameters
        -----------
        evt, station, det
            Event, Station, Detector
        window: list or None
            [time_window_start, time_window_end] in which the signal should be kept (in units of time)
        window_function: string
            select window function
            * rectangular
            * hanning
        around_pulse: float or None
            if not None: specifies time interval around 'signal_time',
            if this option is set, 'window' is ignored
        """

        for channel in station.iter_channels():
            times = channel.get_times()
            trace = channel.get_trace()

            if(around_pulse is not None):
                if(not channel.has_parameter(chp.signal_time)):
                    raise AttributeError("channel parameter 'signal_time' is not set. Did you forget to run the channelSignalReconstructor?")
                tmax = channel[chp.signal_time]
                window = [tmax - window_width * 0.5, tmax + window_width * 0.5]
            if(window_function == 'rectangular'):
                trace[np.where(times < window[0])] = 0.
                trace[np.where(times > window[1])] = 0.
            else:
                window_fkt = np.zeros_like(trace)
                i00 = np.argmin(np.abs(times - (window[0] - window_rise_time)))
                i01 = np.argmin(np.abs(times - window[0]))
                i10 = np.argmin(np.abs(times - window[1]))
                i11 = np.argmin(np.abs(times - (window[1] + window_rise_time)))
#                 print(window)
#                 logger.debug("times = {:.2f} {:.2f} {:.2f} {:.2f}".format(times - (window[0] - window_rise_time),
#                                                           times - window[0],
#                                                           times - window[1],
#                                                           times - (window[1] + window_rise_time)))
                logger.debug("indices = {} {} {} {}".format(i00, i01, i10, i11))
                window_fkt[i01:i10] = 1
                if(window_function == 'hanning'):
                    n = i01 - i00
                    window_fkt[i00:i01] = np.hanning(n * 2)[:n]
                    n = i11 - i10
                    window_fkt[i10:i11] = np.hanning(n * 2)[n:]
                else:
                    raise NotImplementedError("window function {} is not implemented".format(window_function))
                if(self.__debug):
                    from matplotlib import pyplot as plt
                    fig, ax = plt.subplots(1, 1)
                    ax.plot(times, window_fkt)
                    plt.show()
                trace *= window_fkt
            channel.set_trace(trace, channel.get_sampling_rate())

    def end(self):
        pass
