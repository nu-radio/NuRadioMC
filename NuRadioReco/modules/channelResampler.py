from NuRadioReco.modules.base.module import register_run
import numpy as np
import fractions
from scipy import signal
from decimal import Decimal
import copy
from NuRadioReco.utilities import units, fft
import NuRadioReco.framework.sim_station
import logging



class channelResampler:
    """
    Resamples the trace to a new sampling rate.
    """

    def __init__(self):
        self.logger = logging.getLogger('NuRadioReco.channelResampler')
        self.begin()

    def begin(self, debug=False, log_level=logging.WARNING):
        self.__max_upsampling_factor = 5000
        self.__debug = debug
        self.logger.setLevel(log_level)

        """
        Begin the channelResampler

        Parameters
        ---------

        __debug: bool
            Debug switch

        """

    @register_run()
    def run(self, evt, station, det, sampling_rate):
        """
        Run the channelResampler

        Parameters
        ---------

        evt, station, det
            Event, Station, Detector
        sampling_rate: float
            In units 1/time provides the desired sampling rate of the data.

        """
        is_sim_station = isinstance(station, NuRadioReco.framework.sim_station.SimStation)
        if is_sim_station: # sim_stations are structured differently
            orig_binning = 1. / station.get_channel(0)[0].get_sampling_rate()  # assume that all channels have the same sampling rate
        else:
            orig_binning = 1. / station.get_channel(0).get_sampling_rate()
        target_binning = 1. / sampling_rate
        resampling_factor = fractions.Fraction(Decimal(orig_binning / target_binning)).limit_denominator(self.__max_upsampling_factor)
        if resampling_factor == self.__max_upsampling_factor:
            self.logger.warning("Safeguard caught, max upsampling {} factor reached.".format(self.__max_upsampling_factor))
        self.logger.debug("resampling channel trace by {}. Original binning is {:.3g} ns, target binning is {:.3g} ns".format(resampling_factor,
                                                                                                                         orig_binning / units.ns,
                                                                                                                         target_binning / units.ns))
        for channel in station.iter_channels():
            trace = channel.get_trace()
            if(self.__debug):
                ff = np.fft.rfftfreq(len(trace), orig_binning)
                spec = fft.time2freq(trace, channel.get_sampling_rate())
                spec[ff >= 500 * units.MHz] = 0
                trace = fft.freq2time(spec, channel.get_sampling_rate())
                trace_old = copy.copy(trace)
            if(resampling_factor.numerator != 1):
                trace = signal.resample(trace, resampling_factor.numerator * len(trace))  # , window='hann')
            if(resampling_factor.denominator != 1):
                trace = signal.resample(trace, len(trace) // resampling_factor.denominator)  # , window='hann')

            # make sure that trace has even number of samples
            if(len(trace) % 2 != 0):
                self.logger.info("channel trace has a odd number of samples after resampling. The last bin of the trace is discarded to maintain a even number of samples")
                trace = trace[:-1]

            if(self.__debug):

                if(resampling_factor.denominator != 1):
                    trace2 = signal.resample(trace, resampling_factor.denominator * len(trace))  # , window='hann')
                if(resampling_factor.numerator != 1):
                    trace2 = signal.resample(trace, len(trace) // resampling_factor.numerator)  # , window='hann')

                import matplotlib.pyplot as plt
                plt.plot(np.fft.rfftfreq(len(trace_old), orig_binning), np.abs(fft.time2freq(trace_old, 1/orig_binning)))
                plt.plot(np.fft.rfftfreq(len(trace2), orig_binning), np.abs(fft.time2freq(trace2, 1/orig_binning)))
                plt.show()

                tt = np.arange(0, len(trace_old) * orig_binning, orig_binning)
                plt.plot(tt, trace_old, '.-')
                tt = np.arange(0, len(trace2) * orig_binning, orig_binning)
                plt.plot(tt, trace2, '.-')
                tt = np.arange(0, len(trace) / sampling_rate, 1. / sampling_rate)
                plt.plot(tt, trace, 'o-')
                plt.show()

            channel.set_trace(trace, 1. / target_binning)

    def end(self):
        pass
