import numpy as np
import fractions
from scipy import signal
from decimal import Decimal
import copy
from NuRadioReco.utilities import units, fft
import NuRadioReco.framework.sim_station
import logging
logger = logging.getLogger('channelResampler')
logging.basicConfig()


class channelResampler:
    """
    Resamples the trace to a new sampling rate.
    """

    def __init__(self):
        self.begin()

    def begin(self, debug=False):
        self.__max_upsampling_factor = 5000
        self.__debug = debug

        """
        Begin the channelResampler

        Parameters
        ---------

        __debug: bool
            Debug switch

        """


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
            logger.warning("Safeguard caught, max upsampling {} factor reached.".format(self.__max_upsampling_factor))
        logger.debug("resampling channel trace by {}. Original binning is {:.3g} ns, target binning is {:.3g} ns".format(resampling_factor,
                                                                                                                         orig_binning / units.ns,
                                                                                                                         target_binning / units.ns))
        for channel in station.iter_channels():
            if is_sim_station:
                for sub_channel in channel: # a sim_station channel can have multiple sub-channels, corresponding to direct, reflected and refracted signal
                    trace = sub_channel.get_trace()
                    new_trace = [[],[],[]]
                    for i_pol, polarization in enumerate(trace): # in the channels of a sim_station, the channels store e-field traces, which are 3-dimensional
                        if(resampling_factor.numerator != 1):
                            polarization = signal.resample(polarization, resampling_factor.numerator * len(polarization))  # , window='hann')
                        if(resampling_factor.denominator != 1):
                            polarization = signal.resample(polarization, len(polarization) / resampling_factor.denominator)  # , window='hann')
                        if(len(polarization) % 2 != 0):
                            logger.info("channel trace has a odd number of samples after resampling. The last bin of the trace is discarded to maintain a even number of samples")
                            polarization = polarization[:-1]
                        new_trace[i_pol] = polarization
                    sub_channel.set_trace(np.array(new_trace), 1./target_binning)
                    
            else:
                trace = channel.get_trace()
                if(self.__debug):
                    ff = np.fft.rfftfreq(len(trace), orig_binning)
                    spec = fft.time2freq(trace)
                    spec[ff >= 500 * units.MHz] = 0
                    trace = fft.freq2time(spec)
                    trace_old = copy.copy(trace)
                if(resampling_factor.numerator != 1):
                    trace = signal.resample(trace, resampling_factor.numerator * len(trace))  # , window='hann')
                if(resampling_factor.denominator != 1):
                    trace = signal.resample(trace, len(trace) / resampling_factor.denominator)  # , window='hann')
                    
                # make sure that trace has even number of samples
                if(len(trace) % 2 != 0):
                    logger.info("channel trace has a odd number of samples after resampling. The last bin of the trace is discarded to maintain a even number of samples")
                    trace = trace[:-1]

                if(self.__debug):

                    if(resampling_factor.denominator != 1):
                        trace2 = signal.resample(trace, resampling_factor.denominator * len(trace))  # , window='hann')
                    if(resampling_factor.numerator != 1):
                        trace2 = signal.resample(trace, len(trace) / resampling_factor.numerator)  # , window='hann')

                    import matplotlib.pyplot as plt
                    plt.plot(np.fft.rfftfreq(len(trace_old), orig_binning), np.abs(fft.time2freq(trace_old)))
                    plt.plot(np.fft.rfftfreq(len(trace2), orig_binning), np.abs(fft.time2freq(trace2)))
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
