from NuRadioReco.modules.base.module import register_run
import numpy as np
import fractions
from scipy import signal
from decimal import Decimal
from NuRadioReco.utilities import units
import logging
logger = logging.getLogger('stationResampler')


class electricFieldResampler:
    """
    resamples the electric field trace to a new sampling rate
    """

    def __init__(self):
        self.begin()

    def begin(self, debug=False):
        self.__max_upsampling_factor = 5000
        self.__debug = debug
        pass

    @register_run()
    def run(self, event, station, det, sampling_rate):
        """
        resample electric field

        Parameters
        ----------
        event: event

        station: station

        det: detector

        sampling_rate: float
            desired new sampling rate

        """
        # access simulated efield and high level parameters
        # calculate sampling and FFT resolution
        for efield in station.get_electric_fields():
            orig_binning = 1. / efield.get_sampling_rate()

            target_binning = 1. / sampling_rate
            resampling_factor = fractions.Fraction(Decimal(orig_binning / target_binning)).limit_denominator(self.__max_upsampling_factor)
            logger.debug("resampling channel trace by a factor of {}. Original binning {:.3g}ns, target binning {:.3g}".format(resampling_factor,
                                                                                                                                orig_binning / units.ns,
                                                                                                                                target_binning / units.ns))
            new_length = int(efield.get_trace().shape[1] * resampling_factor)
            resampled_efield = np.zeros((3, new_length))  # create new data structure with new efield length

            for iE in range(3):
                trace = efield.get_trace()[iE]

                if(resampling_factor.numerator != 1):
                    trace = signal.resample(trace, resampling_factor.numerator * len(trace))  # , window='hann')
                if(resampling_factor.denominator != 1):
                    trace = signal.resample(trace, len(trace) // resampling_factor.denominator)  # , window='hann')

                resampled_efield[iE] = trace
            # prevent traces to get an odd number of samples. If the trae has an odd number of samples, the last sample is discarded.
            if resampled_efield.shape[-1] % 2 != 0:
                resampled_efield = resampled_efield.T[:-1].T
            efield.set_trace(resampled_efield, sampling_rate)

    def end(self):
        pass
