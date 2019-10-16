from __future__ import absolute_import, division, print_function
import numpy as np
import logging
from NuRadioReco.utilities import fft
try:
    import cPickle as pickle
except ImportError:
    import pickle
logger = logging.getLogger("BaseTrace")


class BaseTrace:

    def __init__(self):
        self._sampling_rate = None
        self._time_trace = None
        self._frequency_spectrum = None
        self.__time_domain_up_to_date = True
        self._trace_start_time = 0

    def get_trace(self):
        """
        returns the time trace. If the frequency spectrum was modified before,
        an ifft is performed automatically to have the time domain representation
        up to date.

        Returns: 1 or N dimensional np.array of floats
            the time trace
        """
        if(not self.__time_domain_up_to_date):
#             logger.debug("time domain is not up to date, calculating FFT on the fly")
            self._time_trace = fft.freq2time(self._frequency_spectrum, self._sampling_rate)
            self.__time_domain_up_to_date = True
            self._frequency_spectrum = None
        return self._time_trace

    def get_frequency_spectrum(self):
        if(self.__time_domain_up_to_date):
#             logger.debug("frequency domain is not up to date, calculating FFT on the fly")
#             logger.debug("time trace has shape {}".format(self._time_trace.shape))
            self._frequency_spectrum = fft.time2freq(self._time_trace, self._sampling_rate)
            self._time_trace = None
#             logger.debug("frequency spectrum has shape {}".format(self._frequency_spectrum.shape))
            self.__time_domain_up_to_date = False
        return self._frequency_spectrum

    def set_trace(self, trace, sampling_rate):
        """
        sets the time trace

        Parameters
        -----------
        trace: np.array of floats
            the time series
        sampling_rate: float
            the sampling rage of the trace, i.e., the inverse of the bin width
        """
        if trace is not None:
            if trace.shape[trace.ndim - 1]%2 != 0:
                raise ValueError('Attempted to set trace with an uneven number ({}) of samples. Only traces with an even number of samples are allowed.'.format(trace.shape[trace.ndim - 1]))
        self.__time_domain_up_to_date = True
        self._time_trace = trace
        self._sampling_rate = sampling_rate
        self._frequency_spectrum = None

    def set_frequency_spectrum(self, frequency_spectrum, sampling_rate):
        self.__time_domain_up_to_date = False
        self._frequency_spectrum = frequency_spectrum
        self._sampling_rate = sampling_rate
        self._time_trace = None

    def get_sampling_rate(self):
        """
        returns the sampling rate of the trace

        Return: float
            sampling rate, i.e., the inverse of the bin width
        """
        return self._sampling_rate

    def get_times(self):
        try:
            length = self.get_number_of_samples()
            times = np.arange(0, length / self._sampling_rate - 0.1/self._sampling_rate, 1. / self._sampling_rate) + self._trace_start_time
            if(len(times) != length):
                logger.error("time array does not have the same length as the trace. n_samples = {:d}, sampling rate = {:.5g}".format(length, self._sampling_rate))
                raise ValueError("time array does not have the same length as the trace")
        except:
            times = np.array([])
        return times

    def set_trace_start_time(self, start_time):
        self._trace_start_time = start_time

    def add_trace_start_time(self, start_time):
        self._trace_start_time += start_time

    def get_trace_start_time(self):
        return self._trace_start_time

    def get_frequencies(self):
        length = self.get_number_of_samples()
        return np.fft.rfftfreq(length, d=(1. / self._sampling_rate))

    def get_hilbert_envelope(self):
        from scipy import signal
        h = signal.hilbert(self.get_trace())
        return np.array([np.abs(h[0]), np.abs(h[1]), np.abs(h[2])])

    def get_hilbert_envelope_mag(self):
        return np.linalg.norm(self.get_hilbert_envelope(), axis=0)

    def get_number_of_samples(self):
        """
        returns the number of samples in the time domain

        Return: int
            number of samples in time domain
        """
        length = 0
        if(self.__time_domain_up_to_date):
            length = self._time_trace.shape[-1]  # returns the correct length independent of the dimension of the array (channels are 1dim, efields are 3dim)
        else:
            length = (self._frequency_spectrum.shape[-1] - 1) * 2
        return length

    def serialize(self):
        data = {'sampling_rate': self.get_sampling_rate(),
                'time_trace': self.get_trace(),
                'trace_start_time': self.get_trace_start_time()}
        return pickle.dumps(data, protocol=4)

    def deserialize(self, data_pkl):
        data = pickle.loads(data_pkl)
        self.set_trace(data['time_trace'], data['sampling_rate'])
        if('trace_start_time' in data.keys()):
            self.set_trace_start_time(data['trace_start_time'])
