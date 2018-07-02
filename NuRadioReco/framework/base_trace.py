from __future__ import absolute_import, division, print_function
import numpy as np
import logging
import cPickle as pickle
logger = logging.getLogger("BaseTrace")


class BaseTrace:

    def __init__(self):
        self._sampling_rate = None
        self._time_trace = None
        self._frequency_spectrum = None
        self.__time_domain_up_to_date = True
        self._trace_start_time = 0

    def get_trace(self):
        if(not self.__time_domain_up_to_date):
#             logger.debug("time domain is not up to date, calculating FFT on the fly")
            self._time_trace = np.fft.irfft(self._frequency_spectrum, axis=-1, norm="ortho") / 2 ** 0.5
            self.__time_domain_up_to_date = True
        return self._time_trace

    def get_frequency_spectrum(self):
        if(self.__time_domain_up_to_date):
#             logger.debug("frequency domain is not up to date, calculating FFT on the fly")
#             logger.debug("time trace has shape {}".format(self._time_trace.shape))
            self._frequency_spectrum = np.fft.rfft(self._time_trace, axis=-1, norm="ortho") * 2 ** 0.5  # an additional sqrt(2) is added because negative frequencies are omitted.
#             logger.debug("frequency spectrum has shape {}".format(self._frequency_spectrum.shape))
            self.__time_domain_up_to_date = False
        return self._frequency_spectrum

    def set_trace(self, trace, sampling_rate):
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
        return self._sampling_rate

    def get_times(self):
        length = self.__get_trace_length()
        return np.arange(0, length / self._sampling_rate, 1. / self._sampling_rate) + self._trace_start_time

    def set_trace_start_time(self, start_time):
        self._trace_start_time = start_time

    def add_trace_start_time(self, start_time):
        self._trace_start_time += start_time

    def get_trace_start_time(self):
        return self._trace_start_time

    def get_frequencies(self):
        length = self.__get_trace_length()
        return np.fft.rfftfreq(length, d=(1. / self._sampling_rate))

    def get_hilbert_envelope(self):
        from scipy import signal
        h = signal.hilbert(self.get_trace())
        return np.array([np.abs(h[0]), np.abs(h[1]), np.abs(h[2])])

    def get_hilbert_envelope_mag(self):
        return np.linalg.norm(self.get_hilbert_envelope(), axis=0)

    def __get_trace_length(self):
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
        return pickle.dumps(data, protocol=2)

    def deserialize(self, data_pkl):
        data = pickle.loads(data_pkl)
        self.set_trace(data['time_trace'], data['sampling_rate'])
        if('trace_start_time' in data.keys()):
            self.set_trace_start_time(data['trace_start_time'])
