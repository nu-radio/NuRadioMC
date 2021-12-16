from __future__ import absolute_import, division, print_function
import numpy as np
import logging
import fractions
import decimal
import numbers
from NuRadioReco.utilities import fft, bandpass_filter
import scipy.signal
import copy
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
            self._time_trace = fft.freq2time(self._frequency_spectrum, self._sampling_rate)
            self.__time_domain_up_to_date = True
            self._frequency_spectrum = None
        return np.copy(self._time_trace)

    def get_filtered_trace(self, passband, filter_type='butter', order=10):
        """
        Returns the trace after applying a filter to it. This does not change the stored trace.

        Parameters
        ----------
        passband: list of floats
            lower and upper bound of the filter passband
        filter_type: string
            type of the applied filter. Options are rectangular, butter and butterabs
        order: int
            Order of the Butterworth filter, if the filter types butter or butterabs are chosen
        """
        spec = copy.copy(self.get_frequency_spectrum())
        freq = self.get_frequencies()
        filter_response = bandpass_filter.get_filter_response(freq, passband, filter_type, order)
        spec *= filter_response
        return fft.freq2time(spec, self.get_sampling_rate())

    def get_frequency_spectrum(self):
        if(self.__time_domain_up_to_date):
            self._frequency_spectrum = fft.time2freq(self._time_trace, self._sampling_rate)
            self._time_trace = None
#             logger.debug("frequency spectrum has shape {}".format(self._frequency_spectrum.shape))
            self.__time_domain_up_to_date = False
        return np.copy(self._frequency_spectrum)

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
            if trace.shape[trace.ndim - 1] % 2 != 0:
                raise ValueError('Attempted to set trace with an uneven number ({}) of samples. Only traces with an even number of samples are allowed.'.format(trace.shape[trace.ndim - 1]))
        self.__time_domain_up_to_date = True
        self._time_trace = np.copy(trace)
        self._sampling_rate = sampling_rate
        self._frequency_spectrum = None

    def set_frequency_spectrum(self, frequency_spectrum, sampling_rate):
        self.__time_domain_up_to_date = False
        self._frequency_spectrum = np.copy(frequency_spectrum)
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
            times = np.arange(0, length / self._sampling_rate - 0.1 / self._sampling_rate, 1. / self._sampling_rate) + self._trace_start_time
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
        # get hilbert envelope for either 1D (N) analytic trace or (3,N) E-field
        h = signal.hilbert(self.get_trace())
        return np.abs(h)

    def get_hilbert_envelope_mag(self):
        # ensure taking axis 0 of a 2D trace (trace might be (N) for analytic trace or (3,N) for E-field
        return np.linalg.norm(np.atleast_2d(self.get_hilbert_envelope()), axis=0)

    def get_number_of_samples(self):
        """
        returns the number of samples in the time domain

        Return: int
            number of samples in time domain
        """
        if(self.__time_domain_up_to_date):
            length = self._time_trace.shape[-1]  # returns the correct length independent of the dimension of the array (channels are 1dim, efields are 3dim)
        else:
            length = (self._frequency_spectrum.shape[-1] - 1) * 2
        return length

    def apply_time_shift(self, delta_t, silent=False):
        """
        Uses the fourier shift theorem to apply a time shift to the trace
        Note that this is a cyclic shift, which means the trace will wrap
        around, which might lead to problems, especially for large time shifts.

        Parameters
        ----------
        delta_t: float
            Time by which the trace should be shifted
        silent: boolean (default:False)
            Turn off warnings if time shift is larger than 10% of trace length
            Only use this option if you are sure that your trace is long enough
            to acommodate the time shift
        """
        if delta_t > .1 * self.get_number_of_samples() / self.get_sampling_rate() and not silent:
            logger.warning('Trace is shifted by more than 10% of its length')
        spec = self.get_frequency_spectrum()
        spec *= np.exp(-2.j * np.pi * delta_t * self.get_frequencies())
        self.set_frequency_spectrum(spec, self._sampling_rate)

    def resample(self, sampling_rate):
        if sampling_rate == self.get_sampling_rate():
            return
        resampling_factor = fractions.Fraction(decimal.Decimal(sampling_rate / self.get_sampling_rate())).limit_denominator(5000)

        resampled_trace = self.get_trace()
        if resampling_factor.numerator != 1:
            # resample and use axis -1 since trace might be either shape (N) for analytic trace or shape (3,N) for E-field
            resampled_trace = scipy.signal.resample(resampled_trace, resampling_factor.numerator * self.get_number_of_samples(), axis=-1)
        if resampling_factor.denominator != 1:
            # resample and use axis -1 since trace might be either shape (N) for analytic trace or shape (3,N) for E-field
            resampled_trace = scipy.signal.resample(resampled_trace, np.shape(resampled_trace)[-1] // resampling_factor.denominator, axis=-1)

        if resampled_trace.shape[-1] % 2 != 0:
            resampled_trace = resampled_trace.T[:-1].T
        self.set_trace(resampled_trace, sampling_rate)

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

    def __add__(self, x):
        """
        Redefine the "+" operator for BaseTrace objects. The operation will return a
        new BaseTrace object containing the sum of the two traces. If the two traces
        have different sampling rates, one of them is upsampled to the higher sampling
        rate.
        """
        # Some sanity checks
        if not isinstance(x, BaseTrace):
            raise TypeError('+ operator is only defined for 2 BaseTrace objects')
        if self.get_trace() is None or x.get_trace() is None:
            raise ValueError('One of the trace objects has no trace set')
        if self.get_trace().ndim != x.get_trace().ndim:
            raise ValueError('Traces have different dimensions')
        if self.get_sampling_rate() != x.get_sampling_rate():
            # Upsample trace with lower sampling rate
            # Create new baseTrace object for the resampling so we don't change the originals
            if self.get_sampling_rate() > x.get_sampling_rate():
                upsampled_trace = BaseTrace()
                upsampled_trace.set_trace(x.get_trace(), x.get_sampling_rate())
                upsampled_trace.resample(self.get_sampling_rate())
                trace_1 = copy.copy(self.get_trace())
                trace_2 = upsampled_trace.get_trace()
                sampling_rate = self.get_sampling_rate()
            else:
                upsampled_trace = BaseTrace()
                upsampled_trace.set_trace(self.get_trace(), self.get_sampling_rate())
                upsampled_trace.resample(x.get_sampling_rate())
                trace_1 = upsampled_trace.get_trace()
                trace_2 = copy.copy(x.get_trace())
                sampling_rate = x.get_sampling_rate()
        else:
            trace_1 = copy.copy(self.get_trace())
            trace_2 = copy.copy(x.get_trace())
            sampling_rate = self.get_sampling_rate()

        # Figure out which of the traces has the earlier trace start time
        if self.get_trace_start_time() <= x.get_trace_start_time():
            first_trace = trace_1
            second_trace = trace_2
            trace_start = self.get_trace_start_time()
        else:
            first_trace = trace_2
            second_trace = trace_1
            trace_start = x.get_trace_start_time()
        # Calculate the difference in the trace start time between the traces and the number of
        # samples that time difference corresponds to
        time_offset = np.abs(x.get_trace_start_time() - self.get_trace_start_time())
        i_start = int(round(time_offset * sampling_rate))
        # We have to distinguish 2 cases: Trace is 1D (channel) or 2D(E-field)
        # and treat them differently
        if trace_1.ndim == 1:
            # Calculate length the new trace needs to hold both input traces
            trace_length = max(first_trace.shape[0], i_start + second_trace.shape[0])
            # Make sure trace has an even number of samples
            trace_length += trace_length % 2
            # Put both pulses at the start of their own traces for now. We correct for different start times later
            early_trace = np.zeros(trace_length)
            early_trace[:first_trace.shape[0]] = first_trace
            late_trace = np.zeros(trace_length)
            late_trace[:second_trace.shape[0]] = second_trace
        else:
            # Same as in the if bracket, but for a 2D trace (like an E-field)
            trace_length = max(first_trace.shape[1], i_start + second_trace.shape[1])
            trace_length += trace_length % 2
            early_trace = np.zeros((first_trace.shape[0], trace_length))
            early_trace[:, :first_trace.shape[1]] = first_trace
            late_trace = np.zeros((second_trace.shape[0], trace_length))
            late_trace[:, :second_trace.shape[1]] = second_trace
        # Correct for different trace start times by using fourier shift theorem to
        # shift the later trace backwards.
        late_trace_object = BaseTrace()
        late_trace_object.set_trace(late_trace, sampling_rate)
        late_trace_object.apply_time_shift(time_offset, True)
        # Create new BaseTrace object holding the summed traces
        new_trace = BaseTrace()
        new_trace.set_trace(early_trace + late_trace_object.get_trace(), sampling_rate)
        new_trace.set_trace_start_time(trace_start)
        return new_trace

    def __mul__(self, x):
        if isinstance(x, numbers.Number):
            if self._time_trace is not None:
                self._time_trace *= x
                return self
            if self._frequency_spectrum is not None:
                self._frequency_spectrum *= x
                return self
            raise ValueError('Cant multiply baseTrace with number because no value is set for trace.')
        else:
            raise TypeError('Multiplication of baseTrace object with object of type {} is not defined'.format(type(x)))

    def __rmul__(self, x):
        return self.__mul__(x)

    def __truediv__(self, x):
        if isinstance(x, numbers.Number):
            if self._time_trace is not None:
                self._time_trace = self._time_trace / x
                return self
            if self._frequency_spectrum is not None:
                self._frequency_spectrum = self._frequency_spectrum / x
                return self
            raise ValueError('Cant divide baseTrace by number because no value is set for trace.')
        else:
            raise TypeError('Division of baseTrace object with object of type {} is not defined'.format(type(x)))
