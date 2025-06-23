from __future__ import absolute_import, division, print_function

from NuRadioReco.utilities import fft, signal_processing
import NuRadioReco.detector.response
from NuRadioReco.utilities import units, signal_processing

import numpy as np
import logging
import numbers
import copy
import pickle
from NuRadioReco.utilities.io_utilities import _dumps
logger = logging.getLogger("NuRadioReco.BaseTrace")


class BaseTrace:

    def __init__(self, trace=None, sampling_rate=None, trace_start_time=0):
        """
        Initialize the BaseTrace object.

        Parameters
        ----------
        trace : np.array of floats (default: None)
            The time trace. Can also be set later with the `set_trace` method.
        sampling_rate : float (default: None)
            The sampling rate of the trace, i.e., the inverse of the bin width.
        trace_start_time : float (default: 0)
            The start time of the trace.
        """
        self._sampling_rate = None
        self._time_trace = None
        self._frequency_spectrum = None
        self.__time_domain_up_to_date = True
        self._trace_start_time = trace_start_time
        if trace is not None:
            self.set_trace(trace, sampling_rate)

    def get_trace(self):
        """
        Returns the time trace.

        If the frequency spectrum was modified before,
        an ifft is performed automatically to have the time domain representation
        up to date.

        Returns
        -------
        trace: np.array of floats
            the time trace
        """
        if not self.__time_domain_up_to_date:
            self._time_trace = fft.freq2time(self._frequency_spectrum, self._sampling_rate)
            self.__time_domain_up_to_date = True
            self._frequency_spectrum = None
        return np.copy(self._time_trace)

    def get_filtered_trace(self, passband, filter_type='butter', order=10, rp=None):
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
        filter_response = signal_processing.get_filter_response(freq, passband, filter_type, order, rp)
        spec *= filter_response
        return fft.freq2time(spec, self.get_sampling_rate())

    def get_frequency_spectrum(self, window_mask=None):
        """
        Returns the frequency spectrum.

        Parameters
        ----------
        window_mask: array of bools (default: None)
            If not None, specifies the time window to be used for the FFT. Has to have the same length as the trace.

        Returns
        -------
        frequency_spectrum: np.array of floats
            The frequency spectrum.
        """
        if window_mask is None:
            if self.__time_domain_up_to_date:
                self._frequency_spectrum = fft.time2freq(self._time_trace, self._sampling_rate)
                self._time_trace = None
                self.__time_domain_up_to_date = False

            return np.copy(self._frequency_spectrum)
        else:
            trace = copy.copy(self.get_trace())
            # The double transpose allows to work with 1D and ND traces
            return fft.time2freq(trace.T[window_mask].T, self._sampling_rate)

    def set_trace(self, trace, sampling_rate, trace_start_time=None):
        """
        Sets the time trace.

        Parameters
        ----------
        trace : np.array of floats
            The time series
        sampling_rate : float or str
            The sampling rate of the trace, i.e., the inverse of the bin width.
            If `sampling_rate="same"`, sampling rate is not changed (requires previous initialisation).
        trace_start_time : float (default: None)
            Set the start time of the trace. If None, the start time is not changed/set.
        """
        if trace is not None:
            if trace.shape[trace.ndim - 1] % 2 != 0:
                raise ValueError(
                    f'Attempted to set trace with an uneven number ({trace.shape[trace.ndim - 1]}) '
                    'of samples. Only traces with an even number of samples are allowed.')
        self.__time_domain_up_to_date = True
        self._time_trace = np.copy(trace)

        self._frequency_spectrum = None

        if isinstance(sampling_rate, str) and sampling_rate.lower() == "same":
            if self._sampling_rate is None:
                raise ValueError(
                    "You specified to keep the sampling rate but no value have been set previously.")
                pass  # keep value of self._sampling_rate
        elif sampling_rate is not None:
            self._sampling_rate = sampling_rate
        else:
            raise ValueError("You have to specify a sampling rate for `BaseTrace.set_trace(...)`")

        if trace_start_time is not None:
            self.set_trace_start_time(trace_start_time)

    def set_frequency_spectrum(self, frequency_spectrum, sampling_rate):
        """
        Sets the frequency spectrum.

        Parameters
        ----------
        frequency_spectrum : np.array of floats
            The frequency spectrum
        sampling_rate : float or str
            The sampling rate of the trace, i.e., the inverse of the bin width.
            If `sampling_rate="same"`, sampling rate is not changed (requires previous initialisation).
        """
        self.__time_domain_up_to_date = False
        self._frequency_spectrum = np.copy(frequency_spectrum)
        self._time_trace = None

        if isinstance(sampling_rate, str) and sampling_rate.lower() == "same":
            if self._sampling_rate is None:
                raise ValueError(
                    "You specified to keep the sampling rate but no value have been set previously.")
            pass  # keep value of self._sampling_rate
        elif sampling_rate is not None:
            self._sampling_rate = sampling_rate
        else:
            raise ValueError("You have to specify a sampling rate for `BaseTrace.set_frequency_spectrum(...)`")

    def get_sampling_rate(self):
        """
        Returns the sampling rate of the trace.

        Returns
        -------
        sampling_rate: float
            sampling rate, i.e., the inverse of the bin width
        """
        return self._sampling_rate

    def get_times(self):
        try:
            length = self.get_number_of_samples()
            times = np.arange(0, length / self._sampling_rate - 0.1 / self._sampling_rate,
                1. / self._sampling_rate) + self._trace_start_time
            if len(times) != length:
                err = ("time array does not have the same length as the trace. "
                    f"n_samples = {length:d}, sampling rate = {self._sampling_rate:.5g}")
                logger.error(err)
                raise ValueError(err)
        except (ValueError, AttributeError):
            times = np.array([])
        return times

    def set_trace_start_time(self, start_time):
        self._trace_start_time = start_time

    def add_trace_start_time(self, start_time):
        self._trace_start_time += start_time

    def get_trace_start_time(self):
        return self._trace_start_time

    def get_frequencies(self, window_mask=None):
        """
        Returns the frequencies of the frequency spectrum.

        Parameters
        ----------
        window_mask: array of bools (default: None)
            If not None, used to determine the number of samples in the time domain used for the frequency spectrum.

        Returns
        -------
        frequencies: np.array of floats
            The frequencies of the frequency spectrum.
        """
        if window_mask is None:
            nsamples = self.get_number_of_samples()
        else:
            nsamples = int(np.sum(window_mask))

        return fft.freqs(nsamples, self._sampling_rate)

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
        Returns the number of samples in the time domain.

        Returns
        -------
        n_samples: int
            number of samples in time domain
        """
        if self.__time_domain_up_to_date:
            length = self._time_trace.shape[-1]  # returns the correct length independent of the dimension of the array (channels are 1dim, efields are 3dim)
        else:
            length = (self._frequency_spectrum.shape[-1] - 1) * 2
        return length

    def apply_time_shift(self, delta_t, silent=False, fourier_shift_threshold = 1e-5 * units.ns):
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
        fourier_shift_threshold: float (default: 1e-5 * units.ns)
            Threshold for the Fourier shift. If the shift is closer to a multiple of
            1 / sampling_rate than this, the trace is rolled instead of using the Fourier
            shift theorem to save time and avoid numerical errors in the Fourier transforms.
        """
        if abs(delta_t) > .1 * self.get_number_of_samples() / self.get_sampling_rate() and not silent:
            logger.warning('Trace is shifted by more than 10% of its length')

        if abs(round(delta_t * self.get_sampling_rate()) - delta_t * self.get_sampling_rate()) < fourier_shift_threshold:
            roll_by = int(round(delta_t * self.get_sampling_rate()))
            trace = self.get_trace()
            trace = np.roll(trace, roll_by, axis=-1)
            self.set_trace(trace, self.get_sampling_rate())
        else:
            spec = self.get_frequency_spectrum()
            spec *= np.exp(-2.j * np.pi * delta_t * self.get_frequencies())
            self.set_frequency_spectrum(spec, self.get_sampling_rate())

    def resample(self, sampling_rate):
        """ Resamples the trace to a new sampling rate.

        Parameters
        ----------
        sampling_rate: float
            The new sampling rate.
        """
        if sampling_rate == self.get_sampling_rate():
            return

        resampled_trace = signal_processing.resample(self.get_trace(), sampling_rate / self.get_sampling_rate())
        self.set_trace(resampled_trace, sampling_rate)

    def serialize(self):
        time_trace = self.get_trace()
        # if there is no trace, the above will return np.array(None).
        if not time_trace.shape:
            return None
        data = {'sampling_rate': self.get_sampling_rate(),
                'time_trace': time_trace,
                'trace_start_time': self.get_trace_start_time()}
        return _dumps(data, protocol=4)

    def deserialize(self, data_pkl):
        data = pickle.loads(data_pkl)
        self.set_trace(data['time_trace'], data['sampling_rate'])
        if 'trace_start_time' in data.keys():
            self.set_trace_start_time(data['trace_start_time'])

    def add_to_trace(self, channel, min_residual_time_offset=1e-5 * units.ns, raise_error=True):
        """
        Adds the trace of another channel to the trace of this channel.

        The trace of the incoming channel is only added within the time window of the current
        channel. If the current channel has an empty trace (i.e., a trace containing zeros) with
        a defined trace_start_time, this function can be seen as recording the incoming channel
        in the specified readout window. Hence, the current channel is referred to as the "readout"
        in the comments of this function.

        Parameters
        ----------
        channel: BaseTrace
            The channel whose trace is to be added to the trace of the current channel.
        min_residual_time_offset: float (default: 1e-5 * units.ns)
            Minimum residual time between the target bin of this channel and the target bin of the channel
            to be added. Below this threshold the residual time shift is not applied to increase performance
            and minimize numerical artifacts from Fourier transforms.
        raise_error: bool (default: True)
            If True, an error is raised if (part of) the current channel
            (readout window) is outside the incoming channel.
        """
        assert self.get_number_of_samples() is not None, "No trace is set for this channel"
        assert self.get_sampling_rate() == channel.get_sampling_rate(), "Sampling rates of the two channels do not match"

        tt_readout = self.get_times()
        t0_readout = self.get_trace_start_time()
        t1_readout = tt_readout[-1]
        sampling_rate_readout = self.get_sampling_rate()
        n_samples_readout = self.get_number_of_samples()

        tt_channel = channel.get_times()
        t0_channel = channel.get_trace_start_time()
        t1_channel = tt_channel[-1]
        sampling_rate_channel = channel.get_sampling_rate()
        n_samples_channel = channel.get_number_of_samples()

        # We handle 1+2x2 cases:
        # 1. Channel is completely outside readout window:
        if t1_channel < t0_readout or t1_readout < t0_channel:
            if raise_error:
                logger.error("The channel is completely outside the readout window")
                raise ValueError('The channel is completely outside the readout window')
            return

        def floor(x):
            return int(np.floor(round(x, int(np.log10(1/(0.01*units.ps))))))

        def ceil(x):
            return int(np.ceil(round(x, int(np.log10(1/(0.01*units.ps))))))

        # 2. Channel starts before readout window:
        if t0_channel < t0_readout:
            i_start_readout = 0
            t_start_readout = t0_readout
            i_start_channel = ceil((t0_readout - t0_channel) * sampling_rate_channel) # The first bin of channel inside readout
            t_start_channel = tt_channel[i_start_channel]
        # 3. Channel starts after readout window:
        elif t0_channel >= t0_readout:
            if raise_error:
                logger.error("The readout window starts before the incoming channel")
                raise ValueError('The readout window starts before the incoming channel')

            i_start_readout = floor((t0_channel - t0_readout) * sampling_rate_readout) # The bin of readout right before channel starts
            t_start_readout = tt_readout[i_start_readout]
            i_start_channel = 0
            t_start_channel = t0_channel

        # 4. Channel ends after readout window:
        if t1_channel >= t1_readout:
            i_end_readout = n_samples_readout
            i_end_channel = ceil((t1_readout - t0_channel) * sampling_rate_channel) + 1 # The bin of channel right after readout ends
        # 5. Channel ends before readout window:
        elif t1_channel < t1_readout:
            if t1_channel > t1_readout and raise_error:
                logger.error("The readout window ends after the incoming channel")
                raise ValueError('The readout window ends after the incoming channel')

            i_end_readout = floor((t1_channel - t0_readout) * sampling_rate_readout) + 1 # The bin of readout right before channel ends
            i_end_channel = n_samples_channel
        # Determine the remaining time between the binning of the two traces and use time shift as interpolation:
        residual_time_offset = t_start_channel - t_start_readout
        if np.abs(residual_time_offset) >= min_residual_time_offset:
            tmp_channel = copy.deepcopy(channel)
            tmp_channel.apply_time_shift(residual_time_offset)

            trace_to_add = tmp_channel.get_trace()
        else:
            trace_to_add = channel.get_trace()

        if i_end_readout - i_start_readout != i_end_channel - i_start_channel:
            logger.error("The traces do not have the same length. This should not happen.")
            raise ValueError('The traces do not have the same length. This should not happen.')

        # Add the trace to the original trace:
        original_trace = self.get_trace()
        # arr[..., start:stop] works for any dimension: 1, 2, 3, ...
        original_trace[..., i_start_readout:i_end_readout] += trace_to_add[..., i_start_channel:i_end_channel]

        self.set_trace(original_trace, sampling_rate_readout)


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
        elif isinstance(x, NuRadioReco.detector.response.Response):
            return x * self  # operation defined in detector.response.Response
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
