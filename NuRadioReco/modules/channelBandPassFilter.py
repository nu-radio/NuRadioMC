import numpy as np
from scipy import signal
import logging
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units, bandpass_filter, signal_processing


class channelBandPassFilter:
    """
    Band pass filters the channels using different band-pass filters.
    """
    __filter_cached = None
    __filter_args = None

    def __init__(self, caching=True):
        """
        Parameters
        ----------
        caching: bool, default True
            If True (default), internally caches the filter. This speeds up
            the (common) case where the same filter is applied to multiple
            channels.

        """
        self.__t = 0
        self.__caching = caching
        self.begin()
        self.logger = logging.getLogger('NuRadioReco.channelBandPassFilter')

    def begin(self):
        pass

    @staticmethod
    def get_filter_arguments(channel_id, passband, filter_type, order=2,
                             rp=None, roll_width=None, half_hann_percent=None):
        if isinstance(passband, dict):
            tmp_passband = passband[channel_id]
        else:
            tmp_passband = passband

        if isinstance(order, dict):
            tmp_order = order[channel_id]
        else:
            tmp_order = order

        if isinstance(filter_type, dict):
            tmp_filter_type = filter_type[channel_id]
        else:
            tmp_filter_type = filter_type

        if isinstance(rp, dict):
            tmp_rp = rp[channel_id]
        else:
            tmp_rp = rp

        if isinstance(roll_width, dict):
            tmp_roll_width = roll_width[channel_id]
        else:
            tmp_roll_width = roll_width

        if isinstance(half_hann_percent, dict):
            tmp_half_hann_percent = half_hann_percent[channel_id]
        else:
            tmp_half_hann_percent = half_hann_percent
        return tmp_passband, tmp_order, tmp_filter_type, tmp_rp, tmp_roll_width, tmp_half_hann_percent

    @register_run()
    def run(self, evt, station, det, passband=None,
            filter_type='rectangular', order=2, rp=None,
            roll_width=2.5 * units.MHz, half_hann_percent=0.1):
        """
        Run the filter

        Parameters
        ----------

        evt, station, det
            Event, Station, Detector
        passband: list or dict of lists, (default: [55 * units.MHz, 1000 * units.MHz])
            passband[0]: lower boundary of filter, passband[1]: upper boundary of filter

            a dict can be used to specify a different bandwidth per channel, the key is the channel_id
        filter_type: string or dict

            * 'rectangular': perfect straight line filter
            * 'butter': butterworth filter from scipy
            * 'butterabs': absolute of butterworth filter from scipy
            * 'gaussian_tapered' : a rectangular bandpass filter convolved with a Gaussian
            * 'hann_tapered' : a rectangular bandpass filter with the ends replaced by a half-Hann window to applied
              in the time domain. In this case the passband parameter is ignored
            * 'FIR <type> <parameter>' - see below for FIR filter options

            or any filter that is implemented in :mod:`NuRadioReco.detector.filterresponse`. In this case the
            passband parameter is ignored

            a dict can be used to specify a different bandwidth per channel, the key is the channel_id
        order: int (optional, default 2) or dict
            for a butterworth filter: specifies the order of the filter
        rp: float
            The maximum ripple allowed below unity gain in the passband. Specified in decibels, as a positive number.
            (for chebyshev filter)

            a dict can be used to specify a different bandwidth per channel, the key is the channel_id
        roll_width : float, default=2.5MHz
            Determines the sigma of the Gaussian to be used in the convolution of the rectangular filter.
            (Relevant for the Gaussian tapered filter)
        half_hann_percent : float, default=0.1
            The size of the half-Hann window expressed as a percentage of the length of the trace.
            (Relevant for the Hann tapered filter)

        Notes
        -----
        Added Jan-07-2018 by robert.lahmann@fau.de:
        FIR filter:
        see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html
        for window types; some windows need additional parameters; Default window is hamming
        (does not need parameters). Examples:

        * filter_type='FIR' : use hamming window
        * filter_type='FIR hamming': same as above
        * filter_type='FIR kaiser 10' : Use Kaiser window with beta parameter 10
        * filter_type='FIR kaiser' : Use Kaiser window with default beta parameter 6

        In principle, window names are just passed on to signal.firwin(), but if parameters
        are required, then these cases must be explicitly implemented in the code below.

        The four main filter types can be implemented:

        * LP: passband[0]=None, passband[1]  = f_cut
        * HP: passband[0]=f_cut, passband[1] = None
        * BP: passband[0]=f_cut_low, passband[1] = f_cut_high
        * BS: passband[0]=f_cut_high, passband[1] = f_cut_low (i.e. passband[0] > passband[1])

        """
        if passband is None:
            passband = [55 * units.MHz, 1000 * units.MHz]
        for channel in station.iter_channels():
            tmp_passband, tmp_order, tmp_filter_type, tmp_rp, tmp_roll_width, tmp_half_hann_percent = \
                self.get_filter_arguments(
                    channel.get_id(), passband, filter_type, order, rp, roll_width, half_hann_percent
                )
            self._apply_filter(channel, tmp_passband, tmp_filter_type, tmp_order, tmp_rp, tmp_roll_width,
                               tmp_half_hann_percent, False)

    def get_filter(self, frequencies, station_id, channel_id, det, passband, filter_type,
                   order=2, rp=None, roll_width=2.5 * units.MHz, half_hann_percent=None):
        """
        helper function to return the filter that the module applies.

        The unused parameters station_id, channel_id, det are needed to have the same signature as the
        ``get_filter`` functions of other modules, e.g. the hardwareResponseIncorporator.

        Note that this function returns the filter response in the **frequency domain**. Filters
        which are applied in the time domain (e.g. 'FIR', 'hann_tapered') will raise a
        ``NotImplementedError``

        Parameters
        ----------
        frequencies: array of floats
            the frequency array for which the filter should be returned
        station_id: int
            the station id
        channel_id: int
            the channel id
        det: detector instance
            the detector
        passband: list or dict of lists, (default: [55 * units.MHz, 1000 * units.MHz])
            passband[0]: lower boundary of filter, passband[1]: upper boundary of filter
            a dict can be used to specify a different bandwidth per channel, the key is the channel_id
        filter_type: string or dict

            * 'rectangular': perfect straight line filter
            * 'butter': butterworth filter from scipy
            * 'butterabs': absolute of butterworth filter from scipy
            * 'gaussian_tapered' : Gaussian tapered version of the rectangular filter

            or any filter that is implemented in :mod:`NuRadioReco.detector.filterresponse`. In this case the
            passband parameter is ignored

            a dict can be used to specify a different bandwidth per channel, the key is the channel_id
        order: int (optional, default 2) or dict
            for a butterworth filter: specifies the order of the filter
        rp: float
            The maximum ripple allowed below unity gain in the passband. Specified in decibels, as a positive number.
            (for chebyshev filter)

            a dict can be used to specify a different bandwidth per channel, the key is the channel_id
        roll_width : float, default=2.5MHz
            Determines the sigma of the Gaussian to be used in the convolution of the rectangular filter.
            (Relevant for the Gaussian tapered filter)

        Other Parameters
        ----------------
        half_hann_percent: None
            This parameter is included to have the same signature as the :func:`run` method.
            As it is only relevant for the hann_tapered filter it is not actually used here.

        Returns
        -------
        array of complex floats
            the complex filter amplitudes
        """
        tmp_passband, tmp_order, tmp_filter_type, tmp_rp, tmp_roll_width, _ = \
            self.get_filter_arguments(channel_id, passband, filter_type, order, rp, roll_width)
        filter_args = frequencies, tmp_passband, tmp_order, tmp_filter_type, tmp_rp, tmp_roll_width
        if self.__filter_args is not None:
            if not test_equality(filter_args, self.__filter_args):
                self.__filter_cached = None
        if (self.__filter_cached is None) or (not self.__caching):
            self.__filter_args = filter_args
            self.__filter_cached = bandpass_filter.get_filter_response(
                frequencies, tmp_passband, tmp_filter_type, tmp_order, tmp_rp, tmp_roll_width
            )

        return self.__filter_cached

    def _apply_filter(self, channel, passband, filter_type, order,
                      rp=None, roll_width=None, half_hann_percent=None, is_efield=False):

        frequencies = channel.get_frequencies()
        trace_fft = channel.get_frequency_spectrum()
        sample_rate = channel.get_sampling_rate()

        # for FIR filters, it is easier to set the trace rather than the FFT to apply the
        # filter response. Eventually, I (Robert Lahmann) might figure out a way to set the
        # FFT (as is done for the IIR filters corresponding to analog filters) but for now
        # I am setting a variable to decide whether the FFT or time trace shall be used to
        # set the trace.

        isFIR = False

        if filter_type == 'rectangular':
            trace_fft *= self.get_filter(frequencies, 0, 0, None, passband, filter_type)
        elif filter_type == 'butter':
            trace_fft *= self.get_filter(frequencies, 0, 0, None, passband, filter_type, order)
        elif filter_type == 'butterabs':
            trace_fft *= self.get_filter(frequencies, 0, 0, None, passband, filter_type, order)
        elif filter_type == 'cheby1':
            trace_fft *= self.get_filter(frequencies, 0, 0, None, passband, filter_type, order, rp)
        elif filter_type == 'gaussian_tapered':
            trace_fft *= self.get_filter(frequencies, 0, 0, None, passband, filter_type, order, rp, roll_width)
        elif filter_type == 'hann_tapered':
            trace_fir = channel.get_trace()
            trace_fir *= signal_processing.half_hann_window(len(trace_fir), half_percent=half_hann_percent)
            # This filter is applied in the time domain, so use the isFIR variable to set the time trace back
            isFIR = True
        elif filter_type.find('FIR') >= 0:
            # print('This is a FIR filter')
            firarray = filter_type.split()
            if len(firarray) == 1:
                wtype = 'hamming'
                # print('hamming window')
            else:
                wtype = firarray[1]
                if wtype.find('kaiser') >= 0:
                    if len(firarray) > 2:
                        beta = float(firarray[2])
                    else:
                        beta = 6.0
                    wtype = ('kaiser', beta)
            # print('window type: ', wtype)
            Nfir = order + 1
            if passband[0] is None:
                # this is a low pass filter
                pass_zero = True
                fcut = passband[1]
            elif passband[1] is None or passband[1] / sample_rate >= 0.5:
                # this is a high pass filter
                pass_zero = False
                fcut = passband[0]
            elif passband[1] > passband[0]:
                # this is a bandpass filter
                pass_zero = False
                fcut = passband
            elif passband[0] > passband[1]:
                # this is a bandstop filter
                pass_zero = True
                fcut = [passband[1], passband[0]]
            else:
                # something went wrong!!
                print("Error, could not define filter type")
            taps = signal.firwin(Nfir, fcut, window=wtype, scale=False, pass_zero=pass_zero, fs=sample_rate)

            if (Nfir // 2) * 2 == Nfir:
                print("odd filter order, rolling is off by T_s/2")

            ndelay = int(0.5 * (Nfir - 1))
            trace_fir = signal.lfilter(taps, 1.0, channel.get_trace())
            trace_fir = np.roll(trace_fir, -ndelay)
            isFIR = True
        else:
            trace_fft *= self.get_filter(frequencies, 0, 0, None, passband, filter_type)
        if isFIR:
            channel.set_trace(trace_fir, sample_rate)
        else:
            channel.set_frequency_spectrum(trace_fft, sample_rate)

    def end(self):
        pass

def test_equality(a, b):
    """
    Test if two things are equal.

    Generalizes a==b to support lists, lists of lists and arrays etc.
    Will return True if and only if a and b are equal.

    Parameters
    ----------
    a: object | list | array
    b: object | list | array

    Returns
    -------
    is_equal: bool
        True if and only if a == b

    """

    if not isinstance(a, list):
        a = list(a)
    if not isinstance(b, list):
        b = list(b)
    if len(a) != len(b):
        return False

    isequal = True
    for i in range(len(a)):
        if hasattr(a[i], '__len__'):
            if not hasattr(b[i], '__len__'):
                return False
            elif len(a[i]) != len(b[i]):
                return False
            elif isinstance(a[i], np.ndarray):
                isequal &= (np.all(a[i] == b[i]))
            else:
                isequal &= all([u == v for u, v in zip(a[i], b[i])])
        else:
            isequal &= a[i] == b[i]

    return isequal