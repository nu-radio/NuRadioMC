import numpy as np
from scipy import signal
import logging
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units, bandpass_filter


class channelBandPassFilter:
    """
    Band pass filters the channels using different band-pass filters.
    """

    def __init__(self):
        self.__t = 0
        self.begin()
        self.logger = logging.getLogger('NuRadioReco.channelBandPassFilter')

    def begin(self):
        pass

    def get_filter_arguments(self, channel_id, passband, filter_type, order=2, rp=None):
        if(isinstance(passband, dict)):
            tmp_passband = passband[channel_id]
        else:
            tmp_passband = passband

        if(isinstance(order, dict)):
            tmp_order = order[channel_id]
        else:
            tmp_order = order

        if(isinstance(filter_type, dict)):
            tmp_filter_type = filter_type[channel_id]
        else:
            tmp_filter_type = filter_type

        if(isinstance(rp, dict)):
            tmp_rp = rp[channel_id]
        else:
            tmp_rp = rp
        return tmp_passband, tmp_order, tmp_filter_type, tmp_rp

    @register_run()
    def run(self, evt, station, det, passband=None,
            filter_type='rectangular', order=2, rp=None):
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
            'rectangular': perfect straight line filter
            'butter': butterworth filter from scipy
            'butterabs': absolute of butterworth filter from scipy
            or any filter that is implemented in NuRadioReco.detector.filterresponse. In this case the
            passband parameter is ignored
            or 'FIR <type> <parameter>' - see below for FIR filter options

            a dict can be used to specify a different bandwidth per channel, the key is the channel_id
        order: int (optional, default 2) or dict
            for a butterworth filter: specifies the order of the filter
        rp: float
            The maximum ripple allowed below unity gain in the passband. Specified in decibels, as a positive number.
            (for chebyshev filter)

            a dict can be used to specify a different bandwidth per channel, the key is the channel_id
        Added Jan-07-2018 by robert.lahmann@fau.de:
        FIR filter:
        see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html
        for window types; some windows need additional parameters; Default window is hamming
        (does not need parameters). Examples:
        filter_type='FIR' : use hamming window
        filter_type='FIR hamming': same as above
        filter_type='FIR kaiser 10' : Use Kaiser window with beta parameter 10
        filter_type='FIR kaiser' : Use Kaiser window with default beta parameter 6

        In principal, window names are just passed on to signal.firwin(), but if parameters
        are required, then these cases must be explicitly implemented in the code below.

        The four main filter types can be implemented:
        LP: passband[0]=None, passband[1]  = f_cut
        HP: passband[0]=f_cut, passband[1] = None
        BP: passband[0]=f_cut_low, passband[1] = f_cut_high
        BS: passband[0]=f_cut_high, passband[1] = f_cut_low (i.e. passband[0] > passband[1])

        """
        if passband is None:
            passband = [55 * units.MHz, 1000 * units.MHz]
        for channel in station.iter_channels():
            tmp_passband, tmp_order, tmp_filter_type, tmp_rp = self.get_filter_arguments(channel.get_id(), passband, filter_type, order, rp)
            self._apply_filter(channel, tmp_passband, tmp_filter_type, tmp_order, tmp_rp, False)

    def get_filter(self, frequencies, station_id, channel_id, det, passband, filter_type, order=2, rp=None):
        """
        helper function to return the filter that the module applies.

        The unused parameters station_id, channel_id, det are needed to have the same signature as the
        `get_filter` functions of other modules, e.g. the hardwareResponseIncorporator.

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
            'rectangular': perfect straight line filter
            'butter': butterworth filter from scipy
            'butterabs': absolute of butterworth filter from scipy
            or any filter that is implemented in NuRadioReco.detector.filterresponse. In this case the
            passband parameter is ignored
            or 'FIR <type> <parameter>' - see below for FIR filter options

            a dict can be used to specify a different bandwidth per channel, the key is the channel_id
        order: int (optional, default 2) or dict
            for a butterworth filter: specifies the order of the filter

        rp: float
            The maximum ripple allowed below unity gain in the passband. Specified in decibels, as a positive number.
            (for chebyshev filter)

            a dict can be used to specify a different bandwidth per channel, the key is the channel_id



        Returns
        -----------------
         array of complex floats
            the complex filter amplitudes
        """
        tmp_passband, tmp_order, tmp_filter_type, tmp_rp = self.get_filter_arguments(channel_id, passband, filter_type, order, rp)
        return bandpass_filter.get_filter_response(frequencies, tmp_passband, tmp_filter_type, tmp_order, tmp_rp)

    def _apply_filter(self, channel, passband, filter_type, order, rp=None, is_efield=False):

        frequencies = channel.get_frequencies()
        trace_fft = channel.get_frequency_spectrum()
        sample_rate = channel.get_sampling_rate()

        # for FIR filters, it is easier to set the trace rather than the FFT to apply the
        # filter response. Eventually, I (Robert Lahmann) might figure out a way to set the
        # FFT (as is done for the IIR filters corresponding to analog filters) but for now
        # I am setting a variable to decide whether the FFT or time trace shall be used to
        # set the trace.

        isFIR = False

        if(filter_type == 'rectangular'):
            trace_fft *= self.get_filter(frequencies, 0, 0, None, passband, filter_type)
        elif(filter_type == 'butter'):
            trace_fft *= self.get_filter(frequencies, 0, 0, None, passband, filter_type, order)
        elif(filter_type == 'butterabs'):
            trace_fft *= self.get_filter(frequencies, 0, 0, None, passband, filter_type, order)
        elif(filter_type == 'cheby1'):
            trace_fft *= self.get_filter(frequencies, 0, 0, None, passband, filter_type, order, rp)
        elif(filter_type.find('FIR') >= 0):
            # print('This is a FIR filter')
            firarray = filter_type.split()
            if (len(firarray) == 1):
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
            if (passband[0] is None):
                # this is a low pass filter
                pass_zero = True
                fcut = passband[1]
            elif (passband[1] is None or passband[1] / sample_rate >= 0.5):
                # this is a high pass filter
                pass_zero = False
                fcut = passband[0]
            elif (passband[1] > passband[0]):
                # this is a bandpass filter
                pass_zero = False
                fcut = passband
            elif(passband[0] > passband[1]):
                # this is a bandstop filter
                pass_zero = True
                fcut = [passband[1], passband[0]]
            else:
                # something went wrong!!
                print("Error, could not define filter type")
            taps = signal.firwin(Nfir, fcut, window=wtype, scale=False, pass_zero=pass_zero, fs=sample_rate)

            if ((Nfir // 2) * 2 == Nfir):
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
