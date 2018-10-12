import numpy as np
from NuRadioReco.utilities import units
import scipy.signal
from NuRadioReco.detector import filterresponse


class channelBandPassFilter:
    """
    Band pass filters the channels using different band-pass filters.
    """

    def begin(self):
        pass

    def run(self, evt, station, det, passband=[55 * units.MHz, 1000 * units.MHz],
            filter_type='rectangular', order=2):
        """
        Run the filter

        Parameters
        ---------

        evt, station, det
            Event, Station, Detector
        passband: list
            passband[0]: lower boundary of filter, passband[1]: upper boundary of filter
        filter_type: string
            'rectangular': perfect straight line filter
            'butter10': butterworth filter from scipy
            'butter10abs': absolute of butterworth filter from scipy
            or any filter that is implemented in NuRadioReco.detector.filterresponse. In this case the
            passband parameter is ignored
        order: int (optional, default 2)
            for a butterworth wilter: specifies the order of the filter

        """
        channels = station.get_channels()
        for channel in channels:
            frequencies = channel.get_frequencies()
            trace_fft = channel.get_frequency_spectrum()

            if(filter_type == 'rectangular'):
                trace_fft[np.where(frequencies < passband[0])] = 0.
                trace_fft[np.where(frequencies > passband[1])] = 0.
            elif(filter_type == 'butter'):
                mask = frequencies > 0
                b, a = scipy.signal.butter(order, passband, 'bandpass', analog=True)
                w, h = scipy.signal.freqs(b, a, frequencies[mask])
                trace_fft[mask] *= h
            elif(filter_type == 'butterabs'):
                mask = frequencies > 0
                b, a = scipy.signal.butter(order, passband, 'bandpass', analog=True)
                w, h = scipy.signal.freqs(b, a, frequencies[mask])
                trace_fft[mask] *= np.abs(h)
            else:
                mask = frequencies > 0
                filt = filterresponse.get_filter_response(frequencies[mask], filter_type)
                trace_fft[mask] *= filt
            channel.set_frequency_spectrum(trace_fft, channel.get_sampling_rate())

    def end(self):
        pass
