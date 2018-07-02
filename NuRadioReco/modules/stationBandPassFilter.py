import numpy as np
from NuRadioReco.utilities import units
import scipy.signal


class stationBandPassFilter:
    """
    """

    def begin(self):
        pass

    def run(self, evt, station, det, passband=[55 * units.MHz, 1000 * units.MHz],
            filter_type='rectangular',
            debug=False):

        frequencies = station.get_frequencies()
        trace_fft = station.get_frequency_spectrum()
        if(filter_type == 'rectangular'):
            trace_fft[:, np.where(frequencies < passband[0])] = 0.
            trace_fft[:, np.where(frequencies > passband[1])] = 0.
        elif(filter_type == 'butter10'):
            b, a = scipy.signal.butter(10, passband, 'bandpass', analog=True)
            w, h = scipy.signal.freqs(b, a, frequencies)
            trace_fft *= h
        elif(filter_type == 'butter10abs'):
            b, a = scipy.signal.butter(10, passband, 'bandpass', analog=True)
            w, h = scipy.signal.freqs(b, a, frequencies)
            trace_fft *= np.abs(h)
        station.set_frequency_spectrum(trace_fft, station.get_sampling_rate())

    def end(self):
        pass
