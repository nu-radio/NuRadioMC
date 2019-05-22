import numpy as np
from NuRadioReco.utilities import units
import scipy.signal


class electricFieldBandPassFilter:
    """
    """

    def begin(self):
        pass

    def run(self, evt, station, det, passband=[55 * units.MHz, 1000 * units.MHz],
            filter_type='rectangular',
            debug=False):
        """
        Applies bandpass filter to electric field

        Parameters
        ----------
        evt: event

        station: station

        det: detector

        passband: seq, array
            passband for filter, in phys units
        filter_type: str
            chose filter type, rectangular, butter10, butter10abs
        debug: bool
            set debug

        """

        for efield in station.get_electric_fields():
            frequencies = efield.get_frequencies()
            trace_fft = efield.get_frequency_spectrum()
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
            efield.set_frequency_spectrum(trace_fft, efield.get_sampling_rate())

    def end(self):
        pass
