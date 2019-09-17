import numpy as np
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units
import scipy.signal


class electricFieldBandPassFilter:
    """
    """

    def begin(self):
        pass

    @register_run()
    def run(self, evt, station, det, passband=[55 * units.MHz, 1000 * units.MHz],
            filter_type='rectangular', order=2, debug=False):
        """
        Applies bandpass filter to electric field

        Parameters
        ----------
        evt: event

        station: station

        det: detector

        passband: seq, array
            passband for filter, in phys units
        filter_type: string
            'rectangular': perfect straight line filter
            'butter': butterworth filter from scipy
            'butterabs': absolute of butterworth filter from scipy
        order: int (optional, default 2)
            for a butterworth filter: specifies the order of the filter
        debug: bool
            set debug

        """

        for efield in station.get_electric_fields():
            frequencies = efield.get_frequencies()
            trace_fft = efield.get_frequency_spectrum()
            if(filter_type == 'rectangular'):
                trace_fft[:, np.where(frequencies < passband[0])] = 0.
                trace_fft[:, np.where(frequencies > passband[1])] = 0.
            elif(filter_type == 'butter'):
                b, a = scipy.signal.butter(order, passband, 'bandpass', analog=True)
                w, h = scipy.signal.freqs(b, a, frequencies)
                trace_fft *= h
            elif(filter_type == 'butterabs'):
                b, a = scipy.signal.butter(order, passband, 'bandpass', analog=True)
                w, h = scipy.signal.freqs(b, a, frequencies)
                trace_fft *= np.abs(h)
            efield.set_frequency_spectrum(trace_fft, efield.get_sampling_rate())

    def end(self):
        pass
