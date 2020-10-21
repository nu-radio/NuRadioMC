from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units, bandpass_filter


class electricFieldBandPassFilter:
    """
    """

    def begin(self):
        pass

    @register_run()
    def run(self, evt, station, det, passband=None,
            filter_type='rectangular', order=2):
        """
        Applies bandpass filter to electric field

        Parameters
        ----------
        evt: event

        station: station

        det: detector

        passband: seq, array (default: [55 * units.MHz, 1000 * units.MHz])
            passband for filter, in phys units
        filter_type: string
            'rectangular': perfect straight line filter
            'butter': butterworth filter from scipy
            'butterabs': absolute of butterworth filter from scipy
        order: int (optional, default 2)
            for a butterworth filter: specifies the order of the filter
        """
        if passband is None:
            passband = [55 * units.MHz, 1000 * units.MHz]
        for efield in station.get_electric_fields():
            frequencies = efield.get_frequencies()
            trace_fft = efield.get_frequency_spectrum()
            filter_response = bandpass_filter.get_filter_response(frequencies, passband, filter_type, order)
            trace_fft *= filter_response
            efield.set_frequency_spectrum(trace_fft, efield.get_sampling_rate())

    def end(self):
        pass
