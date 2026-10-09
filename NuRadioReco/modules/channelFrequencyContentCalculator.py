import logging

from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units, trace_utilities
from NuRadioReco.framework.parameters import channelParameters as chp

logger = logging.getLogger('NuRadioReco.channelFrequencyContentCalculator')


class ChannelFrequencyContentCalculator:
    """
    Calculates the relative frequency content of channel waveforms.

    Per channel it stores the energy ratios of target bands to a reference band, the median
    frequency and the spectral centroid. Energies are taken from ``|spectrum|**2``
    (energy spectral density); the trace length cancels in all quantities.
    The full trace is used, i.e. noise contributes and no signal window is applied.
    """

    def __init__(self):
        self.begin()

    def begin(
            self, ref_band=(200 * units.MHz, 600 * units.MHz), target_bands=((80 * units.MHz, 200 * units.MHz),),
            normalize_bandwidth=True, log_ratio=False, quantile_band=None):
        """
        Bands are (f_min, f_max) tuples in internal units, e.g. ``(80 * units.MHz, 200 * units.MHz)``.
        The default bands are illustrative.

        Parameters
        ----------
        ref_band: tuple
            Reference band.
        target_bands: tuple or list of tuples
            Target band(s); a ratio to the reference band is stored for each.
        normalize_bandwidth: bool
            Use the mean spectral density per band, i.e. a flat spectrum gives a ratio of 1.
            See `NuRadioReco.utilities.trace_utilities.get_band_energy_ratio`.
        log_ratio: bool
            Store log10 of the ratios.
        quantile_band: tuple or None
            Band restricting the median frequency and centroid. Default: full spectrum.
        """
        if not hasattr(target_bands[0], "__len__"):  # single band given
            target_bands = [target_bands]

        self.__ref_band = tuple(ref_band)
        self.__target_bands = [tuple(band) for band in target_bands]
        self.__normalize_bandwidth = normalize_bandwidth
        self.__log_ratio = log_ratio
        self.__quantile_band = quantile_band

    @register_run()
    def run(self, evt, station, det=None):
        """
        Sets the channel parameters ``spectral_energy_ratios`` (list, order of ``target_bands``),
        ``median_frequency`` and ``spectral_centroid`` for all channels of the station.

        Parameters
        ----------
        evt: `NuRadioReco.framework.event.Event`
        station: `NuRadioReco.framework.station.Station`
        det: detector object
            Not used.
        """
        for channel in station.iter_channels():
            freqs = channel.get_frequencies()
            spec = channel.get_frequency_spectrum()

            channel[chp.spectral_energy_ratios] = [
                trace_utilities.get_band_energy_ratio(
                    freqs, spec, target_band, self.__ref_band,
                    normalize_bandwidth=self.__normalize_bandwidth, log=self.__log_ratio)
                for target_band in self.__target_bands]
            channel[chp.median_frequency] = trace_utilities.get_spectral_energy_quantile_frequency(
                freqs, spec, 0.5, band=self.__quantile_band)
            channel[chp.spectral_centroid] = trace_utilities.get_spectral_centroid(
                freqs, spec, band=self.__quantile_band)

    def end(self):
        pass
