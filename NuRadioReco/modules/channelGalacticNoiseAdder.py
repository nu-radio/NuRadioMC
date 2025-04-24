from NuRadioReco.utilities import units, ice, geometryUtilities, signal_processing
from NuRadioReco.modules.base.module import register_run
import NuRadioReco.framework.channel
import NuRadioReco.framework.sim_station
import NuRadioReco.detector.antennapattern
import logging
import warnings
import numpy as np
import scipy.constants
import scipy.interpolate
import functools
from contextlib import redirect_stdout
from numpy.random import Generator, Philox
import healpy
import astropy.coordinates
import astropy.units

logger = logging.getLogger('NuRadioReco.channelGalacticNoiseAdder')

try:
    from pygdsm import (
        GlobalSkyModel16,
        GlobalSkyModel,
        LowFrequencySkyModel,
        HaslamSkyModel,
    )
except ImportError as e:
    logger.error(
        "To use the channelGalacticNoiseAdder, 'pygdsm' needs to be installed:\n\n"
        "\t pip install git+https://github.com/telegraphic/pygdsm\n"
        )
    raise(e)

try:
    with redirect_stdout(None): # suppress (usually irrelevant) print statements from pylfmap
        from pylfmap import LFmap  # Documentation: https://github.com/F-Tomas/pylfmap needs cfitsio installation
except ImportError:
    logger.info(
        "pylfmap import failed. Consider installing it from "
        "https://github.com/F-Tomas/pylfmap to use LFmap as sky model.")
except IndexError: # this is a common error if cfitsio is not found... there are probably others
    logger.error(
        "pylfmap import failed. This might be because you do not have a working "
        "installation of cfitsio. See https://github.com/F-Tomas/pylfmap/issues/2 for potential tips")


class channelGalacticNoiseAdder:
    """
    Class that simulates the noise produced by galactic radio emission

    Uses the pydgsm package (https://github.com/telegraphic/pygdsm), which provides
    radio background data based on Oliveira-Costa et al. (2008) (https://arxiv.org/abs/0802.1525)
    and Zheng et al. (2016) (https://arxiv.org/abs/1605.04920)

    The radio sky model is evaluated on a number of points above the horizon
    folded with the antenna response. Since evaluating every frequency individually
    would be too slow, the model is evaluated for a few frequencies and the log10
    of the brightness temperature is interpolated in between.
    """

    def __init__(self):
        self.__n_side = None
        self.__interpolation_frequencies = None
        self.__radio_sky = None
        self.__noise_temperatures = None
        self.__antenna_pattern_provider = NuRadioReco.detector.antennapattern.AntennaPatternProvider()

    def begin(
            self,
            skymodel=None,
            debug=False,
            n_side=4,
            freq_range=None,
            interpolation_frequencies=None,
            seed=None,
            caching=True
    ):
        """
        Set up important parameters for the module

        Parameters
        ----------
        skymodel: {'gsm2008', 'lfmap', 'lfss', 'gsm2016', 'haslam'}, optional
            Choose the sky model to use. If none is provided, the Global Sky Model (2008) is used as a default.
        debug: bool, default: False
            Deprecated. Will be removed in future versions.
        n_side: int, default: 4
            The n_side parameter of the healpix map. Has to be power of 2
            The radio skymap is downsized to the resolution specified by the n_side
            parameter and for every pixel above the horizon the radio noise coming
            from that direction is calculated. The number of pixels used is
            12 * n_side ** 2, so a larger value for n_side will result better accuracy
            but also greatly increase computing time.
        freq_range: array of len=2, default: [10, 1000] * units.MHZ
            The sky brightness temperature will be evaluated for the frequencies
            within this limit. Brightness temperature for frequencies in between are
            calculated by interpolation the log10 of the temperature
            The interpolation_frequencies have to cover the entire passband
            specified in the run method.
        interpolation_frequencies: array of frequencies to interpolate to.
            Kept for historic purposes with intention to deprecate in the future.
        seed : {None, int, array_like[ints], SeedSequence}, optional
            The seed that is passed on to the `numpy.random.Philox` bitgenerator used for random
            number generation.
        caching: bool, default: True
            If True, the antenna response is cached for each channel. This can speed up this module
            by a lot. If the frequencies of the channels change, the cache is cleared.
        """
        if debug:
            warnings.warn("This argument is deprecated and will be removed in future versions.", DeprecationWarning)

        self.__random_generator = Generator(Philox(seed))
        self.__n_side = n_side
        self.solid_angle = healpy.pixelfunc.nside2pixarea(self.__n_side, degrees=False)

        self.__caching = caching
        self.__freqs = None
        if self.__caching and self.__n_side >= 10:
            logger.warning(
                "Caching for the vector effective length is enabled (with `maxsize=1024`) and `n_side >= 10`, and thus "
                "it produces to many different caching entries for two antenna models to be stored of one `station_time`. "
                "Either decrease `n_side` or increase `maxsize` (has to be done in the source code).")

        if interpolation_frequencies is None:
            if freq_range is None:
                freq_range = np.array([10, 1000]) * units.MHz

            # define interpolation frequencies. Set in logarithmic range from freq_range[0] to freq_range[1],
            # rounded to MHz to avoid import errors from LFmap and tabulated models.
            self.__interpolation_frequencies = np.around(np.logspace(*np.log10(freq_range), num=15), 3)
        else:
            self.__interpolation_frequencies = interpolation_frequencies
            logger.warning("DeprecationWarning: Optional argument 'interpolation_frequencies' was replaced by 'freq_range'.")

        # initialise sky model
        try:
            if skymodel is None:
                sky_model = GlobalSkyModel(freq_unit="MHz")
                logger.info("No sky model specified. Using standard: Global Sky Model (2008). Available models: "
                            "gsm2008, lfmap, lfss, gsm2016, haslam")
            elif skymodel.lower() == 'lfss':
                sky_model = LowFrequencySkyModel(freq_unit="MHz")
                logger.info("Using LFSS as sky model")
            elif skymodel.lower() == 'gsm2008':
                sky_model = GlobalSkyModel(freq_unit="MHz")
                logger.info("Using GSM2008 as sky model")
            elif skymodel.lower() == 'gsm2016':
                sky_model = GlobalSkyModel16(freq_unit="MHz")
                logger.info("Using GSM2016 as sky model")
            elif skymodel.lower() == 'haslam':
                sky_model = HaslamSkyModel(freq_unit="MHz", spectral_index=-2.53)
                logger.info("Using Haslam as sky model")
            elif skymodel.lower() == 'lfmap':
                sky_model = LFmap()
                logger.info("Using LFmap as sky model")
            else:
                logger.error(f"Sky model {skymodel} unknown. Defaulting to Global Sky Model (2008).")
                sky_model = GlobalSkyModel(freq_unit="MHz")

        except NameError:
            logger.error(f"Could not find {skymodel} skymodel. Do you have the correct package installed? \n"
                        f"Defaulting to Global Sky Model (2008) as sky model.")
            sky_model = GlobalSkyModel(freq_unit="MHz")

        self.__noise_temperatures = np.zeros(
            (len(self.__interpolation_frequencies), healpy.pixelfunc.nside2npix(self.__n_side))
        )
        logger.info("generating noise temperatures")

        # generating sky maps and noise temperatures from chosen sky model in given frequency range
        for i_freq, noise_freq in enumerate(self.__interpolation_frequencies):
            self.__radio_sky = sky_model.generate(noise_freq / units.MHz)
            self.__radio_sky = healpy.pixelfunc.ud_grade(self.__radio_sky, self.__n_side)
            self.__noise_temperatures[i_freq] = self.__radio_sky


    @functools.lru_cache(maxsize=1024)
    def _get_cached_antenna_response(self, ant_pattern, zen, azi, *ant_orient):
        """
        Returns the cached antenna reponse for a given antenna patter, antenna orientation
        and signal arrival direction. This wrapper is necessary as arrays and list are not
        hashable (i.e., can not be used as arguments in functions one wants to cache).
        This module ensures that the cache is clearied if the vector `self.__freqs` changes.
        """
        return ant_pattern.get_antenna_response_vectorized(self.__freqs, zen, azi, *ant_orient)


    @register_run()
    def run(
            self,
            event,
            station,
            detector,
            passband=None,
            excluded_channels=None
    ):

        """
        Adds noise resulting from galactic radio emission to the channel traces

        Parameters
        ----------
        event: Event object
            The event containing the station to whose channels noise shall be added
        station: Station object
            The station whose channels noise shall be added to
        detector: Detector object
            The detector description
        passband: list of float, optional
            Lower and upper bound of the frequency range in which noise shall be
            added. The default (no passband specified) is [10, 1000] MHz
        excluded_channels: list, default=None
            A list containing the channels IDs to exclude per station.
            If None, all channels of the selected station in the detector are used.
        """
        if excluded_channels is None:
            selected_channel_ids = station.get_channel_ids()
            logger.debug(f"Using all channels: {selected_channel_ids}")
        else:
            selected_channel_ids = [channel_id for channel_id in station.get_channel_ids() if channel_id not in excluded_channels]
            logger.debug(f"Using selected channel ids: {selected_channel_ids}")

        if self.__noise_temperatures is None: # check if .begin has been called, give helpful error message if not
            msg = "channelGalacticNoiseAdder was not initialized correctly. Maybe you forgot to call `.begin()`?"
            logger.error(msg)
            raise ValueError(msg)

        # check that or all channels channel.get_frequencies() is identical
        last_freqs = None
        for channel in station.iter_channels(use_channels=selected_channel_ids):
            if last_freqs is not None and (
                    not np.allclose(last_freqs, channel.get_frequencies(), rtol=0, atol=0.1 * units.MHz)):
                logger.error("The frequencies of each channel must be the same, but they are not!")
                return

            last_freqs = channel.get_frequencies()

        freqs = last_freqs

        # If we cache the antenna pattern, we need to make sure that the frequencies have not changed
        # between stations. If they have, we need to clear the cache.
        if self.__caching:
            if self.__freqs is None:
                self.__freqs = freqs
            else:
                if len(self.__freqs) != len(freqs):
                    self.__freqs = freqs
                    self._get_cached_antenna_response.cache_clear()
                    logger.warning(
                        "Frequencies have changed (array length). Clearing antenna response cache. "
                        "(If this happens often, something might be wrong...")
                elif not np.allclose(self.__freqs, freqs, rtol=0, atol=0.01 * units.MHz):
                    self.__freqs = freqs
                    self._get_cached_antenna_response.cache_clear()
                    logger.warning(
                        "Frequencies have changed (values). Clearing antenna response cache. "
                        "(If this happens often, something might be wrong...")


        if passband is None:
            passband = [10 * units.MHz, 1000 * units.MHz]

        passband_filter = (freqs > passband[0]) & (freqs < passband[1])

        site_latitude, site_longitude = detector.get_site_coordinates(station.get_id())
        station_time = station.get_station_time()

        local_coordinates = get_local_coordinates((site_latitude, site_longitude), station_time, self.__n_side)

        n_ice = ice.get_refractive_index(-0.01, detector.get_site(station.get_id()))
        n_air = ice.get_refractive_index(depth=1, site=detector.get_site(station.get_id()))

        channel_spectra = {}
        for channel in station.iter_channels(use_channels=selected_channel_ids):
            channel_spectra[channel.get_id()] = channel.get_frequency_spectrum()

        for i_pixel in range(healpy.pixelfunc.nside2npix(self.__n_side)):
            azimuth = local_coordinates[i_pixel].az.rad
            zenith = np.pi / 2. - local_coordinates[i_pixel].alt.rad # this is the in-air zenith

            if zenith > 90. * units.deg:
                continue

            if n_ice != n_air: # consider signal reflection at ice surface
                t_theta = geometryUtilities.get_fresnel_t_p(zenith, n_ice, n_air)
                t_phi = geometryUtilities.get_fresnel_t_s(zenith, n_ice, n_air)
                fresnel_zenith = geometryUtilities.get_fresnel_angle(zenith, n_ice, n_air)
            else: # we are at an in-air site; no refraction
                t_theta = 1
                t_phi = 1
                fresnel_zenith = zenith

            if fresnel_zenith is None:
                continue

            temperature_interpolator = scipy.interpolate.interp1d(
                self.__interpolation_frequencies, np.log10(self.__noise_temperatures[:, i_pixel]), kind='quadratic')
            noise_temperature = np.power(10, temperature_interpolator(freqs[passband_filter]))

            efield_amplitude = signal_processing.get_electric_field_from_temperature(
                freqs[passband_filter], noise_temperature, self.solid_angle)

            # assign random phases to electric field
            noise_spectrum = np.zeros((3, freqs.shape[0]), dtype=complex)
            phases = self.__random_generator.uniform(0, 2. * np.pi, len(efield_amplitude))

            noise_spectrum[1][passband_filter] = np.exp(1j * phases) * efield_amplitude
            noise_spectrum[2][passband_filter] = np.exp(1j * phases) * efield_amplitude

            channel_noise_spec = np.zeros_like(noise_spectrum)

            for channel in station.iter_channels(use_channels=selected_channel_ids):

                channel_pos = detector.get_relative_position(station.get_id(), channel.get_id())
                if channel_pos[2] < 0:
                    curr_t_theta = t_theta
                    curr_t_phi = t_phi
                    curr_fresnel_zenith = fresnel_zenith
                    curr_n = n_ice
                else: # we are in air
                    curr_t_theta = 1
                    curr_t_phi = 1
                    curr_fresnel_zenith = zenith
                    curr_n = n_air

                antenna_pattern = self.__antenna_pattern_provider.load_antenna_pattern(
                    detector.get_antenna_model(station.get_id(), channel.get_id()))
                antenna_orientation = detector.get_antenna_orientation(station.get_id(), channel.get_id())

                # calculate the phase offset in comparison to station center
                # consider additional distance in air & ice
                # assume for air & ice constant index of refraction
                dt = geometryUtilities.get_time_delay_from_direction(
                    curr_fresnel_zenith, azimuth, channel_pos, n=curr_n)
                if channel_pos[2] < -5:
                    logger.warning(
                        "Galactic noise cannot be simulated accurately for deep in-ice channels. "
                        "Coherence and arrival direction of noise are probably inaccurate.")

                delta_phases = -2 * np.pi * freqs[passband_filter] * dt

                # add random polarizations and phase to electric field
                polarizations = self.__random_generator.uniform(0, 2. * np.pi, len(efield_amplitude))

                channel_noise_spec[1][passband_filter] = noise_spectrum[1][passband_filter] * np.exp(
                    1j * delta_phases) * np.cos(polarizations) * curr_t_theta
                channel_noise_spec[2][passband_filter] = noise_spectrum[2][passband_filter] * np.exp(
                    1j * delta_phases) * np.sin(polarizations) * curr_t_phi


                # fold electric field with antenna response
                if self.__caching:
                    antenna_response = self._get_cached_antenna_response(
                        antenna_pattern, curr_fresnel_zenith, azimuth, *antenna_orientation)
                else:
                    antenna_response = antenna_pattern.get_antenna_response_vectorized(
                        freqs, curr_fresnel_zenith, azimuth, *antenna_orientation)

                channel_noise_spectrum = (
                    antenna_response['theta'] * channel_noise_spec[1]
                    + antenna_response['phi'] * channel_noise_spec[2]
                )

                # add noise spectrum from pixel in the sky to channel spectrum
                channel_spectra[channel.get_id()] += channel_noise_spectrum

        # store the updated channel spectra
        for channel in station.iter_channels(use_channels=selected_channel_ids):
            channel.set_frequency_spectrum(channel_spectra[channel.get_id()], "same")


@functools.lru_cache(maxsize=1)
def get_local_coordinates(coordinates, time, n_side):
    """
    Calculates the local coordinates of the pixels of a healpix map given the site coordinates and time.

    Parameters
    ----------
    coordinates: tuple of float
        The latitude and longitude of the site
    time: astropy.time.Time
        The time at which the observation is made (station time)
    n_side: int
        The n_side parameter of the healpix map

    Returns
    -------
    local_coordinates: astropy.coordinates.SkyCoord
        The local coordinates of the pixels of the healpix map
    """
    site_latitude, site_longitude = coordinates
    site_location = astropy.coordinates.EarthLocation(lat=site_latitude * astropy.units.deg,
                                                        lon=site_longitude * astropy.units.deg)

    local_cs = astropy.coordinates.AltAz(location=site_location, obstime=time)

    pixel_longitudes, pixel_latitudes = healpy.pixelfunc.pix2ang(
        n_side, range(healpy.pixelfunc.nside2npix(n_side)), lonlat=True)

    pixel_longitudes *= units.deg
    pixel_latitudes *= units.deg

    galactic_coordinates = astropy.coordinates.Galactic(l=pixel_longitudes * astropy.units.rad,
                                                        b=pixel_latitudes * astropy.units.rad)
    local_coordinates = galactic_coordinates.transform_to(local_cs)

    return local_coordinates
