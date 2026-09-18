from NuRadioReco.utilities import units, ice, geometryUtilities, signal_processing, fft
from NuRadioReco.modules.base.module import register_run

import NuRadioReco.framework.channel
import NuRadioReco.framework.sim_station
import NuRadioReco.detector.antennapattern

from NuRadioMC.utilities import medium, attenuation

import warnings
import functools
import numpy as np
import scipy.constants
import scipy.interpolate
from contextlib import redirect_stdout
from numpy.random import Generator, Philox

import healpy
import astropy.coordinates
import astropy.units

import logging
logger = logging.getLogger('NuRadioReco.channelGalacticNoiseAdder')

# This is the maximum number caching entries for the vector effective length and
# noise temperature.
maxsize = 1024

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

    Notes
    -----
    For an accurate simulation of the galactic noise in deep in-ice antennas, you need to provide
    an ice (describing the refractive index) and (ice) attenuation model. The attenuation is calculated
    along a straight line path, which is obviously incorrect, but for antenna depths of maximum 100 to
    200 meters, the error should be acceptable. Beyond that, the coherence in the noise from one patch
    in the sky between different antennas is still not correctly simulated, as the correct calculation
    of the phase shift would require ray tracing. However, keep in mind that the expected coherence from
    the galactic noise is expected to be very low and will not be detectable without a large number of
    antennas.
    """

    def __init__(self):
        self.__n_side = None
        self.__interpolation_frequencies = None
        self.__radio_sky = None
        self.__noise_temperatures = None
        self.__you_have_been_warned = False
        self.__antenna_pattern_provider = NuRadioReco.detector.antennapattern.AntennaPatternProvider()

    def begin(
            self,
            skymodel=None,
            debug=False,
            n_side=4,
            freq_range=None,
            interpolation_frequencies=None,
            seed=None,
            caching=True,
            scaling=1.0,
            ice_model=None,
            attenuation_model=None,
            simulate_sun=False,
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
        scaling: float, default: 1.0
            Scaling factor for the noise. This is useful when doing interferometry with extremely large arrays
            such as SKA-low. For such an array it is very expensive to simulate/interpolate/process all antennas.
            Instead, one can use every nth antenna and scale the noise by a factor of 1/sqrt(n) (since the SNR
            is expected to scale with the square root of the number of antennas when using interferomtery/beamforming).
        ice_model: str, default: None
            The ice model to use for the simulation of in-ice antennas.
        attenuation_model: str, default: None
            The attenuation model to use for the simulation of in-ice antennas.
        simulate_sun: bool, default: False
            If True, the sun's contribution to the galactic noise is simulated. This is experimental!
        """
        if debug:
            warnings.warn("This argument is deprecated and will be removed in future versions.", DeprecationWarning)

        self.__random_generator = Generator(Philox(seed))
        self.__n_side = n_side
        self.solid_angle = healpy.pixelfunc.nside2pixarea(self.__n_side, degrees=False)

        self._simulate_sun = simulate_sun
        if self._simulate_sun:
            logger.warning("You are simulating the sun's contribution to the galactic noise. "
                "This is experimental and might not be very accurate! See in-code comments for details.")

        self.__caching = caching
        self.scaling = scaling
        self.__freqs = None
        if self.__caching and 12 * n_side ** 2 * 2 > maxsize:
            logger.warning(
                f"Caching for the vector effective length is enabled (with `maxsize={maxsize}`) and `n_side={n_side}` is to large, and thus "
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
        logger.info("Generating noise temperatures ..")

        # generating sky maps and noise temperatures from chosen sky model in given frequency range
        for i_freq, noise_freq in enumerate(self.__interpolation_frequencies):
            self.__radio_sky = sky_model.generate(noise_freq / units.MHz)
            self.__radio_sky = healpy.pixelfunc.ud_grade(self.__radio_sky, self.__n_side)
            self.__noise_temperatures[i_freq] = self.__radio_sky

        # We can not already interpolate the efield amplitudes because for their normalization the
        # frequency resoltion matters.
        self.__noise_temperature_funcs = np.array([
            scipy.interpolate.interp1d(self.__interpolation_frequencies, np.log10(self.__noise_temperatures[:, i_pixel]), kind='quadratic')
            for i_pixel in range(healpy.pixelfunc.nside2npix(self.__n_side))
        ])

        self._ice = medium.get_ice_model(ice_model) if ice_model is not None else None
        self._attenuation_model = attenuation_model

    @functools.lru_cache(maxsize=maxsize)
    def _get_cached_antenna_response(self, ant_pattern, zen, azi, *ant_orient):
        """
        Returns the cached antenna reponse for a given antenna patter, antenna orientation
        and signal arrival direction. This wrapper is necessary as arrays and list are not
        hashable (i.e., can not be used as arguments in functions one wants to cache).
        This module ensures that the cache is clearied if the vector `self.__freqs` changes.
        """
        return ant_pattern.get_antenna_response_vectorized(self.__freqs, zen, azi, *ant_orient)

    @functools.lru_cache(maxsize=maxsize)
    def _get_cached_noise_temperature_for_pixel(self, i_pixel):
        """
        Returns the cached electric field amplitude for a given pixel.
        This wrapper is necessary as arrays and list are not
        hashable (i.e., can not be used as arguments in functions one wants to cache).
        This module ensures that the cache is clearied if the vector `self.__freqs` changes.
        """
        return np.power(10, self.__noise_temperature_funcs[i_pixel](self.__freqs))

    def _check_cache(self, freqs):
        # If we cache the antenna pattern / sky noise temperature, we need to make sure that the frequencies have not changed
        # between stations. If they have, we need to clear the cache.
        if self.__caching:
            if self.__freqs is None:
                self.__freqs = freqs
            else:
                if len(self.__freqs) != len(freqs):
                    self.__freqs = freqs
                    self._get_cached_antenna_response.cache_clear()
                    self._get_cached_noise_temperature_for_pixel.cache_clear()
                    logger.warning(
                        "Frequencies have changed (array length). Clearing antenna response / efield cache. "
                        "(If this happens often, something might be wrong...")
                elif not np.allclose(self.__freqs, freqs, rtol=0, atol=0.01 * units.MHz):
                    self.__freqs = freqs
                    self._get_cached_antenna_response.cache_clear()
                    self._get_cached_noise_temperature_for_pixel.cache_clear()
                    logger.warning(
                        "Frequencies have changed (values). Clearing antenna response / efield cache. "
                        "(If this happens often, something might be wrong...")


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

        if passband is None:
            passband = [10 * units.MHz, 1000 * units.MHz]

        passband_filter = (freqs > passband[0]) & (freqs < passband[1])

        self._check_cache(freqs[passband_filter])

        site_latitude, site_longitude = detector.get_site_coordinates(station.get_id())
        station_time = station.get_station_time()
        local_coordinates = get_local_coordinates((site_latitude, site_longitude), station_time, self.__n_side)


        if self._simulate_sun:
            # If we want to simulate the sun, we need to calculate its position in the sky and
            # the corresponding pixel in the healpix map.
            site_location = astropy.coordinates.EarthLocation(
                lat=site_latitude * astropy.units.deg, lon=site_longitude * astropy.units.deg)
            local_cs = astropy.coordinates.AltAz(
                obstime=station_time, location=site_location)
            sun_coord = astropy.coordinates.get_sun(station_time).transform_to(local_cs)

            sun_zen = 90. * astropy.units.deg - sun_coord.alt
            sun_azi = sun_coord.az
            sun_pixel = healpy.pixelfunc.ang2pix(self.__n_side, sun_zen.rad, sun_azi.rad)

            # Angular radius of the (optical) sun as seen from the earth.
            # The actual size of the sun in radio can be different and
            # frequency dependent, but this is a good approximation for now.
            theta = (astropy.constants.R_sun / sun_coord.distance).decompose()
            sun_omega = np.pi * theta**2  # small angle approximation
            logger.debug(
                f"Simulating the sun's contribution to the galactic noise. "
                f"Sun pixel: {sun_pixel}, Sun zenith: {sun_zen}, Sun azimuth: {sun_azi})"
            )

        if self._ice is not None:
            n_ice_surf = self._ice.get_index_of_refraction(np.array([0, 0, -0.01]))
        else:
            # This can return n_ice_surf = n_air for sites that are not on the ice (auger, lofar, ska)
            n_ice_surf = ice.get_refractive_index(-0.01, detector.get_site(station.get_id()))

        # This is actually better than using the ice model for the refractive index since it returns 1 for
        # in air refractive index. However, this function does currently not return a site/altitude dependent refractive index ...
        n_air = ice.get_refractive_index(depth=1, site=detector.get_site(station.get_id()))

        channel_spectra = {}
        channel_depths = []
        one_over_average_attenuation_length = {}
        for channel in station.iter_channels(use_channels=selected_channel_ids):
            channel_spectra[channel.get_id()] = channel.get_frequency_spectrum()
            channel_depth = detector.get_relative_position(station.get_id(), channel.get_id())[2]
            channel_depths.append(channel_depth)

            # Calculate signal attenuation for in-ice channels. This is a (good) approximation,
            # assumes a straight line path and not a bend trajectory.
            if self._attenuation_model is not None and channel_depth < -10 * units.meter:
                depth_bins = np.linspace(0, channel_depth)
                depths = depth_bins[:-1] + np.diff(depth_bins) / 2
                one_over_average_attenuation_length[channel.get_id()] = np.mean(
                    [1 / attenuation.get_attenuation_length(d, freqs[passband_filter], self._attenuation_model) for d in depths],
                    axis=0,
                )

        if self._ice is None or self._attenuation_model is None:
            if np.min(channel_depths) < -20 and not self.__you_have_been_warned:
                logger.warning(
                    "You are simulating deep in-ice channels (below 20m) but have not provided an ice or attenuation model. "
                    "The refractive index is thus the same for the calculation of the Fresnel coefficients "
                    "and the phase shift as for the ice_surface, which is not correct. This warning is only printed once!"
                )
                self.__you_have_been_warned = True

        any_in_ice_channel = np.any(np.array(channel_depths) < 0)
        for i_pixel in range(healpy.pixelfunc.nside2npix(self.__n_side)):
            azimuth = local_coordinates[i_pixel].az.rad
            zenith = np.pi / 2. - local_coordinates[i_pixel].alt.rad # this is the in-air zenith

            if zenith > 90. * units.deg:
                continue

            if any_in_ice_channel:
                # For sites such as lofar or SKA, even if a channel has a negative z-coordinate
                # and any_in_ice_channel is technically true, the resulting t_theta and t_phi will be 1,
                # and surf_zenith=zenith, because n_air == n_ice_surf.
                t_theta = geometryUtilities.get_fresnel_t_p(zenith, n_ice_surf, n_air)
                t_phi = geometryUtilities.get_fresnel_t_s(zenith, n_ice_surf, n_air)
                surf_zenith = geometryUtilities.get_fresnel_angle(zenith, n_ice_surf, n_air)

            if self.__caching:
                noise_temperature = self._get_cached_noise_temperature_for_pixel(i_pixel)
            else:
                noise_temperature = np.power(10, self.__noise_temperature_funcs[i_pixel](freqs[passband_filter]))

            if self._simulate_sun and i_pixel == sun_pixel:
                noise_temperature += sun_omega / self.solid_angle * Tb_quiet_sun(freqs[passband_filter])

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
                    if self._ice is not None:
                        curr_n = self._ice.get_index_of_refraction(channel_pos)
                        curr_fresnel_zenith = geometryUtilities.get_fresnel_angle(zenith, curr_n, n_air)
                    else:
                        curr_fresnel_zenith = surf_zenith
                        curr_n = n_ice_surf

                    # The (de)focusing due to air-ice boundary is already included in the Fresnel coefficients (t_theta/phi),
                    # here we account for the additional (de)focusing from the gradually changing refractive index in the firm.
                    focusing_factor = np.sqrt((n_ice_surf * np.cos(surf_zenith)) / (curr_n * np.cos(curr_fresnel_zenith)))
                else: # we are in air
                    curr_t_theta = 1
                    curr_t_phi = 1
                    curr_fresnel_zenith = zenith
                    curr_n = n_air
                    focusing_factor = 1

                antenna_pattern = self.__antenna_pattern_provider.load_antenna_pattern(
                    detector.get_antenna_model(station.get_id(), channel.get_id()))
                antenna_orientation = detector.get_antenna_orientation(station.get_id(), channel.get_id())

                # Calculate the phase offset in comparison to station center
                # consider additional distance in air & ice. Assume for air & ice
                # constant index of refraction. This is incorrect in particular
                # for the ice
                dt = geometryUtilities.get_time_delay_from_direction(
                    curr_fresnel_zenith, azimuth, channel_pos, n=curr_n)
                delta_phases = -2 * np.pi * freqs[passband_filter] * dt

                if self._attenuation_model is not None and channel_pos[2] < -10:
                    attenuation_factor = np.exp(
                        -(abs(channel_pos[2]) / np.cos(curr_fresnel_zenith)) *
                        one_over_average_attenuation_length[channel.get_id()])
                else:
                    attenuation_factor = np.ones_like(efield_amplitude)


                # add random polarizations and phase to electric field
                polarizations = self.__random_generator.uniform(0, 2. * np.pi, len(efield_amplitude))

                channel_noise_spec[1][passband_filter] = noise_spectrum[1][passband_filter] * np.exp(
                    1j * delta_phases) * np.cos(polarizations) * curr_t_theta * attenuation_factor * focusing_factor
                channel_noise_spec[2][passband_filter] = noise_spectrum[2][passband_filter] * np.exp(
                    1j * delta_phases) * np.sin(polarizations) * curr_t_phi * attenuation_factor * focusing_factor

                # fold electric field with antenna response
                if self.__caching:
                    antenna_response = self._get_cached_antenna_response(
                        antenna_pattern, curr_fresnel_zenith, azimuth, *antenna_orientation)
                else:
                    antenna_response = antenna_pattern.get_antenna_response_vectorized(
                        freqs[passband_filter], curr_fresnel_zenith, azimuth, *antenna_orientation)

                channel_noise_spectrum = (
                    antenna_response['theta'] * channel_noise_spec[1][passband_filter]
                    + antenna_response['phi'] * channel_noise_spec[2][passband_filter]
                )

                # scale noise spectrum:
                channel_noise_spectrum *= self.scaling

                # add noise spectrum from pixel in the sky to channel spectrum
                channel_spectra[channel.get_id()][passband_filter] += channel_noise_spectrum

        # store the updated channel spectra
        for channel in station.iter_channels(use_channels=selected_channel_ids):
            channel.set_frequency_spectrum(channel_spectra[channel.get_id()], "same")

    def get_electric_field_strength(
            self, location, obs_time, n_samples, sampling_rate, bandpass=None):
        """
        Returns the electric field strength at a given location and time

        Parameters
        ----------
        location: tuple of floats
            The latitude and longitude in deg.
        obs_time: astropy.time.Time
            The time at which the electric field strength is calculated
        n_samples: int
            The number of samples in the time domain
        sampling_rate: float
            The sampling rate of the trace
        bandpass: list of floats, optional
            The lower and upper bound of the frequency range in which the electric field strength
            shall be calculated. By default no bandpass is applied (frequency range is from
            0 to sampling_rate / 2)

        Returns
        -------
        electric_field_strength: float
            The electric field strength at the given location and time
        """

        local_coordinates = get_local_coordinates(location, obs_time, self.__n_side)

        if bandpass is None:
            bandpass = [10 * units.MHz, sampling_rate / 2]

        freqs = fft.freqs(n_samples, sampling_rate)
        spectrum = np.zeros_like(freqs, dtype=complex)

        window = np.zeros_like(freqs, dtype=bool)
        window[np.logical_and(bandpass[0] < freqs, freqs < bandpass[1])] = True

        self._check_cache(freqs[window])

        for i_pixel in range(healpy.pixelfunc.nside2npix(self.__n_side)):
            zenith = np.pi / 2. - local_coordinates[i_pixel].alt.rad # this is the in-air zenith

            if zenith > 90. * units.deg:
                continue

            if self.__caching:
                noise_temperature = self._get_cached_noise_temperature_for_pixel(i_pixel)
            else:
                noise_temperature = np.power(10, self.__noise_temperature_funcs[i_pixel](freqs[window]))

            efield_amplitude = signal_processing.get_electric_field_from_temperature(
                freqs[window], noise_temperature, self.solid_angle)


            phases = self.__random_generator.uniform(0, 2. * np.pi, len(efield_amplitude))
            spectrum_pixel = np.exp(1j * phases) * efield_amplitude
            spectrum[window] += spectrum_pixel

        return np.std(fft.freq2time(spectrum, sampling_rate))


@functools.lru_cache(maxsize=1)
def get_local_coordinates(coordinates, obs_time, n_side):
    """
    Calculates the local coordinates of the pixels of a healpix map given the site coordinates and time.

    Parameters
    ----------
    coordinates: tuple of float
        The latitude and longitude of the site
    obs_time: astropy.time.Time
        The time at which the observation is made (station time)
    n_side: int
        The n_side parameter of the healpix map

    Returns
    -------
    local_coordinates: astropy.coordinates.SkyCoord
        The local coordinates of the pixels of the healpix map
    """
    site_latitude, site_longitude = coordinates
    site_location = astropy.coordinates.EarthLocation(
        lat=site_latitude * astropy.units.deg, lon=site_longitude * astropy.units.deg)

    local_cs = astropy.coordinates.AltAz(location=site_location, obstime=obs_time)

    # because `lonlat=True` function returns angles in degrees
    pixel_longitudes, pixel_latitudes = healpy.pixelfunc.pix2ang(
        n_side, range(healpy.pixelfunc.nside2npix(n_side)), lonlat=True)

    # First convert deg to rad using the NuRadio unit system
    # Than convert them to astropy.Quantities to be used with the
    # astropy class.
    pixel_longitudes = pixel_longitudes * units.deg * astropy.units.rad
    pixel_latitudes = pixel_latitudes * units.deg * astropy.units.rad

    galactic_coordinates = astropy.coordinates.Galactic(
        l=pixel_longitudes, b=pixel_latitudes)

    local_coordinates = galactic_coordinates.transform_to(local_cs)
    return local_coordinates


def Tb_quiet_sun(nu, kind='linear'):
    """ Returns the apparent brigthness temperature for the quite sun for a given frequency

    Returned values are interpolated from digitized data from Fig. 5, of the paper:
    https://iopscience.iop.org/article/10.3847/1538-4357/ac6b37/pdf

    Parameters
    ----------
    nu : float or array of floats
        Frequency
    kind : str (default: 'linear')
        Interpolation method used with ``scipy.interpolate.interp1d``

    Returns
    -------
    tb : float or array of floats
        Sun's apparent brightness temperature in K
    """
    f = _Tb_quiet_sun(kind=kind)
    return np.power(10, f(np.log10(nu)))

@functools.lru_cache(maxsize=2)
def _Tb_quiet_sun(kind='linear'):
    """ Returns function of Tb(f) """

    # First column (i.e., entires data[::2]) is the frequency in MHz,
    # second column (i.e., entries data[1::2]) is the brightness temperature in K.
    data = np.array([
        15.060724325205065, 122701.78220087259,
        15.876542140124316, 146675.73927896278,
        16.84673686448343, 183335.7686269777,
        18.115666561211253, 220786.08426477874,
        20.135947446647886, 269857.8130123856,
        22.981940237261043, 337266.16604412446,
        33.512127308790284, 474607.88378179277,
        40.60340447826144, 550580.2566932675,
        48.872414216886185, 619993.1163195028,
        80.84239412333073, 762960.8522460236,
        94.1472794671951, 785813.3991897015,
        109.64068351827044, 815394.2231007224,
        142.90957239126385, 839664.4631625947,
        180.2185844542181, 814739.8494407722,
        227.28965486883104, 739354.8333126334,
        308.3786113174188, 600026.3965025863,
        412.9377214863189, 448699.73837802495,
        549.3674814026401, 306883.5587758216,
        683.9762948907373, 217867.7976521072,
        880.338134652793, 144647.1870885708,
        1074.4360119917706, 105011.39714303697,
        1373.746716952524, 70240.73705220643,
        1827.6947107331177, 46631.862791560954,
        4721.350657521103, 16431.3945572252,
        7712.462272772942, 12375.17761046132,
        19124.788361772276, 9595.324665349062,
    ])
    freq = data[0::2] * units.MHz
    Tb = data[1::2]
    f = scipy.interpolate.interp1d(np.log10(freq), np.log10(Tb), kind=kind, fill_value="extrapolate")

    return f
