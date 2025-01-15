from NuRadioReco.utilities import units, ice, geometryUtilities
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

import healpy
import astropy.coordinates
import astropy.units

logger = logging.getLogger('NuRadioReco.channelThermalNoiseAdder')


class channelThermalNoiseAdder:
    """

    """

    def __init__(self):
        self.__n_side = None
        self.__antenna_pattern_provider = NuRadioReco.detector.antennapattern.AntennaPatternProvider()

    def begin(self, n_side=4, noise_temperature=300):
        """
        Set up important parameters for the module

        Parameters
        ----------
        n_side: int, default: 4
            The n_side parameter of the healpix map. Has to be power of 2
            The radio skymap is downsized to the resolution specified by the n_side
            parameter and for every pixel above the horizon the radio noise coming
            from that direction is calculated. The number of pixels used is
            12 * n_side ** 2, so a larger value for n_side will result better accuracy
            but also greatly increase computing time.
        noise_temperature: float, default: 300
            The noise temperature of the ambient ice in Kelvin.
        """
        self.__n_side = n_side
        self.noise_temperature = noise_temperature
        self.solid_angle = healpy.pixelfunc.nside2pixarea(self.__n_side, degrees=False)

    @register_run()
    def run(
            self,
            event,
            station,
            detector,
            passband=None
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
        """

        # check that or all channels channel.get_frequencies() is identical
        last_freqs = None
        for channel in station.iter_channels():
            if last_freqs is not None and (
                    not np.allclose(last_freqs, channel.get_frequencies(), rtol=0, atol=0.1 * units.MHz)):
                logger.error("The frequencies of each channel must be the same, but they are not!")
                return

            last_freqs = channel.get_frequencies()

        freqs = last_freqs
        d_f = freqs[2] - freqs[1]

        if passband is None:
            passband = [10 * units.MHz, 1000 * units.MHz]

        passband_filter = (freqs > passband[0]) & (freqs < passband[1])

        site_latitude, site_longitude = detector.get_site_coordinates(station.get_id())
        station_time = station.get_station_time()

        local_coordinates = get_local_coordinates((site_latitude, site_longitude), station_time, self.__n_side)

        n_ice = ice.get_refractive_index(-0.01, detector.get_site(station.get_id()))
        n_air = ice.get_refractive_index(depth=1, site=detector.get_site(station.get_id()))
        c_vac = scipy.constants.c * units.m / units.s

        channel_spectra = {}
        for channel in station.iter_channels():
            channel_spectra[channel.get_id()] = channel.get_frequency_spectrum()

        for i_pixel in range(healpy.pixelfunc.nside2npix(self.__n_side)):
            azimuth = local_coordinates[i_pixel].az.rad
            zenith = np.pi / 2. - local_coordinates[i_pixel].alt.rad # this is the in-air zenith

            # calculate spectral radiance of radio signal using rayleigh-jeans law
            spectral_radiance = (2. * (scipy.constants.Boltzmann * units.joule / units.kelvin)
                * freqs[passband_filter] ** 2 * self.noise_temperature * self.solid_angle / c_vac ** 2)
            spectral_radiance[np.isnan(spectral_radiance)] = 0

            # calculate radiance per energy bin
            spectral_radiance_per_bin = spectral_radiance * d_f

            # calculate electric field per frequency bin from the radiance per bin
            efield_amplitude = np.sqrt(
                spectral_radiance_per_bin / (c_vac * scipy.constants.epsilon_0 * (
                        units.coulomb / units.V / units.m))) / d_f

            # assign random phases to electric field
            noise_spectrum = np.zeros((3, freqs.shape[0]), dtype=complex)
            phases = np.random.uniform(0, 2. * np.pi, len(spectral_radiance))

            noise_spectrum[1][passband_filter] = np.exp(1j * phases) * efield_amplitude
            noise_spectrum[2][passband_filter] = np.exp(1j * phases) * efield_amplitude

            channel_noise_spec = np.zeros_like(noise_spectrum)

            for channel in station.iter_channels():
                channel_pos = detector.get_relative_position(station.get_id(), channel.get_id())

                antenna_pattern = self.__antenna_pattern_provider.load_antenna_pattern(
                    detector.get_antenna_model(station.get_id(), channel.get_id()))
                antenna_orientation = detector.get_antenna_orientation(station.get_id(), channel.get_id())

                # add random polarizations and phase to electric field
                polarizations = np.random.uniform(0, 2. * np.pi, len(spectral_radiance))

                channel_noise_spec[1][passband_filter] = noise_spectrum[1][passband_filter] * np.cos(polarizations)
                channel_noise_spec[2][passband_filter] = noise_spectrum[2][passband_filter] * np.sin(polarizations)

                # fold electric field with antenna response
                antenna_response = antenna_pattern.get_antenna_response_vectorized(freqs, zenith, azimuth,
                                                                                   *antenna_orientation)
                channel_noise_spectrum = (
                    antenna_response['theta'] * channel_noise_spec[1]
                    + antenna_response['phi'] * channel_noise_spec[2]
                )

                # add noise spectrum from pixel in the sky to channel spectrum
                channel_spectra[channel.get_id()] += channel_noise_spectrum

        # store the updated channel spectra
        for channel in station.iter_channels():
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
