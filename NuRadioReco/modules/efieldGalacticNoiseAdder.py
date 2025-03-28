from NuRadioReco.utilities import units, ice, geometryUtilities, signal_processing
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.modules.channelGalacticNoiseAdder import channelGalacticNoiseAdder, get_local_coordinates
import numpy as np
import scipy.interpolate
import logging
import healpy

logger = logging.getLogger('NuRadioReco.efieldGalacticNoiseAdder')

class efieldGalacticNoiseAdder(channelGalacticNoiseAdder):
    """
    Class that simulates the noise produced by galactic radio emission

    Uses the pydgsm package (https://github.com/telegraphic/pygdsm), which provides
    radio background data based on Oliveira-Costa et al. (2008) (https://arxiv.org/abs/0802.1525)
    and Zheng et al. (2016) (https://arxiv.org/abs/1605.04920)

    The radio sky model is evaluated on a number of points above the horizon
    folded with the antenna response. Since evaluating every frequency individually
    would be too slow, the model is evaluated for a few frequencies and the log10
    of the brightness temperature is interpolated in between.

    This module is largely equivalent to the channelGalacticNoiseAdder, but operates on ``ElectricField``
    instead of ``Channel`` objects.

    See Also
    --------
    NuRadioReco.modules.channelGalacticNoiseAdder.channelGalacticNoiseAdder
        Similar class that directly adds the noise to ``Channel`` (voltage trace) objects
        instead of ``ElectricField``\s.
    """

    def __init__(self):
        super().__init__()

    @register_run()
    def run(
            self,
            event,
            station,
            detector,
            passband=None,
    ):

        """
        Adds noise resulting from galactic radio emission to the field traces

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
        if self.__noise_temperatures is None: # check if .begin has been called, give helpful error message if not
            msg = "efieldGalacticNoiseAdder was not initialized correctly. Maybe you forgot to call `.begin()`?"
            logger.error(msg)
            raise ValueError(msg)

        # check that if all channels field.get_frequencies() are identical
        last_freqs = None
        for field in station.get_electric_fields():
            if (not last_freqs is None) and (
                    not np.allclose(last_freqs, field.get_frequencies(), rtol=0, atol=0.1 * units.MHz)):
                logger.error("The frequencies of each field must be the same, but they are not!")
                return
            last_freqs = field.get_frequencies()

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

        field_spectra = {}
        for field in station.get_electric_fields():
            field_spectra[field.get_unique_identifier()] = field.get_frequency_spectrum()

        for i_pixel in range(healpy.pixelfunc.nside2npix(self.__n_side)):
            azimuth = local_coordinates[i_pixel].az.rad
            zenith = np.pi / 2. - local_coordinates[i_pixel].alt.rad

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

            efield_noise_spec = np.zeros_like(noise_spectrum)

            for field in station.get_electric_fields():

                field_pos = field.get_position()
                if field_pos[2] < 0:
                    curr_t_theta = t_theta
                    curr_t_phi = t_phi
                    curr_fresnel_zenith = fresnel_zenith
                    curr_n = n_ice
                else: # we are in air
                    curr_t_theta = 1
                    curr_t_phi = 1
                    curr_fresnel_zenith = zenith
                    curr_n = n_air

                # calculate the phase offset in comparison to station center
                # consider additional distance in air & ice
                # assume for air & ice constant index of refraction
                dt = geometryUtilities.get_time_delay_from_direction(
                    curr_fresnel_zenith, azimuth, field_pos, n=curr_n)
                if field_pos[2] < -5:
                    logger.warning(
                        "Galactic noise cannot be simulated accurately for deep in-ice channels. "
                        "Coherence and arrival direction of noise are probably inaccurate.")

                delta_phases = -2 * np.pi * freqs[passband_filter] * dt

                # add random polarizations and phase to electric field
                polarizations = self.__random_generator.uniform(0, 2. * np.pi, len(efield_amplitude))

                efield_noise_spec[1][passband_filter] = noise_spectrum[1][passband_filter] * np.exp(
                    1j * delta_phases) * np.cos(polarizations) * curr_t_theta
                efield_noise_spec[2][passband_filter] = noise_spectrum[2][passband_filter] * np.exp(
                    1j * delta_phases) * np.sin(polarizations) * curr_t_phi

                # add noise spectrum from this to field freq spectrum (which is in on-sky CS -> no noise in R-direction)
                field_spectra[field.get_unique_identifier()] += efield_noise_spec

        # Store updated fields spectra
        for field in station.get_electric_fields():
            field.set_frequency_spectrum(field_spectra[field.get_unique_identifier()], "same")
