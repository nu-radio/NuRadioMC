from NuRadioReco.utilities import units, fft, ice, geometryUtilities
from NuRadioReco.modules.base.module import register_run
import NuRadioReco.framework.channel
import NuRadioReco.framework.sim_station
import NuRadioReco.detector.antennapattern
import logging
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
import scipy.interpolate

logger = logging.getLogger('channelGalacticNoiseAdder')

try:
    from radiocalibrationtoolkit import * #Documentation: https://github.com/F-Tomas/radiocalibrationtoolkit/tree/main contains SSM, GMOSS, ULSA
except:
    logger.info("radiocalibrationtoolkit import failed. Consider installing it to use more sky models.")

try:
    from pylfmap import LFmap #Documentation: https://github.com/F-Tomas/pylfmap needs cfitsio installation
except:
    logger.info("LFmap import failed. Consider installing it to use LFmap as sky model.")

from pygdsm import (
    GlobalSkyModel16,
    GlobalSkyModel,
    LowFrequencySkyModel,
    HaslamSkyModel,
)

import healpy
import astropy.coordinates
import astropy.units




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
        self.__debug = None
        self.__zenith_sample = None
        self.__azimuth_sample = None
        self.__n_side = None
        self.__interpolation_frequencies = None
        self.__gdsm = None
        self.__antenna_pattern_provider = NuRadioReco.detector.antennapattern.AntennaPatternProvider()
        #self.begin(skymodel=None)


    def begin(
        self,
        skymodel=None, #choose sky model. Available are lfmap, lfss, gsm2016, haslam, ssm, gmoss, ulsa_fdi, ulsa_dpi, ulsa_ci.
        debug=False,
        n_side=4,
        freq_range = [30,80]
    ):
        """
        Set up important parameters for the module

        Parameters
        ----------
        debug: bool, default: False
            It True, debug plots will be shown
        skymodel: string, optional. Choose from set of sky models
        n_side: int, default: 4
            The n_side parameter of the healpix map. Has to be power of 2
            The radio skymap is downsized to the resolution specified by the n_side
            parameter and for every pixel above the horizon the radio noise coming
            from that direction is calculated. The number of pixels used is
            12 * n_side ** 2, so a larger value for n_side will result better accuracy
            but also greatly increase computing time.
        freq_range: array of len=2
            The sky brightness temperature will be evaluated for the frequencies 
            within this limit. Brightness temperature for frequencies in between are
            calculated by interpolation the log10 of the temperature
            The interpolation_frequencies have to cover the entire passband
            specified in the run method. Frequencies should be in MHz, but without
            units specified.
        """
        
        self.__debug = debug
        self.__n_side = n_side

        #define interpolation frequencies. Set in logarithmic range from 10 to 1010MHz, rounded to 0 decimal places to avoid import errors from LFmap abd tabulated models.
        self.__interpolation_frequencies = np.around(np.logspace(np.log(freq_range[0])/np.log(10), np.log(freq_range[1])/np.log(10), num=30),0) * units.MHz
        
        # initialise sky model
        if skymodel == 'lfmap':
            try:
                sky_model = LFmap()
                print("Using LFmap as sky model")
            except:
                sky_model = GlobalSkyModel(freq_unit="MHz")
                print(f"{skymodel} import not found. Defaulting to GSM2008 as sky model.")

        if skymodel == 'lfss':
            sky_model = LowFrequencySkyModel(freq_unit="MHz")
            print("Using LFSS as sky model")
        if skymodel == 'gsm2008':
            sky_model = GlobalSkyModel(freq_unit="MHz")
            print("Using GSM2008 as sky model")
        if skymodel == 'gsm2016':
            sky_model = GlobalSkyModel16(freq_unit="MHz")
            print("Using GSM2016 as sky model")
        if skymodel == 'haslam':
            sky_model = HaslamSkyModel(freq_unit="MHz", spectral_index=-2.53)
            print("Using Haslam as sky model")

        if skymodel == 'ssm':
            try:
                sky_model = SSM()
                print("Using SSM as sky model")
            except:
                sky_model = GlobalSkyModel(freq_unit="MHz")
                print(f"{skymodel} import not found. Defaulting to GSM2008 as sky model.")
        if skymodel == 'gmoss':
            try:
                sky_model = GMOSS()
                print("Using GMOSS as sky model")
            except:
                sky_model = GlobalSkyModel(freq_unit="MHz")
                print(f"{skymodel} import not found. Defaulting to GSM2008 as sky model.")
        if skymodel == 'ulsa_fdi':
            try:
                sky_model = ULSA(index_type='freq_dependent_index')
                print("Using ULSA_fdi as sky model")
            except:
                sky_model = GlobalSkyModel(freq_unit="MHz")
                print(f"{skymodel} import not found. Defaulting to GSM2008 as sky model.")
        if skymodel == 'ulsa_ci':
            try:
                sky_model = ULSA(index_type='constant_index')
                print("Using ULSA_ci as sky model")
            except:
                sky_model = GlobalSkyModel(freq_unit="MHz")
                print(f"{skymodel} import not found. Defaulting to GSM2008 as sky model.")
        if skymodel == 'ulsa_dpi':
            try:
                sky_model = ULSA(index_type='direction_dependent_index')
                print("Using ULSA_dpi as sky model")
            except:
                sky_model = GlobalSkyModel(freq_unit="MHz")
                print(f"{skymodel} import not found. Defaulting to GSM2008 as sky model.")

        if skymodel is None:
            sky_model = GlobalSkyModel(freq_unit="MHz")
            print("No sky model specified. Using standard: Global Sky Model (2008). Other sky models available: lfmap, lfss, gsm2016, haslam, ssm, gmoss, ulsa_fdi, ulsa_dpi, ulsa_ci. ")
        
        self.__noise_temperatures = np.zeros((len(self.__interpolation_frequencies), healpy.pixelfunc.nside2npix(self.__n_side)))
        print("generating noise temperatures")
        
        # generating sky maps and noise temperatures from chosen sky model in given frequency range
        for i_freq, noise_freq in enumerate(self.__interpolation_frequencies):
            self.__radio_sky = sky_model.generate(noise_freq / units.MHz)
            self.__radio_sky = healpy.pixelfunc.ud_grade(self.__radio_sky, self.__n_side)
            self.__noise_temperatures[i_freq] = self.__radio_sky
            
      
                


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
        passband: list of float
            Lower and upper bound of the frequency range in which noise shall be
            added
        """
        
        
        
        # check that or all channels channel.get_frequencies() is identical
        last_freqs = None
        for channel in station.iter_channels():
            if (not last_freqs is None) and (not np.allclose(last_freqs, channel.get_frequencies(), rtol = 0, atol = 0.1 * units.MHz)):
                logger.error("The frequencies of each channel must be the same, but they are not!")
                return
                
            last_freqs = channel.get_frequencies()
        
        freqs = last_freqs
        d_f = freqs[2] - freqs[1]
        
        
        if passband is None:
            passband = [30 * units.MHz, 80 * units.MHz]
        passband_filter = (freqs > passband[0]) & (freqs < passband[1])
            
                
        site_latitude, site_longitude = detector.get_site_coordinates(station.get_id())
        site_location = astropy.coordinates.EarthLocation(lat=site_latitude * astropy.units.deg, lon=site_longitude * astropy.units.deg)
        station_time = station.get_station_time()

        
        local_cs = astropy.coordinates.AltAz(location=site_location, obstime=station_time)
        solid_angle = healpy.pixelfunc.nside2pixarea(self.__n_side, degrees=False)
        
        pixel_longitudes, pixel_latitudes = healpy.pixelfunc.pix2ang(self.__n_side, range(healpy.pixelfunc.nside2npix(self.__n_side)), lonlat=True)
        pixel_longitudes *= units.deg
        pixel_latitudes *= units.deg
        
        galactic_coordinates = astropy.coordinates.Galactic(l=pixel_longitudes * astropy.units.rad, b=pixel_latitudes * astropy.units.rad)
        local_coordinates = galactic_coordinates.transform_to(local_cs)
        
        
        n_ice = ice.get_refractive_index(-0.01, detector.get_site(station.get_id()))
        n_air = 1.000292 # TODO: This value applies under standard conditions. Does it hold for Greenland?
        
        
        for i_pixel in range(healpy.pixelfunc.nside2npix(self.__n_side)):
            azimuth = local_coordinates[i_pixel].az.rad
            zenith = np.pi / 2. - local_coordinates[i_pixel].alt.rad
            
            if zenith > 90. * units.deg:
                continue
                
            # consider signal reflection at ice surface
            t_theta = geometryUtilities.get_fresnel_t_p(zenith, n_ice, 1)
            t_phi = geometryUtilities.get_fresnel_t_s(zenith, n_ice, 1)
            fresnel_zenith = geometryUtilities.get_fresnel_angle(zenith, n_ice, 1.)

            if fresnel_zenith is None:
                continue
            
            
            temperature_interpolator = scipy.interpolate.interp1d(self.__interpolation_frequencies, np.log10(self.__noise_temperatures[:, i_pixel]), kind='quadratic')
            noise_temperature = np.power(10, temperature_interpolator(freqs[passband_filter]))
                
            
            # calculate spectral radiance of radio signal using rayleigh-jeans law
            S = 2. * (scipy.constants.Boltzmann * units.joule / units.kelvin) * freqs[passband_filter]**2 / (scipy.constants.c * units.m / units.s)**2 * noise_temperature * solid_angle
            S[np.isnan(S)] = 0

            # calculate radiance per energy bin
            S_per_bin = S * d_f

            # calculate electric field per energy bin from the radiance per bin
            E = np.sqrt(S_per_bin / (scipy.constants.c * units.m / units.s * scipy.constants.epsilon_0 * (units.coulomb / units.V / units.m))) / d_f
            
            
            # assign random phases to electric field
            noise_spectrum = np.zeros((3, freqs.shape[0]), dtype=np.complex128)
            phases = np.random.uniform(0, 2. * np.pi, len(S))

            noise_spectrum[1][passband_filter] = np.exp(1j * phases) * E
            noise_spectrum[2][passband_filter] = np.exp(1j * phases) * E
            
            
            channel_noise_spec = np.zeros_like(noise_spectrum)
            
            for channel in station.iter_channels():
                if detector.get_relative_position(station.get_id(), channel.get_id())[2] < 0:
                    curr_t_theta = t_theta
                    curr_t_phi = t_phi
                    curr_fresnel_zenith = fresnel_zenith
                else:
                    curr_t_theta = 1
                    curr_t_phi = 1
                    curr_fresnel_zenith = zenith
                
                
                antenna_pattern = self.__antenna_pattern_provider.load_antenna_pattern(detector.get_antenna_model(station.get_id(), channel.get_id()))
                antenna_orientation = detector.get_antenna_orientation(station.get_id(), channel.get_id())
                
                # calculate the phase offset in comparison to station center
                # consider additional distance in air & ice
                # assume for air & ice constant index of refraction
                channel_pos_x, channel_pos_y = detector.get_relative_position(station.get_id(), channel.get_id())[[0, 1]]
                channel_depth = abs(min(detector.get_relative_position(station.get_id(), channel.get_id())[2], 0))
                sin_zenith = np.sin(zenith)
                delta_phases = 2 * np.pi * freqs[passband_filter] / (scipy.constants.c * units.m / units.s) * n_air * ( sin_zenith * (np.cos(azimuth) * channel_pos_x + channel_pos_y * np.sin(azimuth)) + channel_depth * ((n_ice / n_air)**2 + sin_zenith**2) / np.sqrt((n_ice / n_air)**2 - sin_zenith**2) )
                
                
                # add random polarizations and phase to electric field
                polarizations = np.random.uniform(0, 2. * np.pi, len(S))
                
                channel_noise_spec[1][passband_filter] = noise_spectrum[1][passband_filter] * np.exp(1j * delta_phases) * np.cos(polarizations)
                channel_noise_spec[2][passband_filter] = noise_spectrum[2][passband_filter] * np.exp(1j * delta_phases) * np.sin(polarizations)
                
                
                # fold electric field with antenna response
                antenna_response = antenna_pattern.get_antenna_response_vectorized(freqs, curr_fresnel_zenith, azimuth, *antenna_orientation)
                channel_noise_spectrum = antenna_response['theta'] * channel_noise_spec[1] * curr_t_theta + antenna_response['phi'] * channel_noise_spec[2] * curr_t_phi
                
                
                # add noise spectrum to channel freq spectrum
                channel_spectrum = channel.get_frequency_spectrum()
                channel_spectrum += channel_noise_spectrum
                channel.set_frequency_spectrum(channel_spectrum, channel.get_sampling_rate())
        
        
        return
