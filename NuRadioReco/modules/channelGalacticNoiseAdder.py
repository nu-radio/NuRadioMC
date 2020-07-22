from NuRadioReco.utilities import units, fft, trace_utilities
from NuRadioReco.modules.base.module import register_run
import NuRadioReco.framework.channel
import NuRadioReco.detector.antennapattern
import logging
import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.constants
import scipy.interpolate
import pygsm
import healpy

logger = logging.getLogger('channelGalacticNoiseAdder')

class channelGalacticNoiseAdder:
    def __init__(self):
        self.begin()

    def begin(self, debug=False, n_zenith=18, n_azimuth=36, interpolation_frequencies=np.arange(10, 1100, 100)*units.MHz):
        self.__debug = debug
        self.__zenith_sample = np.linspace(0, 90, n_zenith)[1:] * units.deg
        self.__azimuth_sample = np.linspace(0, 360, n_azimuth)[1:] * units.deg
        self.__delta_zenith = self.__zenith_sample[1] - self.__zenith_sample[0]
        self.__delta_azimuth = self.__azimuth_sample[1] - self.__azimuth_sample[0]
        self.__antenna_pattern_provider = NuRadioReco.detector.antennapattern.AntennaPatternProvider()
        self.__interpolaiton_frequencies = interpolation_frequencies
    def run(self, event, station, detector, passband=[10*units.MHz, 1000*units.MHz]):
        self.__sky_observer = pygsm.GSMObserver()
        site_latitude, site_longitude = detector.get_site_coordinates(station.get_id())
        self.__sky_observer.longitude = site_longitude
        self.__sky_observer.latitude = site_latitude
        station_time = station.get_station_time()
        station_time.format = 'iso'
        self.__sky_observer.date = station_time.value
        noise_temperatures = np.zeros((len(self.__interpolaiton_frequencies), len(self.__zenith_sample), len(self.__azimuth_sample)))
        for i_freq, noise_freq in enumerate(self.__interpolaiton_frequencies):
            radio_sky = self.__sky_observer.generate(noise_freq/units.MHz)
            healpy.orthview(radio_sky, half_sky=True, title='Brightness temperature at f={:.0f}MHz'.format(noise_freq/units.MHz))
            import pylab
            #pylab.show()
            radio_sky[np.isnan(radio_sky)] = 0
            for i_zenith, zenith in enumerate(self.__zenith_sample):
                elevation = 90.*units.deg - zenith
                for i_azimuth, azimuth in enumerate(self.__azimuth_sample):
                    i_pix = healpy.pixelfunc.ang2pix(self.__sky_observer._n_side, elevation, azimuth, lonlat=True)
                    noise_temperatures[i_freq,i_zenith, i_azimuth] = radio_sky[i_pix]


        for channel in station.iter_channels():
            freqs = channel.get_frequencies()
            times = channel.get_times()
            d_f = freqs[2] - freqs[1]
            sampling_rate = channel.get_sampling_rate()
            channel_spectrum = channel.get_frequency_spectrum()
            passband_filter = (freqs>passband[0])&(freqs<passband[1])
            self.__solid_angle_sum = 0
            plt.close('all')
            fig = plt.figure(figsize=(12,8))
            ax_1 = fig.add_subplot(221)
            ax_2 = fig.add_subplot(222)
            ax_3 = fig.add_subplot(223)
            ax_4 = fig.add_subplot(224)
            ax_1.grid()
            ax_1.set_yscale('log')
            ax_2.grid()
            ax_2.set_yscale('log')
            ax_3.grid()
            ax_3.set_yscale('log')
            ax_4.grid()
            ax_4.set_yscale('log')
            ax_1.set_xlabel('f [MHz]')
            ax_2.set_xlabel('f [MHz]')
            ax_3.set_xlabel('f [MHz]')
            ax_4.set_xlabel('f [MHz]')
            ax_1.set_ylabel('T [K]')
            ax_2.set_ylabel('S [W/m²/MHz]')
            ax_3.set_ylabel('E [V/m]')
            ax_4.set_ylabel('U [V]')
            ax_4.plot(channel.get_frequencies()/units.MHz, np.abs(channel.get_frequency_spectrum()), c='C0')
            ax_4.set_ylim([1.e-8, None])
            noise_spec_sum = np.zeros_like(channel.get_frequency_spectrum())
            flux_sum = np.zeros(freqs[passband_filter].shape)
            efield_sum = np.zeros((3, freqs.shape[0]), dtype=np.complex)
            if self.__debug:
                fig1 = plt.figure()
                ax1 = fig1.add_subplot(211)
                ax2 = fig1.add_subplot(212)
                ax1.plot(channel.get_times(), channel.get_trace(), label='original trace')
                ax2.plot(channel.get_frequencies()/units.MHz, np.abs(channel.get_frequency_spectrum()), label='original spectrum')
                ax1.grid()
                ax2.grid()
            for i_zenith, zenith in enumerate(self.__zenith_sample):
                solid_angle = np.sin(zenith) * self.__delta_zenith * self.__delta_azimuth
                for i_azimuth, azimuth in enumerate(self.__azimuth_sample):
                    temperature_interpolator = scipy.interpolate.interp1d(self.__interpolaiton_frequencies, np.log10(noise_temperatures[:,i_zenith,i_azimuth]), kind='quadratic')
                    noise_temperature = np.power(10, temperature_interpolator(freqs[passband_filter]))
                    S = (2. * scipy.constants.Boltzmann * (freqs[passband_filter]/units.Hz)**2 / scipy.constants.c**2 * (noise_temperature/units.kelvin) * solid_angle) * (units.watt/units.m**2/units.Hz)
                    S[np.isnan(S)] = 0
                    S_per_bin = S * d_f
                    flux_sum += S_per_bin
                    E = np.sqrt(S_per_bin/(units.watt/units.m**2)/(scipy.constants.c*scipy.constants.epsilon_0))/(d_f) * units.V/units.m
                    ax_1.scatter(self.__interpolaiton_frequencies/units.MHz, noise_temperatures[:,i_zenith, i_azimuth]/units.kelvin, c='k', alpha=.01)
                    ax_1.plot(freqs[passband_filter]/units.MHz, noise_temperature, c='k', alpha=.02)
                    ax_2.plot(freqs[passband_filter]/units.MHz, S_per_bin/d_f/(units.watt/units.m**2/units.MHz), c='k', alpha=.02)
                    ax_3.plot(freqs[passband_filter]/units.MHz, E/(units.V/units.m), c='k', alpha=.02)

                    noise_spectrum = np.zeros((3, freqs.shape[0]), dtype=np.complex)
                    phases = np.random.uniform(0, 2. * np.pi, len(S))
                    polarizations = np.random.uniform(0,2.*np.pi, len(S))
                    noise_spectrum[1][passband_filter] = E * np.exp(1j*phases) * np.cos(polarizations)
                    noise_spectrum[2][passband_filter] = E * np.exp(1j*phases) * np.sin(polarizations)
                    efield_sum += noise_spectrum
                    VEL = trace_utilities.get_efield_antenna_factor(
                        station,
                        freqs,
                        [channel.get_id()],
                        detector,
                        zenith,
                        azimuth,
                        self.__antenna_pattern_provider
                    )[0]
                    channel_noise_spectrum = np.sum(VEL * np.array([noise_spectrum[1], noise_spectrum[2]]), axis=0)
                    ax_4.plot(freqs/units.MHz, np.abs(channel_noise_spectrum)/units.V, c='k', alpha=.01)
                    channel_spectrum += channel_noise_spectrum
                    noise_spec_sum += channel_noise_spectrum
                for efield in station.get_sim_station().get_electric_fields_for_channels([channel.get_id()]):
                    ax_3.plot(efield.get_frequencies()/units.MHz, np.abs(efield.get_frequency_spectrum()[1])/(units.V/units.m), c='C0')
                    ax_3.plot(efield.get_frequencies()/units.MHz, np.abs(efield.get_frequency_spectrum()[2])/(units.V/units.m), c='C1')
            ax_2.plot(freqs[passband_filter]/units.MHz, flux_sum/d_f/(units.watt/units.m**2/units.MHz), c='C0', label='total flux')
            ax_3.plot(freqs[passband_filter]/units.MHz, np.sqrt(np.abs(efield_sum[1])**2+np.abs(efield_sum[2])**2)[passband_filter]/(units.V/units.m), c='k', linestyle='-', label='sum of E-fields')
            ax_3.plot(freqs[passband_filter]/units.MHz, np.sqrt(flux_sum/(scipy.constants.c*(units.m/units.s))/(scipy.constants.epsilon_0*(units.farad/units.m)))/d_f/(units.V/units.m), c='C2', label='E-field from total flux')
            ax_4.plot(channel.get_frequencies()/units.MHz, np.abs(noise_spec_sum), c='k', linestyle=':', label='total noise')
            ax_2.legend()
            ax_3.legend()
            ax_4.legend()
            fig.tight_layout()
            channel.set_frequency_spectrum(channel_spectrum, sampling_rate)
            if self.__debug:
                ax1.plot(channel.get_times(), channel.get_trace(), label='new trace')
                ax1.plot(channel.get_times(), fft.freq2time(noise_spec_sum, channel.get_sampling_rate()), label='noise')
                ax2.plot(channel.get_frequencies()/units.MHz, np.abs(channel.get_frequency_spectrum()), label='new spectrum')
                ax2.plot(channel.get_frequencies()/units.MHz, np.abs(noise_spec_sum), label='noise')
                ax1.legend()
                ax2.legend()
                plt.show()
    def __get_galactic_noise_temp(self, x, zenith=0, azimuth=0, n=-2.41, k=10 ** 7.88):
        solid_angle = np.sin(zenith) * self.__delta_zenith * self.__delta_azimuth / 2./np.pi
        self.__solid_angle_sum += solid_angle
        return solid_angle * k * x ** n
