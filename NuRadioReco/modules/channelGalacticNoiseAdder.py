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

logger = logging.getLogger('channelGalacticNoiseAdder')

class channelGalacticNoiseAdder:
    def __init__(self):
        self.begin()

    def begin(self, debug=False, n_zenith=18, n_azimuth=36):
        self.__debug = debug
        self.__zenith_sample = np.linspace(0, 90, n_zenith)[1:] * units.deg
        self.__azimuth_sample = np.linspace(0, 360, n_azimuth)[1:] * units.deg
        self.__delta_zenith = self.__zenith_sample[1] - self.__zenith_sample[0]
        self.__delta_azimuth = self.__azimuth_sample[1] - self.__azimuth_sample[0]
        self.__antenna_pattern_provider = NuRadioReco.detector.antennapattern.AntennaPatternProvider()
    def run(self, event, station, detector, passband=[10*units.MHz, 1000*units.MHz]):

        for channel in station.iter_channels():
            freqs = channel.get_frequencies()   #assume all channels have the same fequencies
            times = channel.get_times()
            sampling_rate = channel.get_sampling_rate()
            channel_trace = channel.get_trace()

            passband_filter = (freqs>passband[0])&(freqs<passband[1])
            #S = np.zeros(freqs[passband_filter].shape)
            self.__solid_angle_sum = 0
            for zenith in self.__zenith_sample:
                for azimuth in self.__azimuth_sample:
                    noise_temperature = self.__get_galactic_noise_temp(freqs[passband_filter] / units.MHz, zenith, azimuth)
                    S = 4 * np.pi * ((2 * sp.constants.Boltzmann/(units.joule/units.kelvin)) / ((sp.constants.speed_of_light/(units.m/units.s)) ** 2)) * (freqs[passband_filter] / units.MHz) ** 2 * noise_temperature * 50 * units.ohm

                    noise_spectrum = np.zeros((3, freqs.shape[0]), dtype=np.complex)
                    phases = np.random.uniform(0, 2 * np.pi, len(S))
                    polarizations = np.random.uniform(0,2*np.pi, len(S))
                    noise_spectrum[1][passband_filter] = S * np.exp(1j*phases) * np.cos(polarizations)
                    noise_spectrum[2][passband_filter] = S * np.exp(1j*phases) * np.sin(polarizations)
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
                    #channel_trace += fft.freq2time(channel_noise_spectrum, sampling_rate)
                    #if self.__debug:
                    #    plt.close('all')
                    #    fig = plt.figure()
                    #    ax1 = fig.add_subplot(221)
                    #    ax2 = fig.add_subplot(222)
                    #    ax3 = fig.add_subplot(223)
                    #    ax4 = fig.add_subplot(224)
                    #    ax1.plot(freqs/units.MHz, np.abs(noise_spectrum[1]))
                    #    ax1.plot(freqs/units.MHz, np.abs(noise_spectrum[2]))
                    #    ax1.plot(freqs/units.MHz, np.sqrt(np.abs(noise_spectrum[1])**2+np.abs(noise_spectrum[2])**2))
                    #    ax2.plot(times, fft.freq2time(noise_spectrum[1], sampling_rate))
                    #    ax2.plot(times, fft.freq2time(noise_spectrum[2], sampling_rate))
                    #    ax3.plot(freqs/units.MHz, np.abs(fft.time2freq(channel_trace, sampling_rate)))
                    #    ax4.plot(times, channel_trace)
                    #    plt.show()
            channel.set_trace(channel_trace, sampling_rate)
            if self.__debug:
                plt.close('all')
                fig = plt.figure()
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)
                ax1.plot(channel.get_times(), channel.get_trace())
                ax2.plot(channel.get_frequencies()/units.MHz, np.abs(channel.get_frequency_spectrum()))
                plt.show()
    def __get_galactic_noise_temp(self, x, zenith=0, azimuth=0, n=-2.41, k=10 ** 7.88):
        solid_angle = np.sin(zenith) * self.__delta_zenith * self.__delta_azimuth / 2./np.pi
        self.__solid_angle_sum += solid_angle
        return solid_angle * k * x ** n
