from NuRadioReco.utilities import units
from NuRadioReco.modules.base.module import register_run
import logging
import copy
import NuRadioReco.framework.channel
import numpy as np
logger = logging.getLogger('efieldGenericNoiseAdder')
import matplotlib.pyplot as plt
import scipy as sp


class efieldGenericNoiseAdder:
    """
    Module that adds noise to the electric field -- such as Galactic emission and background from human communication.
    
    
    """

    def __init__(self):
        self.begin()

    def begin(self, debug=False):
        self.__debug = debug

    @register_run()
    def run(self, evt, station, det, type, narrowband_freq, narrowband_power, passband):
        """
        
        Parameters
        ----------
        event
        
        station
        
        detector
        
        type: string
            narrowband: single frequency background with FM
            galactic: galactic emission calculated from temperature plots
        narrowband_freq: list
            mean frequency of one or more narrowband transmitters
        narrowband_power: list
            must be same size as narrowband_freq list. Each element in the list should be a list consisting of 3 elements: the power of the signal in x,y,z polarization.
        passband: list
            galactic background will not be simulated below the lower cutoff
        """

        # access simulated efield and high level parameters

        for electric_field in station.get_electric_fields():
            sampling_rate = electric_field.get_sampling_rate()

            efield_fft = copy.copy(electric_field.get_frequency_spectrum())
            efield = copy.copy(electric_field.get_trace())
            n_samples = efield.shape[1]

            if type == 'narrowband':

                for j, peak_pos in enumerate(narrowband_freq):
                    norm_freq = np.random.normal(peak_pos, 0.05 * units.MHz)  # Typical width found by fitting to data
                    t = np.linspace(0, n_samples * (1 / sampling_rate), n_samples)
                    for i, pol_power in enumerate(narrowband_power[j]):
                        pol_noise = pol_power * np.sin(norm_freq * 2 * np.pi * t)
                        efield[i] = efield[i] + pol_noise

                electric_field.set_trace(efield, sampling_rate)

            elif type == 'galactic':

                freq_range_raw = electric_field.get_frequencies()
                lower_lim = freq_range_raw > passband[0]
                freq_range = freq_range_raw[lower_lim]
                noise_idx = int(efield_fft.shape[1]) - int(freq_range.shape[0])

                # Power law found by fit to Fig 3.2. "An absolute calibration of the antennas at LOFAR", Tijs Karskens (2015)
                def GalTemp(x, n=-2.41, k=10 ** 7.88):
                    return k * x ** n

                S = np.zeros(efield_fft.shape[1])
                S[noise_idx:] = 4 * np.pi * ((2 * sp.constants.Boltzmann) / (sp.constants.speed_of_light ** 2)) * (freq_range / units.MHz) ** 2 * GalTemp(freq_range / units.MHz) * 50 * units.ohm
                S = S * sampling_rate

                for i, x in enumerate(S):
                    if x != 0.: S[i] = np.sqrt(x)

                angle = np.random.uniform(0, 2 * np.pi)
                gal_noise = S * np.exp(angle * 1j) * 100.

                for i in range(3):  # The power of the galactic background is assumed to be the same in all polarizations
                    efield_fft[i] = efield_fft[i] + gal_noise

                electric_field.set_frequency_spectrum(efield_fft, sampling_rate)

            else:
                logger.error("Other types of noise not yet implemented.")

    def end():
        pass
