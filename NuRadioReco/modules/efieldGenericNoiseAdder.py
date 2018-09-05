from NuRadioReco.utilities import units
import logging
import copy
import NuRadioReco.framework.channel
import numpy as np
logger = logging.getLogger('efieldGenericNoiseAdder')
import matplotlib.pyplot as plt
import scipy as sp

class efieldGenericNoiseAdder:
    """
    Module that adds noise to the electric field -- such as Galactic emission.
    """

    def __init__(self):
        self.begin()

    def begin(self, debug=False):
        self.__debug = debug

    def run(self, evt, station, det, type, narrowband_freq, narrowband_power, passband):

        # access simulated efield and high level parameters
        sim_station = station.get_sim_station()
        sim_station_id = sim_station.get_id()
        azimuth = sim_station['azimuth']
        zenith = sim_station['zenith']
        event_time = sim_station.get_station_time()

        nChannels = det.get_number_of_channels(sim_station_id)
        sampling_rate = sim_station.get_sampling_rate()

        ff = sim_station.get_frequencies()

        efield_fft = copy.copy(sim_station.get_frequency_spectrum())
        efield = copy.copy(sim_station.get_trace()) # trace of the efield
        n_samples = efield.shape[1]

        if type == 'narrowband':

            for j, peak_pos in enumerate(narrowband_freq):
                norm_freq = np.random.normal(peak_pos, 0.05*units.MHz) # Typical width found by fitting to data
                t = np.linspace(0, n_samples * (1/sampling_rate), n_samples)
                for i, pol_power in enumerate(narrowband_power[j]):
                    pol_noise = pol_power * np.sin(norm_freq * 2 * np.pi * t)
                    efield[i] = efield[i] + pol_noise

            station.set_trace(efield, sim_station.get_sampling_rate())
            sim_station.set_trace(efield, sim_station.get_sampling_rate()) # To be removed once genericnoiseAdder module is corrected to accont for already configured station
            
        if type == 'galactic':

            freq_range_raw = sim_station.get_frequencies() # GHz
            lower_lim = freq_range_raw > passband[0]
            freq_range = freq_range_raw[lower_lim]
            noise_idx = int(efield_fft.shape[1]) - int(freq_range.shape[0])
            
            def GalTemp(x,n=-2.41,k=10**7.88): # expression found by fitting to fig. 3.2 in LOFAR
                return k * x**n
            
            B = np.zeros(efield_fft.shape[1])
            B[noise_idx:] = 4*np.pi * ((2*sp.constants.Boltzmann)/(sp.constants.speed_of_light**2)) * (freq_range/units.MHz)**2 * GalTemp(freq_range/units.MHz) * 50*units.ohm # unit W m^-2 Hz^-1
            # GalTemp term gives correct output - checked via plot
            Bnew = B*sampling_rate
            for i,x in enumerate(Bnew):
                if x != 0.: Bnew[i] = np.sqrt(x)
            

            logger.warning('Something is wrong with the normalization')
            const = 10**20
            #B = B*const
            
            # From Timo: Power density spectrum must be converter to fourier transform
            #fft_amp = (B * (n_samples - 1)) / (2 * sampling_rate)
            #for i,x in enumerate(fft_amp):
            #    if x != 0.: fft_amp[i] = np.sqrt(x)
            angle = np.random.uniform(0, 2*np.pi)
            gal_noise = Bnew * np.exp(angle * 1j)
            
            for i in range(3): # The polarization of the galactic emission is the same in all directions
                efield_fft[i] = efield_fft[i] + gal_noise
                
            station.set_frequency_spectrum(efield_fft, sampling_rate)
            sim_station.set_frequency_spectrum(efield_fft, sampling_rate) # To be removed once genericnoiseAdder module is corrected to account for already configured station
            
    def end():
        pass
