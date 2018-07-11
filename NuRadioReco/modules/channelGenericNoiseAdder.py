import numpy as np
from NuRadioReco.utilities import units
import logging
logger = logging.getLogger('channelGenericNoiseAdder')


class channelGenericNoiseAdder:
    """
    Module that generates noise in some generic fashion (not based on measured data), which can be added to data. 
    """
    
    def fftnoise(self,f):
        """
        Adding random phase information to given amplitude spectrum. 
        """
        f = np.array(f, dtype='complex')
        Np = (len(f) - 1) // 2
        phases = np.random.rand(Np) * 2 * np.pi
        phases = np.cos(phases) + 1j * np.sin(phases)
        f[1:Np+1] *= phases
        f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
        return np.fft.ifft(f).real
    
    def bandlimited_noise(self,min_freq, max_freq, n_samples, sampling_rate,amplitude, type='perfect_white'):
        """
        Generating noise in a certain bandwidth. 
        
        type = perfect_white: flat frequency spectrum
        type = white: flat frequency spectrum with random jitter
        
        amplitude = desired voltage of noise in spectrum 
        """
        frequencies = np.abs(np.fft.fftfreq(n_samples, 1/sampling_rate))
        f = np.zeros(n_samples)
        selection = np.where((frequencies>=min_freq)& (frequencies<=max_freq))
        if type == 'perfect_white':
            f[selection] = amplitude
        elif type == 'white':
            jitter = np.random.rand(n_samples)*0.001*amplitude
            f[selection] = jitter[selection]
        else:
            logger.error("Other type of noise not yet implemented.")
            
        return self.fftnoise(f)
    
    def __init__(self):
        pass
        
    def begin(self, debug=False):
        self.__debug = debug
    
    def run(self, event, station, detector, 
                            amplitude=1*units.mV,
                            min_freq=50*units.MHz,
                            max_freq=2000*units.MHz):
        
        channels = station.get_channels()
        for channel in channels:
            trace = channel.get_trace()
            sampling_rate = channel.get_sampling_rate()
            
            noise = self.bandlimited_noise(min_freq=min_freq,
                                          max_freq=max_freq,
                                          n_samples=trace.shape[0],
                                          sampling_rate=sampling_rate,
                                          amplitude=amplitude)
            
            if self.__debug:
                new_trace = trace + noise
            
                import matplotlib.pyplot as plt
                plt.plot(trace)
                plt.plot(noise)
                plt.plot(new_trace)
            
            
                plt.figure()
                plt.plot(np.abs(np.fft.rfft(trace)))
                plt.plot(np.abs(np.fft.rfft(noise)))
                plt.plot(np.abs(np.fft.rfft(new_trace)))

                plt.show()
                1/0
            
            trace += noise
        
    def end(self):
        pass