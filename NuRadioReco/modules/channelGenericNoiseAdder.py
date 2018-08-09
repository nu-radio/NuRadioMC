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

        Parameters
        ---------

        f: array of floats
            Data that random phase is added to.
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
        Generating noise of n_samples in a bandwidth [min_freq,max_freq].

        Parameters
        ---------

        min_freq: float
            Minimum frequency of passband for noise generation
        max_freq: float
            Maximum frequency of passband for noise generation
        n_samples: int
            how many samples of noise should be generated
        sampling_rate: float
            desired sampling rate of data
        amplitude: float
            desired voltage of noise as V_rms (only roughly, since bandpass limited)
        type: string
            perfect_white: flat frequency spectrum
            white: flat frequency spectrum with random jitter

        """
        frequencies = np.abs(np.fft.fftfreq(n_samples, 1/sampling_rate))
        f = np.zeros(n_samples)
        selection = np.where((frequencies>=min_freq)& (frequencies<=max_freq))
        npise = None
        if type == 'perfect_white':
            f[selection] = amplitude*np.sqrt(2.*n_samples*2)
            noise = self.fftnoise(f)
        elif type == 'white':
            jitter = np.random.rand(n_samples)*0.05*amplitude + amplitude*np.sqrt(2.*n_samples*2)
            f[selection] = jitter[selection]
            noise = self.fftnoise(f)
        elif type == "narrowband":
            noise = None
        else:
            logger.error("Other types of noise not yet implemented.")

        return noise

    def __init__(self):
        pass

    def begin(self, debug=False):
        self.__debug = debug

    def run(self, event, station, detector,
                            amplitude=1*units.mV,
                            min_freq=50*units.MHz,
                            max_freq=2000*units.MHz,
                            type='white'):

        """
        Add noise to given event.

        Parameters
        ---------

        event

        station

        detector

        amplitude: float
            desired voltage of noise as V_rms (only roughly, since bandpass limited)
        min_freq: float
            Minimum frequency of passband for noise generation
        max_freq: float
            Maximum frequency of passband for noise generation
        type: string
            perfect_white: flat frequency spectrum
            white: flat frequency spectrum with random jitter

        """


        channels = station.get_channels()
        for channel in channels:

            trace = channel.get_trace()
            sampling_rate = channel.get_sampling_rate()

            noise = self.bandlimited_noise(min_freq=min_freq,
                                          max_freq=max_freq,
                                          n_samples=trace.shape[0],
                                          sampling_rate=sampling_rate,
                                          amplitude=amplitude,
                                          type=type)

            if self.__debug:
                new_trace = trace + noise

                logger.debug("imput amplitude {}".format(amplitude))
                logger.debug("voltage RMS {}".format(np.sqrt(np.mean(noise**2))))

                import matplotlib.pyplot as plt
                plt.plot(trace)
                plt.plot(noise)
                plt.plot(new_trace)

                plt.figure()
                plt.plot(np.abs(np.fft.rfft(trace)))
                plt.plot(np.abs(np.fft.rfft(noise)))
                plt.plot(np.abs(np.fft.rfft(new_trace)))

                plt.show()

            new_trace = trace + noise
            channel.set_trace(new_trace,sampling_rate)

    def end(self):
        pass