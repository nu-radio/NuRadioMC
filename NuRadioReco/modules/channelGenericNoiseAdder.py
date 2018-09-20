from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units
import logging
logger = logging.getLogger('channelGenericNoiseAdder')
logging.basicConfig() # basicConfig adds a StreamHandler to the root logger.


class channelGenericNoiseAdder:
    """
    Module that generates noise in some generic fashion (not based on measured data), which can be added to data.


    """

    def fftnoise(self, f):
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
        f[1:Np + 1] *= phases # Note that the last entry of the index slice is f[Np] !
        f[-1:-1 - Np:-1] = np.conj(f[1:Np + 1])
        
        logger.debug(' fftnoise: Length of frequency array = {} '.format(len(f)))
        logger.debug(' fftnoise: Number of points for unilateral spectrum = {} '.format(Np))
        logger.debug(' fftnoise: Max index and amplitude of positive part of spectrum: index = {}, A = |{}| = {} '.format(Np,f[Np],abs(f[Np])))
        logger.debug(' fftnoise: Min index and amplitude of negative part of spectrum: index = {}, A = |{}| '.format(len(f)-Np,f[-Np]))
        
        fftprec = max(abs(np.fft.ifft(f) - np.fft.ifft(f).real))
        fftcheck = fftprec - np.finfo(float).resolution
        logger.debug(' fftnoise: fft precision {} < {} (float resolution) is : {} !'.format(fftprec ,np.finfo(float).resolution, fftcheck<0))
        
        if fftcheck>=0:
            logger.warning(' fftnoise: Non negligibe imagniary part of inverse FFT: {} '.format(fftcheck))
        
        return np.fft.ifft(f).real

    def bandlimited_noise(self, min_freq, max_freq, n_samples, sampling_rate, amplitude, type='perfect_white'):
        """
        Generating noise of n_samples in a bandwidth [min_freq,max_freq].

        Parameters
        ---------

        min_freq: float
            Minimum frequency of passband for noise generation
            min_freq = None: Only the DC component is removed. If the DC component should be included, 
            min_freq = 0 has to be specified
        max_freq: float
            Maximum frequency of passband for noise generation
            max_freq = None: Frequencies up to Nyquist freq are used.
        n_samples: int
            how many samples of noise should be generated
        sampling_rate: float
            desired sampling rate of data
        amplitude: float
            desired voltage of noise as V_rms (only roughly, since bandpass limited)
        type: string
            perfect_white: flat frequency spectrum
            white: flat frequency spectrum with random jitter
            multi_white: Allows for the specifications of a amplitude distribution by providing the argument
                'multi_white <ampdist>' with the possible strings for <ampdist>:
                    'flat' : same as type='perfect_white'
                    'rayleigh': Amplitude of each frequency bin is drawn from a Rayleigh distribution
        Comments
        --------
        *   Note that by design the max frequency is the Nyquist frequency, even if a bigger max_freq
            is implemented (RL 17-Sept-2018)
        
        *   Add 'multi_white' noise option on 20-Sept-2018 (RL)

        """
        # for noise type 'multi_white', the character string also contains the type of amplitude distribution
        # that shall be assumed, e.g. 'multi_white flat', where 'flat is the default'
        typearray = type.split()
        if typearray[0]=='multi_white':
            type = typearray[0]
            ntypewords = len(typearray)
            logger.debug(' Noise type *** {} *** provided with *** {} *** arguments'.format(typearray[0],ntypewords-1))
            if ntypewords > 1:
                ampdist = typearray[1]
                logger.debug(' Amplitude distribution is: {} '.format(ampdist))
            else:
                ampdist = 'flat'
                logger.debug(' Amplitude distribution set to default: {} '.format(ampdist))
                
        frequencies = np.abs(np.fft.fftfreq(n_samples, 1 / sampling_rate))
        
        f = np.zeros(n_samples)
        if min_freq == None:
            # remove DC component; fftfreq returns the DC component as 0-th element and the negative
            # frequencies at the end, so frequencies[1] should be the lowest frequency; it seems safer,
            # to take the difference between two frequencies to determine the minimum frequency, in case
            # future versions of numpy change the order and maybe put the negative frequencies first 
            min_freq = 0.5*(frequencies[2]-frequencies[1]) 
            logger.info(' Set min_freq from None to {} MHz!'.format(min_freq/units.MHz))
        if max_freq == None:
            # sample up to Nyquist frequency
            max_freq = max(frequencies)
            logger.info(' Set max_freq from None to {} GHz!'.format(max_freq/units.GHz))
        selection = np.where((frequencies >= min_freq) & (frequencies <= max_freq))
        
        fbinsactive = np.zeros(n_samples)
        fbinsactive[selection]=1
        
        nbinsactive = sum(fbinsactive)
        logger.debug('Total number of frequency bins (bilateral spectrum) : {} , of those active: {} '.format(n_samples,nbinsactive))
        
        # Debug plots
#         f1 = plt.figure()
#         plt.plot (frequencies/max(frequencies))
#         plt.plot(fbinsactive,'kx')
        
        if type == 'perfect_white':
            f[selection] = amplitude * np.sqrt(2.*n_samples * 2)
            noise = self.fftnoise(f)
            # quick hack to normalize the Vrms to the correct value
            noise *= amplitude / noise.std()
        elif type == 'white':
            jitter = np.random.rand(n_samples) * 0.05 * amplitude + amplitude * np.sqrt(2.*n_samples * 2)
            f[selection] = jitter[selection]
            noise = self.fftnoise(f)
            # quick hack to normalize the Vrms to the correct value
            noise *= amplitude / noise.std()
        elif type == 'multi_white':
            # get amplitude distribution
            ampl = np.ones(len(f))
            sigscale = (1.*n_samples)/np.sqrt(nbinsactive)
            if ampdist=='flat':
                ampl *= amplitude * sigscale
            elif ampdist=='rayleigh':
                fsigma=amplitude*sigscale/np.sqrt(2.)
                ampl = np.random.rayleigh(fsigma, n_samples)
            else:
                logger.error(" 'multi_white' has no amplitude distribution '{}' !".format(ampdist))
                ampl *=0;
                
            # multiply each frequency with random phase to get noise
            f[selection] = ampl[selection]
            noise = self.fftnoise(f)
        elif type == "narrowband":
            noise = None
        else:
            logger.error("Other types of noise not yet implemented.")

        # quick hack to normalize the Vrms to the correct value
        #noise *= amplitude / noise.std()
        
        return noise

    def __init__(self):
        self.begin()

    def begin(self, debug=False):
        self.__debug = debug
        if debug:
            #logger = logging.getLogger("channelGenericNoiseAdder")
            logger.setLevel(logging.DEBUG)

    def run(self, event, station, detector,
                            amplitude=1 * units.mV,
                            min_freq=50 * units.MHz,
                            max_freq=2000 * units.MHz,
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
                logger.debug("voltage RMS {}".format(np.sqrt(np.mean(noise ** 2))))

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
            channel.set_trace(new_trace, sampling_rate)

    def end(self):
        pass
