from __future__ import print_function
from NuRadioReco.modules.base.module import register_run
import numpy as np
from NuRadioReco.utilities import units, fft
import logging



class channelGenericNoiseAdder:
    """
    Module that generates noise in some generic fashion (not based on measured data), which can be added to data.


    """

    def add_random_phases(self, amps, n_samples_time_domain):
        """
        Adding random phase information to given amplitude spectrum.

        Parameters
        ---------

        amps: array of floats
            Data that random phase is added to.
        n_samples_time_domain: int
            number of samples in the time domain to differentiate between odd and even number of samples
        """
        amps = np.array(amps, dtype='complex')
        Np = (n_samples_time_domain - 1) // 2
        phases = np.random.rand(Np) * 2 * np.pi
        phases = np.cos(phases) + 1j * np.sin(phases)
        amps[1:Np + 1] *= phases  # Note that the last entry of the index slice is f[Np] !

        return amps

    def fftnoise_fullfft(self, f):
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
        f[1:Np + 1] *= phases  # Note that the last entry of the index slice is f[Np] !
        f[-1:-1 - Np:-1] = np.conj(f[1:Np + 1])

        self.logger.debug(' fftnoise: Length of frequency array = {} '.format(len(f)))
        self.logger.debug(' fftnoise: Number of points for unilateral spectrum = {} '.format(Np))
        self.logger.debug(' fftnoise: Max index and amplitude of positive part of spectrum: index = {}, A = |{}| = {} '.format(Np, f[Np], abs(f[Np])))
        self.logger.debug(' fftnoise: Min index and amplitude of negative part of spectrum: index = {}, A = |{}| '.format(len(f) - Np, f[-Np]))

        fftprec = max(abs(np.fft.ifft(f) - np.fft.ifft(f).real))
        fftcheck = fftprec - np.finfo(float).resolution
        self.logger.debug(' fftnoise: fft precision {} < {} (float resolution) is : {} !'.format(fftprec , np.finfo(float).resolution, fftcheck < 0))

        if fftcheck >= 0:
            self.logger.warning(' fftnoise: Non negligibe imagniary part of inverse FFT: {} '.format(fftcheck))

        return np.fft.ifft(f).real

    def bandlimited_noise(self, min_freq, max_freq, n_samples, sampling_rate, amplitude, type='perfect_white',
                          time_domain=True, bandwidth=None):
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
            If the maximum frequency is above the Nquist frequencey (0.5 * sampling rate), the Nquist frequency is used
            max_freq = None: Frequencies up to Nyquist freq are used.
        n_samples: int
            number of samples in the time domain
        sampling_rate: float
            desired sampling rate of data
        amplitude: float
            desired voltage of noise as V_rms (only roughly, since bandpass limited)
        type: string
            perfect_white: flat frequency spectrum
            rayleigh: Amplitude of each frequency bin is drawn from a Rayleigh distribution
            # white: flat frequency spectrum with random jitter
        time_domain: bool (default True)
            if True returns noise in the time domain, if False it returns the noise in the frequency domain. The latter
            might be more performant as the noise is generated internally in the frequency domain.
        bandwidth: float or None (default)
            if this parameter is specified, the amplitude is interpreted as the amplitude for the bandwidth specified here
            Otherwise the amplitude is interpreted for the bandwidth of min(max_freq, 0.5 * sampling rate) - min_freq
            If `bandwidth` is larger then (min(max_freq, 0.5 * sampling rate) - min_freq) it has the same effect as `None`

        Comments
        --------
        *   Note that by design the max frequency is the Nyquist frequency, even if a bigger max_freq
            is implemented (RL 17-Sept-2018)

        *   Add 'multi_white' noise option on 20-Sept-2018 (RL)

        """
        frequencies = np.fft.rfftfreq(n_samples, 1. / sampling_rate)

        n_samples_freq = len(frequencies)

        if min_freq == None or min_freq == 0:
            # remove DC component; fftfreq returns the DC component as 0-th element and the negative
            # frequencies at the end, so frequencies[1] should be the lowest frequency; it seems safer,
            # to take the difference between two frequencies to determine the minimum frequency, in case
            # future versions of numpy change the order and maybe put the negative frequencies first
            min_freq = 0.5 * (frequencies[2] - frequencies[1])
            self.logger.info(' Set min_freq from None to {} MHz!'.format(min_freq / units.MHz))
        if max_freq == None:
            # sample up to Nyquist frequency
            max_freq = max(frequencies)
            self.logger.info(' Set max_freq from None to {} GHz!'.format(max_freq / units.GHz))
        selection = (frequencies >= min_freq) & (frequencies <= max_freq)

        nbinsactive = np.sum(selection)
        self.logger.debug('Total number of frequency bins (bilateral spectrum) : {} , of those active: {} '.format(n_samples, nbinsactive))

        # Debug plots
#         f1 = plt.figure()
#         plt.plot (frequencies/max(frequencies))
#         plt.plot(fbinsactive,'kx')

        if(bandwidth is not None):
            sampling_bandwidth = min(0.5 * sampling_rate, max_freq) - min_freq
            amplitude *= 1. / (bandwidth / (sampling_bandwidth)) ** 0.5  # normalize noise level to the bandwidth its generated for

        ampl = np.zeros(n_samples_freq)
        sigscale = (1.*n_samples) / np.sqrt(nbinsactive)
        if type == 'perfect_white':
            ampl[selection] = amplitude * sigscale
        elif type == 'rayleigh':
            fsigma = amplitude * sigscale / np.sqrt(2.)
            ampl[selection] = np.random.rayleigh(fsigma, nbinsactive)
#         elif type == 'white':
# FIXME: amplitude normalization is not correct for 'white'
#             ampl = np.random.rand(n_samples) * 0.05 * amplitude + amplitude * np.sqrt(2.*n_samples * 2)
        else:
            self.logger.error("Other types of noise not yet implemented.")
            raise NotImplementedError("Other types of noise not yet implemented.")

        noise = self.add_random_phases(ampl, n_samples) / sampling_rate
        if(time_domain):
            return fft.freq2time(noise, sampling_rate, n=n_samples)
        else:
            return noise

    def __init__(self):
        self.begin()
        self.logger = logging.getLogger('NuRadioReco.channelGenericNoiseAdder')

    def begin(self, debug=False):
        self.__debug = debug
        if debug:
            self.logger.setLevel(logging.DEBUG)

    @register_run()
    def run(self, event, station, detector,
                            amplitude=1 * units.mV,
                            min_freq=50 * units.MHz,
                            max_freq=2000 * units.MHz,
                            type='perfect_white',
                            excluded_channels=[],
                            bandwidth=None):

        """
        Add noise to given event.

        Parameters
        ---------

        event

        station

        detector

        amplitude: float
            desired voltage of noise as V_rms for the specified bandwidth
        min_freq: float
            Minimum frequency of passband for noise generation
        max_freq: float
            Maximum frequency of passband for noise generation
            If the maximum frequency is above the Nquist frequencey (0.5 * sampling rate), the Nquist frequency is used
        type: string
            perfect_white: flat frequency spectrum
            rayleigh: Amplitude of each frequency bin is drawn from a Rayleigh distribution
        excluded_channels: list of ints
            the channels ids of channels where no noise will be added, default is that no channel is excluded
        bandwidth: float or None (default)
            if this parameter is specified, the amplitude is interpreted as the amplitude for the bandwidth specified here
            Otherwise the amplitude is interpreted for the bandwidth of min(max_freq, 0.5 * sampling rate) - min_freq
            If `bandwidth` is larger then (min(max_freq, 0.5 * sampling rate) - min_freq) it has the same effect as `None`

        """

        channels = station.iter_channels()
        for channel in channels:
            if(channel.get_id() in excluded_channels):
                continue

            trace = channel.get_trace()
            sampling_rate = channel.get_sampling_rate()

            noise = self.bandlimited_noise(min_freq=min_freq,
                                          max_freq=max_freq,
                                          n_samples=trace.shape[0],
                                          sampling_rate=sampling_rate,
                                          amplitude=amplitude,
                                          type=type,
                                          bandwidth=bandwidth)

            if self.__debug:
                new_trace = trace + noise

                self.logger.debug("imput amplitude {}".format(amplitude))
                self.logger.debug("voltage RMS {}".format(np.sqrt(np.mean(noise ** 2))))

                import matplotlib.pyplot as plt
                plt.plot(trace)
                plt.plot(noise)
                plt.plot(new_trace)

                plt.figure()
                plt.plot(np.abs(fft.time2freq(trace, channel.get_sampling_rate())))
                plt.plot(np.abs(fft.time2freq(noise, channel.get_sampling_rate())))
                plt.plot(np.abs(fft.time2freq(new_trace, channel.get_sampling_rate())))

                plt.show()

            new_trace = trace + noise
            channel.set_trace(new_trace, sampling_rate)

    def end(self):
        pass
