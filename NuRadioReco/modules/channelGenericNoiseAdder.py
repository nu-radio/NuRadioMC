import logging
import numpy as np
from functools import lru_cache
from scipy import integrate
from numpy.random import Generator, Philox
from NuRadioReco.utilities import units, fft
from NuRadioReco.modules.base.module import register_run
import warnings


class channelGenericNoiseAdder:
    """
    Module that generates noise in some generic fashion (not based on measured data), which can be added to data.


    """
    def add_random_phases(self, amps, n_samples_time_domain):
        """
        Adding random phase information to given amplitude spectrum.

        Parameters
        ----------

        amps: array of floats
            Data that random phase is added to.
        n_samples_time_domain: int
            number of samples in the time domain to differentiate between odd and even number of samples
        """
        amps = np.array(amps, dtype='complex')
        Np = (n_samples_time_domain - 1) // 2
        phases = self.__random_generator.random(Np) * 2 * np.pi
        phases = np.cos(phases) + 1j * np.sin(phases)
        amps[1:Np + 1] *= phases  # Note that the last entry of the index slice is f[Np] !

        return amps

    def fftnoise_fullfft(self, *args, **kwargs):
        """Deprecated"""
        warnings.warn("The 'fftnoise_fullfft' method will be deprecated in a future release", DeprecationWarning)
        return self._fftnoise_fullfft(*args, **kwargs)

    def _fftnoise_fullfft(self, f):
        """
        Adding random phase information to given amplitude spectrum.

        Parameters
        ----------

        f: array of floats
            Data that random phase is added to.
        """
        f = np.array(f, dtype='complex')
        Np = (len(f) - 1) // 2
        phases = self.__random_generator.random(Np) * 2 * np.pi
        phases = np.cos(phases) + 1j * np.sin(phases)
        f[1:Np + 1] *= phases  # Note that the last entry of the index slice is f[Np] !
        f[-1:-1 - Np:-1] = np.conj(f[1:Np + 1])

        self.logger.debug(' fftnoise: Length of frequency array = {} '.format(len(f)))
        self.logger.debug(' fftnoise: Number of points for unilateral spectrum = {} '.format(Np))
        self.logger.debug(' fftnoise: Max index and amplitude of positive part of spectrum: index = {}, A = |{}| = {} '.format(Np, f[Np], abs(f[Np])))
        self.logger.debug(' fftnoise: Min index and amplitude of negative part of spectrum: index = {}, A = |{}| '.format(len(f) - Np, f[-Np]))

        fftprec = max(abs(np.fft.ifft(f) - np.fft.ifft(f).real))
        fftcheck = fftprec - np.finfo(float).resolution
        self.logger.debug(' fftnoise: fft precision {} < {} (float resolution) is : {} !'.format(fftprec, np.finfo(float).resolution, fftcheck < 0))

        if fftcheck >= 0:
            self.logger.warning(' fftnoise: Non negligibe imagniary part of inverse FFT: {} '.format(fftcheck))

        return np.fft.ifft(f).real

    def _get_noise_generation_parameters(self, min_freq, max_freq, n_samples, sampling_rate, bandwidth):
        """
        Compute (and cache) the frequency-domain bookkeeping needed to generate bandlimited noise.

        The frequency binning and the selection of bins within the requested passband only depend on
        `min_freq`, `max_freq`, `n_samples`, `sampling_rate` and `bandwidth` - not on the noise
        amplitude, type, or the random realization. In many use cases (e.g. adding noise to every
        channel of every event with the same passband), this method is called repeatedly with the
        same parameters, so we cache the result to avoid recomputing it every time.

        Parameters
        ----------
        min_freq: float
            Minimum frequency of passband for noise generation.
            min_freq = None: Only the DC component is removed. If the DC component should be included,
            min_freq = 0 has to be specified.
        max_freq: float
            Maximum frequency of passband for noise generation.
            If the maximum frequency is above the Nyquist frequency (0.5 * sampling rate), the Nyquist
            frequency is used. max_freq = None: Frequencies up to Nyquist freq are used.
        n_samples: int
            number of samples in the time domain
        sampling_rate: float
            desired sampling rate of data
        bandwidth: float or None
            if this parameter is specified, the amplitude is interpreted as the amplitude for the
            bandwidth specified here. Otherwise the amplitude is interpreted for the bandwidth of
            min(max_freq, 0.5 * sampling rate) - min_freq. If `bandwidth` is larger than
            (min(max_freq, 0.5 * sampling rate) - min_freq) it has the same effect as `None`.

        Returns
        -------
        params: dict
            Dictionary with the keys ``n_samples``, ``n_samples_freq``, ``sampling_rate``,
            ``selection``, ``nbinsactive``, ``sigscale`` and ``bandwidth_scale``.

        Notes
        -----
        This method is wrapped with `functools.lru_cache` in `__init__` (bound per-instance,
        not as a class-level decorator - see the note there), so repeated calls with the same
        arguments hit the cache instead of recomputing.
        """
        frequencies = fft.freqs(n_samples, sampling_rate)
        n_samples_freq = len(frequencies)

        if min_freq is None or min_freq == 0:
            # remove DC component; fftfreq returns the DC component as 0-th element and the negative
            # frequencies at the end, so frequencies[1] should be the lowest frequency; it seems safer,
            # to take the difference between two frequencies to determine the minimum frequency, in case
            # future versions of numpy change the order and maybe put the negative frequencies first
            min_freq = 0.5 * (frequencies[2] - frequencies[1])
            self.logger.info('Set min_freq from None to {} MHz!'.format(min_freq / units.MHz))

        if max_freq is None:
            # sample up to Nyquist frequency
            max_freq = max(frequencies)
            self.logger.info('Set max_freq from None to {} GHz!'.format(max_freq / units.GHz))
        else:
            if round(max_freq, 3) > round(frequencies[-1], 3):
                self.logger.warning(
                    f'max_freq ({max_freq / units.MHz:.2f} MHz) is above the Nyquist frequency '
                    f'({frequencies[-1] / units.MHz:.2f} MHz). This means the simulated noise ampitude '
                    'might deviate from what you intended. To fix that, you either need to lower '
                    'max_freq or increase the sampling_rate.')

        selection = (frequencies >= min_freq) & (frequencies <= max_freq)
        nbinsactive = np.sum(selection)
        self.logger.debug('Total number of frequency bins (bilateral spectrum) : {} , of those active: {} '.format(n_samples, nbinsactive))

        if bandwidth is not None:
            sampling_bandwidth = min(0.5 * sampling_rate, max_freq) - min_freq
            bandwidth_scale = 1. / (bandwidth / sampling_bandwidth) ** 0.5  # normalize noise level to the bandwidth its generated for
        else:
            bandwidth_scale = 1.

        sigscale = (1. * n_samples) / np.sqrt(nbinsactive)

        params = {
            "n_samples": n_samples,
            "n_samples_freq": n_samples_freq,
            "sampling_rate": sampling_rate,
            "selection": selection,
            "nbinsactive": nbinsactive,
            "sigscale": sigscale,
            "bandwidth_scale": bandwidth_scale,
        }

        return params

    def _draw_amplitude_spectrum(self, n_samples_freq, selection, nbinsactive, amplitude, sigscale, type):
        """
        Draw a random amplitude spectrum for the frequency bins selected by `selection`.

        This is the shared core of `bandlimited_noise` and `bandlimited_noise_from_spectrum`: given
        the number of active (in-passband) frequency bins and the amplitude scaling to use, it draws
        the per-bin amplitudes for the requested noise `type`.

        Parameters
        ----------
        n_samples_freq: int
            length of the (unilateral) frequency spectrum
        selection: array of bool
            boolean mask of length `n_samples_freq` selecting the frequency bins in the passband
        nbinsactive: int
            number of `True` entries in `selection`
        amplitude: float
            desired voltage of noise as V_rms
        sigscale: float
            scaling factor converting the target V_rms into the per-bin Rayleigh scale parameter
        type: string
            rayleigh: Amplitude of each frequency bin is drawn from a Rayleigh distribution

        Returns
        -------
        ampl: array of floats
            amplitude spectrum of length `n_samples_freq`
        """
        if type is None:
            raise ValueError("You have to provide a noise \"type\" for the channelGenericNoiseAdder. "
                             "We removed \"perfect_white\", only option is \"rayleigh\".")

        if type in ('white', 'perfect_white'):
            raise ValueError(
                f"The noise type '{type}' is no longer supported. Please use type='rayleigh' instead, "
                "which draws the amplitude of every frequency bin from a Rayleigh distribution and is "
                "the physically correct model for thermal/white noise."
            )
        if type != 'rayleigh':
            self.logger.error("Other types of noise not yet implemented.")
            raise NotImplementedError("Other types of noise not yet implemented.")

        ampl = np.zeros(n_samples_freq)
        fsigma = amplitude * sigscale / np.sqrt(2.)
        ampl[selection] = self.__random_generator.rayleigh(fsigma, nbinsactive)

        return ampl

    def bandlimited_noise(self, min_freq, max_freq, n_samples, sampling_rate, amplitude, type=None,
                          time_domain=True, bandwidth=None):
        """
        Generate noise of n_samples in a bandwidth [min_freq,max_freq].

        Parameters
        ----------

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
            required, no default. Only "rayleigh" is currently supported: the amplitude of each
            frequency bin is drawn from a Rayleigh distribution. "white"/"perfect_white" (a flat,
            non-physical amplitude spectrum) are no longer supported and raise a ValueError.
        time_domain: bool (default True)
            if True returns noise in the time domain, if False it returns the noise in the frequency domain. The latter
            might be more performant as the noise is generated internally in the frequency domain.
        bandwidth: float or None (default)
            if this parameter is specified, the amplitude is interpreted as the amplitude for the bandwidth specified here
            Otherwise the amplitude is interpreted for the bandwidth of min(max_freq, 0.5 * sampling rate) - min_freq
            If `bandwidth` is larger then (min(max_freq, 0.5 * sampling rate) - min_freq) it has the same effect as `None`

        Notes
        -----
        *   Note that by design the max frequency is the Nyquist frequency, even if a bigger max_freq
            is implemented (RL 17-Sept-2018)

        *   The frequency-domain bookkeeping (frequency binning, passband selection) is cached internally,
            keyed on (min_freq, max_freq, n_samples, sampling_rate, bandwidth). Calling this method
            repeatedly with the same values for those parameters (e.g. once per channel per event) is
            therefore cheap, even though a new random noise realization is drawn on every call.

        """
        params = self._get_noise_generation_parameters(min_freq, max_freq, n_samples, sampling_rate, bandwidth)
        amplitude = amplitude * params["bandwidth_scale"]

        ampl = self._draw_amplitude_spectrum(
            params["n_samples_freq"], params["selection"], params["nbinsactive"], amplitude, params["sigscale"], type)

        noise = self.add_random_phases(ampl, params["n_samples"]) / params["sampling_rate"]
        if time_domain:
            return fft.freq2time(noise, params["sampling_rate"], n=params["n_samples"])
        else:
            return noise

    def bandlimited_noise_from_spectrum(self, n_samples, sampling_rate, spectrum, amplitude=None, type=None,
                          time_domain=True):
        """
        Generate noise of n_samples with a given frequency spectrum shape.

        Parameters
        ----------
        n_samples: int
            number of samples in the time domain
        sampling_rate: float
            desired sampling rate of data
        spectrum: numpy.ndarray, function
            desired spectrum of the noise, either as a numpy.ndarray of length n_frequencies or a function
            that takes the frequencies as an argument and returns the amplitudes. The overall normalization
            of the spectrum is ignored if the paramter "amplitude" is set.
        amplitude: float, optional
            desired voltage of noise as V_rms. If set to None the power of the noise will be equal to the
            power of the spectrum.
        type: string
            required, no default. Only "rayleigh" is currently supported: the amplitude of each
            frequency bin is drawn from a Rayleigh distribution. "white"/"perfect_white" (a flat,
            non-physical amplitude spectrum) are no longer supported and raise a ValueError.
        time_domain: bool (default True)
            if True returns noise in the time domain, if False it returns the noise in the frequency domain. The latter
            might be more performant as the noise is generated internally in the frequency domain.

        See Also
        --------
        bandlimited_noise: generates noise with a flat/rectangular passband instead of an arbitrary spectrum shape
        """
        frequencies = fft.freqs(n_samples, sampling_rate)
        selection = frequencies > 0
        n_samples_freq = np.sum(selection)

        if callable(spectrum):
            spectrum = spectrum(frequencies)

        if amplitude is not None:
            norm = integrate.trapezoid(np.abs(spectrum) ** 2, frequencies)
            max_freq = frequencies[-1]
            amplitude = amplitude / (norm / max_freq) ** 0.5
            sigscale = (1. * n_samples) / np.sqrt(n_samples_freq)
        else:
            amplitude = np.sqrt(n_samples)
            sigscale = 1.

        ampl = self._draw_amplitude_spectrum(len(frequencies), selection, n_samples_freq, amplitude, sigscale, type)

        noise = self.add_random_phases(ampl, n_samples) / sampling_rate
        noise *= spectrum
        if time_domain:
            return fft.freq2time(noise, sampling_rate, n=n_samples)
        else:
            return noise

    def __init__(self):
        self.__debug = None
        self.__random_generator = None
        self.logger = logging.getLogger('NuRadioReco.channelGenericNoiseAdder')

        # Wrap `_get_noise_generation_parameters` with `functools.lru_cache`, bound here per-instance
        # rather than as a `@lru_cache` decorator on the class method. Decorating the method directly
        # would key the (module-wide, class-level) cache on `self` too, keeping every instance that
        # was ever called alive forever (and mixing entries from all instances in one cache). Rebinding
        # it here instead gives each `channelGenericNoiseAdder` its own independent, instance-scoped
        # cache that is garbage collected along with the instance.
        self._get_noise_generation_parameters = lru_cache(maxsize=32)(self._get_noise_generation_parameters)

        self.begin()

    def begin(self, debug=False, seed=None):
        self.__debug = debug
        self.__random_generator = Generator(Philox(seed))
        if debug:
            self.logger.setLevel(logging.DEBUG)

    @register_run()
    def run(self, event, station, detector,
            amplitude=1 * units.mV,
            min_freq=50 * units.MHz,
            max_freq=2000 * units.MHz,
            type=None,
            excluded_channels=None,
            bandwidth=None):

        """
        Add noise to given event.

        Parameters
        ----------

        event

        station

        detector

        amplitude: float or dict of floats
            desired voltage of noise as V_rms for the specified bandwidth
            a dict can be used to specify a different amplitude per channel, the key is the channel_id
        min_freq: float
            Minimum frequency of passband for noise generation
        max_freq: float
            Maximum frequency of passband for noise generation
            If the maximum frequency is above the Nquist frequencey (0.5 * sampling rate), the Nquist frequency is used
        type: string
            required, no default. Only "rayleigh" is currently supported: the amplitude of each
            frequency bin is drawn from a Rayleigh distribution. "white"/"perfect_white" (a flat,
            non-physical amplitude spectrum) are no longer supported and raise a ValueError.
        excluded_channels: list of ints
            the channels ids of channels where no noise will be added, default is that no channel is excluded
        bandwidth: float or None (default)
            if this parameter is specified, the amplitude is interpreted as the amplitude for the bandwidth specified here
            Otherwise the amplitude is interpreted for the bandwidth of min(max_freq, 0.5 * sampling rate) - min_freq
            If `bandwidth` is larger then (min(max_freq, 0.5 * sampling rate) - min_freq) it has the same effect as `None`

        """
        if excluded_channels is None:
            excluded_channels = []
        channels = station.iter_channels()
        for channel in channels:
            if(channel.get_id() in excluded_channels):
                continue

            trace = channel.get_trace()
            sampling_rate = channel.get_sampling_rate()

            if(isinstance(amplitude, dict)):
                tmp_ampl = amplitude[channel.get_id()]
            else:
                tmp_ampl = amplitude

            noise = self.bandlimited_noise(min_freq=min_freq,
                                           max_freq=max_freq,
                                           n_samples=trace.shape[0],
                                           sampling_rate=sampling_rate,
                                           amplitude=tmp_ampl,
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
