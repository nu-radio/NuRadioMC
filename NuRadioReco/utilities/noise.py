import numpy as np
from NuRadioReco.modules import channelGenericNoiseAdder
from NuRadioReco.utilities import units, fft
from NuRadioReco.modules.trigger.highLowThreshold import get_high_low_triggers


class thermalNoiseGenerator():

    def __init__(self, n_samples, sampling_rate, Vrms, threshold, time_coincidence, n_majority, time_coincidence_majority,
                 n_channels, trigger_time, filt, noise_type="rayleigh"):
        """
        Efficient algorithms to generate thermal noise fluctuations that fulfill a high/low trigger + a majority
        coincidence logic (as used by ARIANNA)

        Parameters
        ----------
        n_samples: int
            the number of samples of the trace
        sampling_rate: float
            the sampling rate
        Vrms: float
            the RMS noise
        threshold: float
            the trigger threshold (assuming a symmetric high and low threshold)
        time_coincidence: float
            the high/low coincidence time
        n_majority: int
            specifies how many channels need to have a trigger
        time_coincidence_majority: float
            multi channel coincidence window
        n_channels: int
            number of channels to generate
        trigger_time: float
            the trigger time (time when the trigger completes)
        filt: array of floats
            the filter that should be applied after noise generation (needs to match frequency binning)
        noise_type: string
            the type of the noise, can be
            * "rayleigh" (default)
            * "noise"
        """
        self.n_samples = n_samples
        self.sampling_rate = sampling_rate

        self.Vrms = Vrms
        self.threshold = threshold
        self.time_coincidence = time_coincidence
        self.n_majority = n_majority
        self.time_coincidence_majority = time_coincidence_majority
        self.trigger_time = trigger_time
        self.n_channels = n_channels
        self.noise_type = noise_type

        self.min_freq = 0 * units.MHz
        self.max_freq = 0.5 * self.sampling_rate
        self.dt = 1. / self.sampling_rate
        self.ff = np.fft.rfftfreq(self.n_samples, 1. / self.sampling_rate)
        self.filt = filt

        self.trigger_bin = int(self.trigger_time / self.dt)
        self.trigger_bin_low = int((self.trigger_time - self.time_coincidence_majority) / self.dt)

        self.norm = np.trapz(np.abs(self.filt) ** 2, self.ff)
        self.amplitude = (self.max_freq - self.min_freq)**0.5 / self.norm**0.5 * self.Vrms

        self.noise = channelGenericNoiseAdder.channelGenericNoiseAdder()

    def generate_noise(self):
        """
        generates noise traces for all channels that will cause a high/low majority logic trigger

        Returns np.array of shape (n_channels, n_samples)
        """
        n_traces = [None] * self.n_majority
        t_bins = [None] * self.n_majority
        for iCh in range(self.n_majority):
            while n_traces[iCh] is None:
                spec = self.noise.bandlimited_noise(self.min_freq, self.max_freq, self.n_samples, self.sampling_rate,
                                                    self.amplitude, self.noise_type, time_domain=False)
                spec *= self.filt
                trace = fft.freq2time(spec, self.sampling_rate)
                if(np.any(trace > self.threshold) and np.any(trace < -self.threshold)):
                    triggered_bins = get_high_low_triggers(trace, self.threshold, -self.threshold, self.time_coincidence, self.dt)
                    if(True in triggered_bins):
                        t_bins[iCh] = triggered_bins
                        if(iCh == 0):
                            n_traces[iCh] = np.roll(trace, self.trigger_bin - np.argwhere(triggered_bins == True)[0])
                        else:
                            tmp = np.random.randint(self.trigger_bin_low, self.trigger_bin)
                            n_traces[iCh] = np.roll(trace, tmp - np.argwhere(triggered_bins == True)[0])
        traces = np.zeros((self.n_channels, self.n_samples))
        rnd_iterator = list(range(self.n_channels))
        np.random.shuffle(rnd_iterator)
        for i, iCh in enumerate(rnd_iterator):
            if(i < self.n_majority):
                traces[iCh] = n_traces[i]
            else:
                spec = self.noise.bandlimited_noise(self.min_freq, self.max_freq, self.n_samples, self.sampling_rate,
                                                    self.amplitude, type=self.noise_type, time_domain=False)
                spec *= self.filt
                traces[iCh] = fft.freq2time(spec, self.sampling_rate)
        return traces
