import numpy as np
from NuRadioReco.modules import channelGenericNoiseAdder
from NuRadioReco.utilities import units, fft
from NuRadioReco.modules.trigger.highLowThreshold import get_high_low_triggers
from NuRadioReco.detector import detector
from scipy import constants


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
        self.amplitude = (self.max_freq - self.min_freq) ** 0.5 / self.norm ** 0.5 * self.Vrms

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
                            n_traces[iCh] = np.roll(trace, self.trigger_bin - np.argwhere(triggered_bins is True)[0])
                        else:
                            tmp = np.random.randint(self.trigger_bin_low, self.trigger_bin)
                            n_traces[iCh] = np.roll(trace, tmp - np.argwhere(triggered_bins is True)[0])
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


from NuRadioReco.modules.analogToDigitalConverter import perfect_floor_comparator


class thermalNoiseGeneratorPhasedArray():

    def __init__(self, detector_filename, station_id, triggered_channels,
                 Vrms, threshold, ref_index,
                 trigger_time, filt, noise_type="rayleigh"):
        """
        Efficient algorithms to generate thermal noise fluctuations that fulfill a phased array trigger

        Parameters
        ----------
        detector_filename: string
            the filename to the detector description
        station_id: int
            the station id of the station from the detector file
        triggered_channels: array of ints
            list of channel ids that are part of the trigger
        Vrms: float
            the RMS noise
        threshold: float
            the trigger threshold (assuming a symmetric high and low threshold)
        trigger_time: float
            the trigger time (time when the trigger completes)
        filt: array of floats
            the filter that should be applied after noise generation (needs to match frequency binning)
        noise_type: string
            the type of the noise, can be
            * "rayleigh" (default)
            * "noise"
        """
        self.upsampling = 2
        self.det = detector.Detector(json_filename=detector_filename)
        self.n_samples = self.det.get_number_of_samples(station_id, triggered_channels[0]) # assuming same settings for all channels
        self.sampling_rate = self.det.get_sampling_frequency(station_id, triggered_channels[0])
        
        det_channel = self.det.get_channel(station_id, triggered_channels[0])
        self.adc_n_bits = det_channel["adc_nbits"]
        self.adc_noise_nbits = det_channel["adc_noise_nbits"]
        
        self.n_channels = len(triggered_channels)
        self.triggered_channels = triggered_channels
        self.ant_pos = {}
        for i, channel_id in enumerate(self.triggered_channels):
            self.ant_pos[channel_id] = self.det.get_relative_position(station_id, channel_id)[2]
        ref_z = np.max(np.fromiter(self.ant_z.values(), dtype=float))
        # Need to add in delay for trigger delay
        cable_delays = {}
        for channel_id in triggered_channels:
            cable_delays[channel_id] = self.det.get_cable_delay(station_id, channel_id)

        self.beam_rolls = []
        main_low_angle = np.deg2rad(-59.54968597864437)
        main_high_angle = np.deg2rad(59.54968597864437)
        phasing_angles_4ant = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 11))
        cspeed = constants.c * units.m / units.s
        for angle in phasing_angles_4ant:

            delays = []
            for key in self.ant_z:
                delays += [-(self.ant_z[key] - ref_z) / cspeed * ref_index * np.sin(angle) - cable_delays[key]]

            delays -= np.max(delays)

            roll = np.array(np.round(np.array(delays) * self.sampling_rate * self.upsampling)).astype(int)

            subbeam_rolls = dict(zip(triggered_channels, roll))

            self.beam_rolls.append(subbeam_rolls)


        self.Vrms = Vrms
        self.threshold = threshold
        self.trigger_time = trigger_time
        self.noise_type = noise_type

        self.min_freq = 0 * units.MHz
        self.max_freq = 0.5 * self.sampling_rate
        self.dt = 1. / self.sampling_rate
        self.ff = np.fft.rfftfreq(self.n_samples * self.upsampling, 1. / (self.sampling_rate * self.upsampling))
        self.filt = filt


        self.adc_ref_voltage = self.Vrms * (2 ** (self.adc_n_bits - 1) - 1) / (2 ** (self.adc_noise_n_bits - 1) - 1)

        self.window_4ant = int(16 * units.ns * self.sampling_rate * 2.0)
        self.step_4ant = int(8 * units.ns * self.sampling_rate * 2.0)

        self.trigger_bin = int(self.trigger_time / self.dt)
        self.trigger_bin_low = int((self.trigger_time - self.time_coincidence_majority) / self.dt)

        self.norm = np.trapz(np.abs(self.filt) ** 2, self.ff)
        self.amplitude = (self.max_freq - self.min_freq) ** 0.5 / self.norm ** 0.5 * self.Vrms

        self.noise = channelGenericNoiseAdder.channelGenericNoiseAdder()

    def generate_noise(self):
        """
        generates noise traces for all channels that will cause a high/low majority logic trigger

        Returns np.array of shape (n_channels, n_samples)
        """
        traces = np.zeros((self.n_channels, self.n_samples * self.upsampling))
        while True:
            for iCh in range(self.n_channels):
                spec = self.noise.bandlimited_noise(self.min_freq, self.max_freq, self.n_samples * self.upsampling,
                                                    self.sampling_rate * self.upsampling,
                                                    self.amplitude, self.noise_type, time_domain=False)
                spec *= self.filt
                trace = fft.freq2time(spec, self.sampling_rate)

                traces[iCh] = perfect_floor_comparator(trace, self.adc_n_bits, self.adc_ref_voltage)

            beam_rolls = self.calculate_time_delays(station, det,
                                            triggered_channels,
                                            phasing_angles,
                                            ref_index=ref_index,
                                            sampling_frequency=adc_sampling_frequency)

            phased_traces = self.phase_signals(traces, beam_rolls)

            for iTrace, phased_trace in enumerate(phased_traces):

                # Create a sliding window
                squared_mean, num_frames = self.power_sum(coh_sum=phased_trace, window=window, step=step, adc_output=adc_output)

                if True in (squared_mean > threshold):

                    trigger_delays = {}

                    for channel_id in beam_rolls[iTrace]:
                        trigger_delays[channel_id] = beam_rolls[iTrace][channel_id] * time_step

                    logger.debug("Station has triggered")
                    is_triggered = True

            return is_triggered, trigger_delays

                if(np.any(trace > self.threshold) and np.any(trace < -self.threshold)):
                    triggered_bins = get_high_low_triggers(trace, self.threshold, -self.threshold, self.time_coincidence, self.dt)
                    if(True in triggered_bins):
                        if(iCh == 0):
                            n_traces[iCh] = np.roll(trace, self.trigger_bin - np.argwhere(triggered_bins is True)[0])
                        else:
                            tmp = np.random.randint(self.trigger_bin_low, self.trigger_bin)
                            n_traces[iCh] = np.roll(trace, tmp - np.argwhere(triggered_bins is True)[0])
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

