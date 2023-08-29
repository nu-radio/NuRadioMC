import numpy as np
from NuRadioReco.modules import channelGenericNoiseAdder
from NuRadioReco.utilities import units, fft
from NuRadioReco.modules.trigger.highLowThreshold import get_high_low_triggers
from NuRadioReco.detector import generic_detector as detector
from scipy import constants
import datetime
import scipy
import scipy.signal
import copy
import time

import logging
logger = logging.getLogger('noiseTriggerSimulation')


def rolled_sum_roll(traces, rolling):
    """
    calculates rolled sum via np.roll

    Parameters
    ----------
    traces: list
        list containing 1D traces
    rolling: list
        list of offsets per trace for rolling

    Returns
    -------
    sumtr: np.array
        1D array of summed traces
    """

    # assume first trace always has no rolling
    sumtr = traces[0].copy()
    for i in range(1, len(traces)):
        sumtr += np.roll(traces[i], rolling[i])
    return sumtr


def rolling_indices(traces, rolling):
    """
    pre calculates rolling index array for rolled sum via take
  
    Parameters
    ----------
    traces: list
        list containing (at least one) 1D trace
    rolling: list
        list of offsets per trace for rolling
    """

    rolling_indices = []
    idx = np.arange(len(traces[0]))
    for roll in rolling:
        rolling_indices.append(np.roll(idx, roll))
    return np.array(rolling_indices).astype(int)


def rolled_sum_take(traces, rolling_indices):
    """
    calculates rolled sum via np.take

    Parameters
    ----------
    traces: list
        list containing 1D traces
    rolling_indices: list
        list of pre-calculated 1D index arrays for rolled sum

    Returns
    -------
    sumtr: np.array
        1D array of summed traces
    """

    # assume first trace always has no rolling
    sumtr = traces[0].copy()
    for i in range(1, len(traces)):
        sumtr += np.take(traces[i], rolling_indices[i])
    return sumtr


def rolled_sum_slicing(traces, rolling):
    """
    calculates rolled sum via slicing

    Parameters
    ----------
    traces: list
        list containing 1D traces
    rolling: list
        list of offsets per trace for rolling

    Returns
    -------
    sumtr: np.array
        1D array of summed traces
    """

    # assume first trace always has no rolling
    sumtr = traces[0].copy()
    for i in range(1, len(traces)):
        r = rolling[i]
        if r > 0:
            sumtr[:-r] += traces[i][r:]
            sumtr[-r:] += traces[i][:r]
        elif r < 0:
            sumtr[:r] += traces[i][-r:]
            sumtr[r:] += traces[i][:-r]
        else:
            sumtr += traces[i]
    return sumtr


class thermalNoiseGenerator():

    def __init__(self, n_samples, sampling_rate, Vrms, threshold, time_coincidence, n_majority, time_coincidence_majority,
                 n_channels, trigger_time, filt, noise_type="rayleigh", keep_full_band=False):
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
        keep_full_band: bool
            Flag to keep the full-band waveform instead of the one with the filter applied. Allows for
            triggering on a different (i.e. `filt`) band than what is returned, (default is False)
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

        self.keep_full_band = keep_full_band

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

                if self.keep_full_band:
                    trace_copy = fft.freq2time(spec, self.sampling_rate)

                spec *= self.filt
                trace = fft.freq2time(spec, self.sampling_rate)
                if(np.any(trace > self.threshold) and np.any(trace < -self.threshold)):
                    triggered_bins = get_high_low_triggers(trace, self.threshold, -self.threshold, self.time_coincidence, self.dt)
                    if(True in triggered_bins):
                        t_bins[iCh] = triggered_bins
                        trace_to_keep = trace if not self.keep_full_band else trace_copy
                        if(iCh == 0):
                            n_traces[iCh] = np.roll(trace_to_keep, self.trigger_bin - np.argwhere(triggered_bins == True)[0])
                        else:
                            tmp = np.random.randint(self.trigger_bin_low, self.trigger_bin)
                            n_traces[iCh] = np.roll(trace_to_keep, tmp - np.argwhere(triggered_bins == True)[0])
        traces = np.zeros((self.n_channels, self.n_samples))
        rnd_iterator = list(range(self.n_channels))
        np.random.shuffle(rnd_iterator)
        for i, iCh in enumerate(rnd_iterator):
            if(i < self.n_majority):
                traces[iCh] = n_traces[i]
            else:
                spec = self.noise.bandlimited_noise(self.min_freq, self.max_freq, self.n_samples, self.sampling_rate,
                                                    self.amplitude, type=self.noise_type, time_domain=False)

                if self.keep_full_band:
                    traces[iCh] = fft.freq2time(spec, self.sampling_rate)
                else:
                    traces[iCh] = fft.freq2time(spec * self.filt, self.sampling_rate)
        return traces


from NuRadioReco.modules.analogToDigitalConverter import perfect_floor_comparator


class thermalNoiseGeneratorPhasedArray():

    def __init__(self, detector_filename, station_id, triggered_channels,
                 Vrms, threshold, ref_index,
                 noise_type="rayleigh", log_level=logging.WARNING,
                 pre_trigger_time=100 * units.ns, trace_length=512 * units.ns, filt=None,
                 upsampling=2, window_length=16 * units.ns, step_size=8 * units.ns,
                 main_low_angle=np.deg2rad(-59.54968597864437), 
                 main_high_angle=np.deg2rad(59.54968597864437),
                 n_beams=11, quantize=True):
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
        ref_index: float
            reference refractive index for calculating time delays of the beams

        Other Parameters
        ----------------
        noise_type: string
            the type of the noise, can be
            * "rayleigh" (default)
            * "noise"
        log_level: logging enum, default warn 
            the print level for this module
        pre_trigger_time: float, default 100 ns
            the time in the trace before the trigger happens
        trace_length: float, default 512 ns
            the total trace length
        filt: array of complex values, default None
            the filter that should be applied after noise generation (needs to match frequency binning in upsampled domain)
            if `None`, a default filter is calculated from 96 to 220 MHz
        upsampling: int, default 2
            factor by which the waveforms will be upsampled before calculating time delays and beamforming
        window_length: float, default 16 ns
            time interval of the integration window
        step_size: float, default 8 ns
            duration of a stride between window calcuations
        main_low_angle: float, default -59.5 deg
            angle (radians) of the lowest beam
        main_high_angle: float 59.5 deg
            angle (radians) of the highest beam
        n_beams: int, default 11
            number of beams to calculate
        quantize: bool, default True
            If set to true, the conversion to and from ADC will be performed to mimic digitizations
        """
        logger.setLevel(log_level)
        self.debug = False
        self.max_amp = 0

        self.upsampling = upsampling
        self.det = detector.GenericDetector(json_filename=detector_filename)
        self.det.update(datetime.datetime(2018, 10, 1))
        self.n_samples = self.det.get_number_of_samples(station_id, triggered_channels[0])  # assuming same settings for all channels
        self.sampling_rate = self.det.get_sampling_frequency(station_id, triggered_channels[0])

        self.pre_trigger_bins = int(pre_trigger_time * self.sampling_rate)
        self.n_samples_trigger = int(trace_length * self.sampling_rate)

        if self.n_samples_trigger > self.n_samples:
            raise ValueError(f"Requested `trace_length` of {trace_length/units.ns:.1f}ns ({self.n_samples_trigger} bins)" +
                             f" is longer than the number of samples specified in the detector file ({self.n_samples} bins)")

        self.quantize = quantize
        if self.quantize:
            det_channel = self.det.get_channel(station_id, triggered_channels[0])
            self.adc_n_bits = det_channel["trigger_adc_nbits"]
            self.adc_noise_n_bits = det_channel["trigger_adc_noise_nbits"]

        self.n_channels = len(triggered_channels)
        self.triggered_channels = triggered_channels
        self.ant_z = {}
        for i, channel_id in enumerate(self.triggered_channels):
            self.ant_z[channel_id] = self.det.get_relative_position(station_id, channel_id)[2]
        ref_z = np.max(np.fromiter(self.ant_z.values(), dtype=float))
        # Need to add in delay for trigger delay
        cable_delays = {}
        for channel_id in triggered_channels:
            cable_delays[channel_id] = self.det.get_cable_delay(station_id, channel_id)

        phasing_angles = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), n_beams))
        cspeed = constants.c * units.m / units.s
        self.beam_time_delays = np.zeros((len(phasing_angles), self.n_channels), dtype=np.int)
        for iBeam, angle in enumerate(phasing_angles):

            delays = []
            for key in self.ant_z:
                delays += [-(self.ant_z[key] - ref_z) / cspeed * ref_index * np.sin(angle) - cable_delays[key]]

            delays -= np.max(delays)

            roll = np.array(np.round(np.array(delays) * self.sampling_rate * self.upsampling)).astype(int)
            self.beam_time_delays[iBeam] = roll

        self.Vrms = Vrms
        self.threshold = threshold
        self.noise_type = noise_type

        self.min_freq = 0 * units.MHz
        self.max_freq = 0.5 * self.sampling_rate * self.upsampling
        self.dt = 1. / self.sampling_rate
        self.ff = np.fft.rfftfreq(self.n_samples * self.upsampling, 1. / (self.sampling_rate * self.upsampling))

        # Construct a default filter if one is not supplied
        if filt is None:
            import NuRadioReco.modules.channelBandPassFilter
            channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
            self.filt = channelBandPassFilter.get_filter(self.ff, station_id, channel_id, self.det,
                              passband=[96 * units.MHz, 100 * units.GHz], filter_type='cheby1', order=4, rp=0.1)
            self.filt *= channelBandPassFilter.get_filter(self.ff, station_id, channel_id, self.det,
                              passband=[1 * units.MHz, 220 * units.MHz], filter_type='cheby1', order=7, rp=0.1)
        else:
            if len(filt) != len(self.ff):
                raise ValueError(f"Frequency filter supplied has {len(filt)} bins. It should match the upsampled" +
                                 f" frequency binning of {len(self.ff)} bins from {self.ff[0] / units.MHz:.0f} to {self.ff[-1] / units.MHz:.0f} MHz")
            self.filt = np.array(filt)

        self.norm = np.trapz(np.abs(self.filt) ** 2, self.ff)
        self.amplitude = (self.max_freq - self.min_freq) ** 0.5 / self.norm ** 0.5 * self.Vrms
        logger.info(f"Vrms = {self.Vrms:.3g}V, noise amplitude = {self.amplitude:.3g}V, bandwidth = {self.norm / units.MHz:.0f}MHz")
        logger.info(f"frequency range {self.min_freq / units.MHz}MHz - {self.max_freq / units.MHz}MHz")

        if self.quantize:
            self.adc_ref_voltage = self.Vrms * (2 ** (self.adc_n_bits - 1) - 1) / (2 ** (self.adc_noise_n_bits - 1) - 1)

        self.window = int(window_length * self.sampling_rate * self.upsampling)
        self.step = int(step_size * self.sampling_rate * self.upsampling)

        if self.window >= self.pre_trigger_bins:
            logger.warning(f"Pre-trigger time ({pre_trigger_time / units.ns:0.2} ns, {self.pre_trigger_bins} bins)" +
                           f" is within one window ({self.window} bins) of the beginning of the waveform" +
                           " it is recommended to choose a larger value to avoid clipping effects")

        self.noise = channelGenericNoiseAdder.channelGenericNoiseAdder()

        # pre-calculate all parameters which will be used to simulate each triggered noise event to avoid re-calculation
        self.noise.precalculate_bandlimited_noise_parameters(self.min_freq, self.max_freq, self.n_samples * self.upsampling,
                                                  self.sampling_rate * self.upsampling,
                                                  self.amplitude, self.noise_type)

    def __generation(self):
        """ separated trace generation part for PA noise trigger """

        for iCh in range(self.n_channels):
            # spec = self.noise.bandlimited_noise(self.min_freq, self.max_freq, self.n_samples * self.upsampling,
            #                                    self.sampling_rate * self.upsampling,
            #                                    self.amplitude, self.noise_type, time_domain=False)

            # function that does not re-calculate parameters in each simulated trace
            spec = self.noise.bandlimited_noise_from_precalculated_parameters(self.noise_type, time_domain=False)
            spec *= self.filt
            trace = fft.freq2time(spec, self.sampling_rate * self.upsampling)

            if self.quantize:
                self._traces[iCh] = perfect_floor_comparator(trace, self.adc_n_bits, self.adc_ref_voltage)
            else:
                self._traces[iCh] = trace

    def __phasing(self):
        """ separated phasing part for PA noise trigger """

        self._phased_traces = np.zeros((len(self.beam_time_delays), self.n_samples * self.upsampling))
        for iBeam, beam_time_delay in enumerate(self.beam_time_delays):
            self._phased_traces[iBeam] += rolled_sum_slicing(self._traces, beam_time_delay)

    def __phasing_roll(self):
        """ separated phasing part for PA noise trigger via np.roll """

        self._phased_traces = np.zeros((len(self.beam_time_delays), self.n_samples * self.upsampling))
        for iBeam, beam_time_delay in enumerate(self.beam_time_delays):
            for iCh in range(self.n_channels):
                self._phased_traces[iBeam] += np.roll(self._traces[iCh], beam_time_delay[iCh])

    def __triggering(self):
        """ separated trigger part for PA noise trigger """

        self.max_amp = 0
        # take square over entire array
        coh_sum_squared = self._phased_traces ** 2
        # bin the data into windows of length self.step and normalise to step length
        reduced_array = np.add.reduceat(coh_sum_squared.T, np.arange(0, np.shape(coh_sum_squared)[1], self.step)).T / self.step

        sliding_windows = []
        # self.window can extend over multiple steps,
        # assuming self.window being an integer multiple of self.step the reduction sums over subsequent steps
        steps_per_window = self.window // self.step
        # better extend the array in order to also trigger on sum of last/first (matching a previous implementation)
        extended_reduced_array = np.column_stack([reduced_array, reduced_array[:, 0:steps_per_window]])
        for offset in range(steps_per_window):
            # sum over steps_per_window adjacent steps
            window_sum = np.add.reduceat(extended_reduced_array.T, np.arange(offset, np.shape(extended_reduced_array)[1], steps_per_window)).T / steps_per_window
            sliding_windows.append(window_sum)
        # self.max_amp = max(np.array(sliding_windows).max(), self.max_amp)
        self.max_amp = np.array(sliding_windows).max()

        # check if trigger condition is fulfilled anywhere
        if self.max_amp > self.threshold:
            sliding_windows = np.array(sliding_windows)
            tmp = np.argwhere(sliding_windows > self.threshold)
            triggered_step, triggered_beam, triggered_bin = tmp[0]
            triggered_bin *= (self.window + triggered_step * steps_per_window)
            if(self.debug):
                # check in which beam the trigger condition was fulfilled
                sliding_windows = np.concatenate(sliding_windows, axis=1)
                triggered_beams = np.amax(sliding_windows, axis=1) > self.threshold
                for iBeam, is_triggered in enumerate(triggered_beams):
                    # print out each beam that has triggered
                    if is_triggered:
                        logger.info(f"triggered at beam {iBeam}")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(self.n_channels + 1, 1, sharex=True)
                for iCh in range(self.n_channels):
                    ax[iCh].plot(self._traces[iCh])
                    logger.info(f"{self._traces[iCh].std():.2f}")
                ax[self.n_channels].plot(self._phased_traces[iBeam])
                fig.tight_layout()
                plt.show()
            return True, triggered_bin, triggered_beam
        return False, None, None

    def __triggering_strided(self):
        """ separated trigger part for PA noise trigger using np.lib.stride_tricks.as_strided """

        self.max_amp = 0
        for iBeam, phased_trace in enumerate(self._phased_traces):
            # Create a sliding window
            coh_sum_squared = phased_trace ** 2
            num_frames = int(np.floor((len(phased_trace) - self.window) / self.step))
            coh_sum_windowed = np.lib.stride_tricks.as_strided(coh_sum_squared, (num_frames, self.window),
                                                       (coh_sum_squared.strides[0] * self.step, coh_sum_squared.strides[0]))
            squared_mean = np.sum(coh_sum_windowed, axis=1) / self.window
            # self.max_amp = max(squared_mean.max(), self.max_amp)
            self.max_amp = max(squared_mean.max(), self.max_amp)
            if True in (squared_mean > self.threshold):
                triggered_bin = np.where(squared_mean > self.threshold)[0,0]
                logger.debug(f"triggered at beam {iBeam}")
                if(self.debug):
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(self.n_channels+1, 1, sharex=True)
                    for iCh in range(self.n_channels):
                        ax[iCh].plot(self._traces[iCh])
                        logger.info(f"{self._traces[iCh].std():.2f}")
                    ax[self.n_channels].plot(self._phased_traces[iBeam])
                    fig.tight_layout()
                    plt.show()
                return True, triggered_bin, iBeam
        return False, None, None

    def generate_noise(self, phasing_mode="slice", trigger_mode="binned_sum", debug=False):
        """
        generates noise traces for all channels that will cause a high/low majority logic trigger

        Parameters
        ----------
        phasing_mode: string (default: "slice")
            "slice" or "roll", two implementations for phasing by either slicing the array or using
            np.roll. The default shows better performance, but "roll" is kept as an alternative
        trigger_mode: string (default: "binned_sum")
            "binned_sum" or "stride", two implementations for triggering
            The default shows better performance, but "stride" is kept as an alternative
        debug:
            generate debug plot
        Returns
        -------
        np.array of shape (n_channels, n_samples), index of triggered bin, index of triggered beam
        """
        self.debug = debug

        # some variables for profiling code
        dt_generation = 0
        dt_phasing = 0
        dt_triggering = 0

        # generate empty trace array
        self._traces = np.zeros((self.n_channels, self.n_samples * self.upsampling))
        counter = 0

        while True:
            counter += 1
            if(counter % 1000 == 0):
                logger.info(f"{counter:d}, {self.max_amp:.2g}, threshold = {self.threshold:.2g}")
                # some printout for profiling
                logger.info(f"Time consumption: GENERATION: {dt_generation:.4f}, PHASING: {dt_phasing:.4f}, TRIGGER: {dt_triggering:.4f}")
            tstart = time.process_time()

            self.__generation()

            # time profiling generation
            dt_generation += time.process_time() - tstart
            tstart = time.process_time()

            if phasing_mode == "slice":
                self.__phasing()
            elif phasing_mode == "roll":
                # more time consuming attempt to do phasing compared to slicing the array
                self.__phasing_roll()
            else:
                logger.error(f"Requested phasing_mode {phasing_mode}. Only 'slice' and 'roll' are allowed")
                raise NotImplementedError(f"Requested phasing_mode {phasing_mode}. Only 'slice' and 'roll' are allowed")

            # time profiling phasing
            dt_phasing += time.process_time() - tstart
            tstart = time.process_time()

            if trigger_mode == "binned_sum":
                is_triggered, triggered_bin, triggered_beam = self.__triggering()
            elif trigger_mode == "stride":
                # more time consuming attempt to do triggering compared to taking binned sums
                is_triggered, triggered_bin, triggered_beam = self.__triggering_strided()
            else:
                logger.error(f"Requested trigger_mode {trigger_mode}. Only 'binned_sum' and 'stride' are allowed")
                raise NotImplementedError(f"Requested trigger_mode {trigger_mode}. Only 'binned_sum'  and 'stride' are supported")

            # time profiling trigger
            dt_triggering += time.process_time() - tstart

            if is_triggered:
                triggered_bin = triggered_bin // self.upsampling  # the trace is cut in the downsampled version. Therefore, triggered bin is factor of two smaller. 
                i_low = triggered_bin - self.pre_trigger_bins
                i_high = i_low + self.n_samples_trigger

                # traces need to be downsampled
                # resample and use axis -1 since trace might be either shape (N) for analytic trace or shape (3,N) for E-field
                self._traces = scipy.signal.resample(self._traces, np.shape(self._traces)[-1] // self.upsampling, axis=-1)

                if (i_low >= 0) and (i_high < self.n_samples): # If range is directly a subset of the waveform
                    return self._traces[:, i_low:i_high], self._phased_traces, triggered_beam

                # Otherwise, roll the waveforms. Safe as long as noise is generated in the freq domain
                self._phased_traces = np.roll(self._phased_traces, -i_low * self.upsampling, axis=-1)
                self._traces = np.roll(self._traces, -i_low, axis=-1)
                return self._traces[:, :self.n_samples_trigger], self._phased_traces, triggered_beam

    def generate_noise2(self, debug=False):
        """
        generates noise traces for all channels that will cause a high/low majority logic trigger
        This implementation is slow due to nested python loops each rolling the trace over and over again

        Returns np.array of shape (n_channels, n_samples)
        """
        traces = np.zeros((self.n_channels, self.n_samples * self.upsampling))
        counter = 0
        max_amp = 0
        while True:
            counter += 1
            if(counter % 1000 == 0):
                print(f"{counter:d}, {max_amp:.3g}, threshold = {self.threshold:.3g}")
            for iCh in range(self.n_channels):
                spec = self.noise.bandlimited_noise(self.min_freq, self.max_freq, self.n_samples * self.upsampling,
                                                    self.sampling_rate * self.upsampling,
                                                    self.amplitude, self.noise_type, time_domain=False)
                spec *= self.filt
                trace = fft.freq2time(spec, self.sampling_rate * self.upsampling)

                if self.quantize:
                    self._traces[iCh] = perfect_floor_comparator(trace, self.adc_n_bits, self.adc_ref_voltage)
                else:
                    self._traces[iCh] = trace

            shifts = np.zeros(self.n_channels, dtype=np.int)
            shifted_traces = copy.copy(traces)
            for shift1 in np.arange(-100, 100, 4, dtype=np.int):
                shifted_traces[1] = np.roll(traces[1], shift1)
                shifts[1] = shift1
                for shift2 in np.arange(-100, 100, 4, dtype=np.int):
                    shifts[2] = shift2
                    shifted_traces[2] = np.roll(traces[2], shift2)

                    for shift3 in np.arange(-100, 100, 4, dtype=np.int):
                        shifts[3] = shift3
                        shifted_traces[3] = np.roll(traces[3], shift3)
                        phased_trace = np.zeros(self.n_samples * self.upsampling)
                        for iCh in range(self.n_channels):
                            phased_trace += shifted_traces[iCh]

                        # Create a sliding window
                        coh_sum_squared = phased_trace ** 2
                        num_frames = int(np.floor((len(phased_trace) - self.window) / self.step))
                        coh_sum_windowed = np.lib.stride_tricks.as_strided(coh_sum_squared, (num_frames, self.window),
                                                                   (coh_sum_squared.strides[0] * self.step, coh_sum_squared.strides[0]))
                        squared_mean = np.sum(coh_sum_windowed, axis=1) / self.window
                        max_amp = max(squared_mean.max(), max_amp)

                        if True in (squared_mean > self.threshold):
                            logger.info(f"triggered at beam {shifts}")
                            if(debug):
                                import matplotlib.pyplot as plt
                                fig, ax = plt.subplots(5, 1, sharex=True)
                                for iCh in range(self.n_channels):
                                    ax[iCh].plot(traces[iCh])
                                    logger.info(f"{traces[iCh].std():.2f}")
                                ax[4].plot(phased_trace)
                                fig.tight_layout()
                                plt.show()
                            return traces, phased_trace
