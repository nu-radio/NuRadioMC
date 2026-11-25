from NuRadioReco.framework.parameters import stationParametersRNOG as stpRNOG
from NuRadioReco.framework.parameters import channelParameters as chp

from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import trace_utilities, units

import matplotlib
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

import logging
import warnings
logger = logging.getLogger('NuRadioReco.RNO_G.stationCoherentlySummedWaveforms')


class stationCoherentlySummedWaveforms:
    """
    Generates a coherently-summed waveform (CSW) and calculates its signal-to-noise-ratio (SNR).

    When multiple waveforms and a referance waveform are given,
    one can find the cross correlation between each waveform and the referance waveform,
    then each waveform will be rolled to line up the signal position with the referance
    based on the time lag when both waveforms are most correlated,
    and then all waveforms including the reference will be summed up
    to become a coherently-summed waveform (CSW).
    Thermal noise is random so it scales with the square root of added waveforms while coherent signals
    will added up linearly. Once summed the CSW is treated as a regular waveform and different analysis variables
    are calculated.
    """

    def __init__(self):
        """(Unused)"""
        pass

    def begin(self, coincidence_window_size=6 * units.ns, pad_length=500, channel_ids=[0, 1, 2, 3], time_lag_range=None,
              ref_time_window=None, crop_to_overlap=False, lag_channel_ids=None, lag_template="sum",
              time_lag_ref_ch_id=None):
        """
        Parameters
        ----------
        coincidence_window_size: float (default: 6 * units.ns)
            Window size used for calculating the maximum peak to peak amplitude in nanoseconds

        pad_length: int (default: 500)
            Padding length used for calculating the coherent sum

        channel_ids: array of int (default: [0, 1, 2, 3])
            Channels which are summed (and for which time lags are calculated)

        time_lag_range: tuple of (float, float), dict or None (default: None)
            ``(min, max)`` of the allowed time lags for the cross correlation, in units of time
            (e.g. ``(-50 * units.ns, 50 * units.ns)``), for all channels, or a dict
            ``{channel_id: (min, max)}`` per channel (channels not listed: all lags). If None,
            the full trace is used. Time lags are relative to ``time_lag_ref_ch_id`` (``coherent_time_lags``),
            also if the reference channel of the sum is another one (``ref_selection`` of ``run``): the
            ranges are then translated into lags relative to that reference channel.

        ref_time_window: tuple of (float, float) or None (default: None)
            ``(start, stop)`` time window of the reference channel used for the cross correlation,
            in units of time and relative to the trace start of the reference channel (i.e., ``(0, 100 * units.ns)``
            are the first 100 ns of the trace). Only this part of the reference trace contributes to the
            correlation. If None, the full reference trace is used.

        crop_to_overlap: bool (default: False)
            If False, traces are aligned with circular shifts (``np.roll``), i.e., samples shifted out
            at one end re-enter at the other (the cross correlation itself uses zero padding).
            If True, traces are shifted with zero filling and the coherent sum is cropped to the
            samples covered by all channels. See `NuRadioReco.utilities.trace_utilities.get_coherent_sum`.

        lag_channel_ids: array of int or None (default: None)
            Additional channels for which only the time lags (and correlations) are calculated:
            they are not part of the coherent sum, i.e., do not change its SNR, impulsivity, etc.

        lag_template: str (default: "sum")
            Template onto which the channels of ``lag_channel_ids`` are correlated: "sum" (the coherently-summed
            waveform, aligned with the reference channel) or "reference" (the reference channel).

        time_lag_ref_ch_id: int or None (default: None)
            Fixed channel to which the stored time lags (``coherent_time_lags``, T_fix - T_i) and
            ``time_lag_range`` refer. If it is not the reference channel of the coherent sum, it must be one
            of ``channel_ids``. None: the reference channel of the coherent sum (``ref_selection`` of ``run``,
            which must then be a channel id).
        """
        if lag_template not in ("sum", "reference"):
            raise ValueError(f"lag_template must be 'sum' or 'reference', not {lag_template!r}.")

        self.__coincidence_window_size = coincidence_window_size
        self.__pad_length = pad_length
        self.__channel_ids = channel_ids
        self.__time_lag_range = time_lag_range
        self.__ref_time_window = ref_time_window
        self.__crop_to_overlap = crop_to_overlap
        self.__lag_channel_ids = [ch_id for ch_id in (lag_channel_ids or []) if ch_id not in channel_ids]
        self.__lag_template = lag_template
        self.__time_lag_ref_ch_id = time_lag_ref_ch_id

    def __get_snr(self, channel):
        """
        SNR of a single channel, with the definition used for the coherent sum: key
        'peak_2_peak_amplitude_split_noise_rms' of channelParameters.SNR (as set by the channelSignalReconstructor).
        Taken from the channel if set there, else calculated and stored in the channel.
        """
        key = 'peak_2_peak_amplitude_split_noise_rms'
        snrs = channel.get_parameter(chp.SNR) if channel.has_parameter(chp.SNR) else {}
        if key not in snrs:
            window = int(round(self.__coincidence_window_size * channel.get_sampling_rate()))
            trace = channel.get_trace()
            rms = trace_utilities.get_split_trace_noise_RMS(trace, segments=4, lowest=2)
            snrs[key] = trace_utilities.get_signal_to_noise_ratio(trace, rms, window_size=window)
            channel.set_parameter(chp.SNR, snrs)
        return snrs[key]

    @register_run()
    def run(self, evt, station, det, ref_selection=0, use_envelope=True, ref_ch_id=None):
        """
        Calculate the SNR of the coherently-summed waveform and add to the station object.

        Parameters
        ----------
        evt, station, det
            Event, Station, Detector
        ref_selection: int or str (default: 0)
            Reference channel of the coherent sum: a channel id, or "max_snr" for the channel of
            ``channel_ids`` with the highest SNR, chosen per event (key ``peak_2_peak_amplitude_split_noise_rms`` of ``channelParameters.SNR``, taken from the channel if set, else calculated and stored there).
            The reference channel is stored as ``coherent_ref_channel_id``. The time lags are stored relative
            to ``time_lag_ref_ch_id`` (see ``begin``).
        use_envelope: bool (default: True)
            If use Hilbert envelopes to find the cross correlation or not
        ref_ch_id: int or None (default: None)
            Deprecated, use ``ref_selection``: if set, it is passed on as ``ref_selection``.
        """
        if ref_ch_id is not None:
            warnings.warn("The argument ref_ch_id of stationCoherentlySummedWaveforms.run is deprecated, "
                          "use ref_selection instead.", DeprecationWarning)
            ref_selection = ref_ch_id

        sum_channels = list(station.iter_channels(use_channels=self.__channel_ids))
        lag_channels = list(station.iter_channels(use_channels=self.__lag_channel_ids))
        sum_ids = [ch.get_id() for ch in sum_channels]

        if ref_selection == "max_snr":
            ref_ch = max(sum_channels, key=self.__get_snr)
        elif isinstance(ref_selection, (int, np.integer)):
            ref_ch = station.get_channel(ref_selection)
        else:
            raise ValueError(f"ref_selection must be a channel id or 'max_snr', not {ref_selection!r}.")

        # Lags are found relative to the reference channel ref of the sum and stored relative to the fixed
        # channel fix_id: lag_i(fix) = T_fix - T_i = lag_i(ref) - lag_fix(ref).
        fix_id = self.__time_lag_ref_ch_id
        if fix_id is None:
            if ref_selection == "max_snr":
                raise ValueError("Set time_lag_ref_ch_id in begin if the reference channel is chosen per event.")
            fix_id = ref_ch.get_id()
        if fix_id != ref_ch.get_id() and fix_id not in sum_ids:
            raise ValueError(f"time_lag_ref_ch_id {fix_id} must be one of the summed channels {sum_ids} "
                             f"if it is not the reference channel ({ref_ch.get_id()}) of the coherent sum.")

        ref_trace = ref_ch.get_trace()
        sampling_rate = ref_ch.get_sampling_rate()
        channels = [ch for ch in sum_channels if ch.get_id() != ref_ch.get_id()]

        coincidence_window_size_bins_ref = int(round(self.__coincidence_window_size * sampling_rate))
        if coincidence_window_size_bins_ref < 2:
            logger.warning(f"Coincidence window size of {coincidence_window_size_bins_ref} samples is too small for channel {ref_ch.get_id()}.")

        for ch in lag_channels:
            if ch.get_sampling_rate() != sampling_rate or ch.get_number_of_samples() != len(ref_trace):
                raise ValueError(f"Channel {ch.get_id()} has another sampling rate or number of samples "
                                 f"than the reference channel {ref_ch.get_id()}.")

        # (min, max) in units of time relative to the fixed channel, or None, per channel id
        if isinstance(self.__time_lag_range, dict):
            allowed = {ch.get_id() for ch in sum_channels + lag_channels} - {fix_id}
            unknown = set(self.__time_lag_range) - allowed
            if unknown:
                raise ValueError(f"time_lag_range has channels {sorted(unknown)} which are not aligned "
                                 f"onto the fixed channel {fix_id} (channel_ids, lag_channel_ids).")
            time_lag_ranges = dict(self.__time_lag_range)
        else:
            time_lag_ranges = {ch.get_id(): self.__time_lag_range for ch in sum_channels + lag_channels}

        def to_samples(time_range, offset=0):
            # (min, max) in samples, shifted by offset (in samples)
            if time_range is None:
                return None
            return tuple(int(round(t * sampling_rate)) + offset for t in time_range)

        ref_window_samples = None
        if self.__ref_time_window is not None:
            ref_times = ref_ch.get_times() - ref_ch.get_trace_start_time()
            idxs = np.nonzero((ref_times >= self.__ref_time_window[0]) & (ref_times < self.__ref_time_window[1]))[0]
            if not len(idxs):
                raise ValueError(
                    f"ref_time_window {self.__ref_time_window} does not overlap with the trace of "
                    f"channel {ref_ch.get_id()} ({ref_times[0]:.1f} - {ref_times[-1]:.1f} ns).")
            ref_window_samples = (idxs[0], idxs[-1] + 1)

        # Lag of the fixed channel relative to the reference channel, from the allowed lags of the
        # reference channel relative to the fixed one (lag_fix(ref) = -lag_ref(fix)). All other ranges
        # are shifted by it: lag_i(ref) = lag_i(fix) + lag_fix(ref).
        fix_lag = 0
        range_samples = {ch_id: to_samples(time_range) for ch_id, time_range in time_lag_ranges.items()}
        if ref_ch.get_id() != fix_id:
            ref_range = range_samples.get(ref_ch.get_id())
            (fix_lag,), _ = trace_utilities.get_coherent_lags(
                [station.get_channel(fix_id).get_trace()], ref_trace, use_envelope,
                lag_range=None if ref_range is None else (-ref_range[1], -ref_range[0]),
                ref_window=ref_window_samples)
            range_samples[fix_id] = (0, 0)  # the sum uses the lag found above for the fixed channel

        def shifted_range(ch_id, offset):
            lag_range = range_samples.get(ch_id)
            return None if lag_range is None else (lag_range[0] + offset, lag_range[1] + offset)

        sum_trace, lags_samples, correlations = trace_utilities.get_coherent_sum(
            [ch.get_trace() for ch in channels], ref_trace, use_envelope,
            lag_range=[shifted_range(ch.get_id(), fix_lag) for ch in channels],
            ref_window=ref_window_samples, crop_to_overlap=self.__crop_to_overlap,
            return_lags=True, return_correlations=True)

        lag_lags_samples, lag_correlations = np.zeros(0, dtype=int), np.zeros(0)
        if lag_channels:
            # The template's sample 0 is the reference channel's sample start
            if self.__lag_template == "sum":
                template = sum_trace
                start = max(0, lags_samples.max()) if self.__crop_to_overlap else 0
            else:
                template, start = ref_trace, 0
            template_window = None
            if ref_window_samples is not None:
                template_window = tuple(int(np.clip(i - start, 0, len(template))) for i in ref_window_samples)
            lag_lags_samples, lag_correlations = trace_utilities.get_coherent_lags(
                [ch.get_trace() for ch in lag_channels], template, use_envelope,
                lag_range=[shifted_range(ch.get_id(), fix_lag - start) for ch in lag_channels],
                ref_window=template_window)
            lag_lags_samples = lag_lags_samples + start

        time_lags = {ref_ch.get_id(): -fix_lag / sampling_rate}
        time_lags.update({ch.get_id(): (lag - fix_lag) / sampling_rate for ch, lag in
                          zip(channels + lag_channels, np.concatenate([lags_samples, lag_lags_samples]))})
        time_lags[fix_id] = 0.
        coherent_correlations = {ch.get_id(): float(corr) for ch, corr in
                                 zip(channels + lag_channels, np.concatenate([correlations, lag_correlations]))}
        rms = trace_utilities.get_split_trace_noise_RMS(sum_trace, segments=4, lowest=2)
        snr = trace_utilities.get_signal_to_noise_ratio(sum_trace, rms, window_size=coincidence_window_size_bins_ref)
        impulsivity = trace_utilities.get_impulsivity(sum_trace)
        entropy = trace_utilities.get_entropy(sum_trace)
        kurtosis = trace_utilities.get_kurtosis(sum_trace)

        station.set_parameter(stpRNOG.coherent_snr, snr)
        station.set_parameter(stpRNOG.coherent_impulsivity, impulsivity)
        station.set_parameter(stpRNOG.coherent_entropy, entropy)
        station.set_parameter(stpRNOG.coherent_kurtosis, kurtosis)
        station.set_parameter(stpRNOG.coherent_time_lags, time_lags)
        station.set_parameter(stpRNOG.coherent_correlations, coherent_correlations)
        station.set_parameter(stpRNOG.coherent_ref_channel_id, ref_ch.get_id())

    def end(self):
        """(Unused)"""
        pass

    def coherent_sum_step_by_step(self, station):
        """ Plot the four original waveforms before any alignment and save """

        matplotlib.rcParams.update({"font.size": 20})
        plt.figure(figsize = (10, 6))
        for channel in station.iter_channels(use_channels = self.__channel_ids):
            plt.plot(channel.get_trace(), label = f'Original wf[{channel.get_id()}]')
        plt.title('Original Waveforms')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.savefig('original_waveforms.png')
        plt.show()

        # Iterate over each channel as a potential reference
        for ref_ch in station.iter_channels(use_channels = self.__channel_ids):
            fig, axs = plt.subplots(2, 2, figsize = (12, 10))  # Create 2x2 grid for each step with a single reference
            fig.suptitle(f'Coherent Sum Steps with Reference: wf[{ref_ch.get_id()}]', fontsize = 16)

            sum_chan = np.pad(ref_ch.get_trace(), self.__pad_length, mode = 'constant')  # Pad reference waveform
            channels = [ch for ch in station.iter_channels(use_channels = self.__channel_ids) if ch.get_id() != ref_ch.get_id()]  # Exclude the reference channel

            # Initialize the coherent sum with the padded reference waveform
            current_sum = sum_chan.copy()

            for step, ch in enumerate(channels):
                ax = axs[step // 2, step % 2]  # Select subplot for each step

                # Perform full-range cross-correlation to find the necessary shift
                cor = signal.correlate(current_sum[self.__pad_length:-self.__pad_length], ch.get_trace(), mode = 'full')
                shift = np.argmax(cor) - (len(ch.get_trace()) - 1)  # Calculate shift based on maximum correlation

                # Pad and shift the waveform to align with the reference sum
                padded_wf = np.pad(ch.get_trace(), self.__pad_length, mode = 'constant')
                aligned_wf = np.roll(padded_wf, shift)

                # Plot the current coherent sum before adding the next waveform
                ax.plot(current_sum[self.__pad_length:-self.__pad_length], label = 'Current Coherent Sum', linestyle = '--')
                ax.plot(aligned_wf[self.__pad_length:-self.__pad_length], label = f'Next wf[{ch.get_id()}]', alpha = 0.7)

                # Add the aligned waveform to the current sum
                current_sum += aligned_wf

                # Customize plot
                ax.legend(loc = 'upper right')
                ax.set_xlabel('Sample')
                ax.set_ylabel('Amplitude')
                ax.set_title(f'Step {step + 1}: Adding wf[{ch.get_id()}]')

            # Save each coherent summing figure for the current reference
            plt.tight_layout(rect = [0, 0, 1, 0.96])
            plt.savefig(f'coherent_sum_steps_reference_{ref_ch.get_id()}.png')
            plt.show()
