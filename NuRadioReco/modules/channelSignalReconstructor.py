from NuRadioReco.modules.base.module import register_run
import numpy as np
from scipy import signal
import time

from NuRadioReco.utilities import units
from NuRadioReco.utilities import trace_utilities
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import stationParameters as stnp

import logging
logger = logging.getLogger('NuRadioReco.channelSignalReconstructor')


class channelSignalReconstructor:
    """
    Calculates basic signal parameters.

    """

    def __init__(self, log_level=logging.NOTSET):
        self.__t = 0
        logger.setLevel(log_level)
        self.__conversion_factor_integrated_signal = trace_utilities.conversion_factor_integrated_signal
        self.__signal_window_start = None
        self.__signal_window_stop = None
        self.__noise_window_start = None
        self.__noise_window_stop = None
        self.__signal_window_length = None
        self.__noise_window_length = None
        self.__debug = None
        self.begin()

    def begin(
        self,
        debug = False,
        signal_window_start = None,
        signal_window_length = 120 * units.ns,
        noise_window_start = None,
        noise_window_length = None,
        coincidence_window_size = 6 * units.ns
    ):
        """
        Parameters
        ----------
        debug: bool
            Set module to debug output
        signal_window_start: float or None
            Start time (relative to the trace start time) of the window in which signal quantities will be calculated, with time units
            If None is passed as a parameter, the signal window is laid around the trace maximum
        signal_window_length: float
            Length of the signal window, with time units
        noise_window_start: float or None
            Start time (relative to the trace start time) of the window in which noise quantities will be calculated, with time units
            If noise_window_start or noise_window_length are None, the noise window is the part of the trace outside the signal window
        noise_window_length: float or None
            Length of the noise window, with time units
            If noise_window_start or noise_window_length are None, the noise window is the part of the trace outside the signal window
        coincidence_window_size : float (default: 6ns)
            Window size used for calculating the maximum peak to peak amplitude used for the max_a_norm variable
        """
        self.__signal_window_start = signal_window_start
        self.__signal_window_length = signal_window_length
        self.__noise_window_start = noise_window_start
        self.__noise_window_length = noise_window_length
        self.__coincidence_window_size = coincidence_window_size
        self.__debug = debug

    def get_SNR(self, station_id, channel, det, stored_noise = False, rms_stage = None):
        """
        Parameters
        ----------
        station_id: int
            ID of the station
        channel : Channel
        det : Detector
        stored_noise: bool, optional
            Calculates noise from pre-computed forced triggers. Default ``False``
        rms_stage: string, optional
            See functionality of det.get_noise_RMS. Only relevant if ``stored_noise==True``

        Returns
        -------
        SNR: dict
            dictionary of various SNR parameters
        RMS: float
            noise root mean square of a channel
        """

        trace = channel.get_trace()
        times = channel.get_times() - channel.get_trace_start_time()
        stored_noise_rms = self._get_stored_noise_rms(station_id, [channel], det, rms_stage) if stored_noise else None
        snrs, noise_rms, _ = self._get_SNR(trace[None], times[None], channel.get_sampling_rate(), stored_noise_rms)
        return snrs[0], noise_rms[0]

    @staticmethod
    def _get_stored_noise_rms(station_id, channels, det, rms_stage):
        """ Noise RMS of each channel from pre-computed forced triggers """
        return np.array([det.get_noise_RMS(station_id, channel.get_id(), stage=rms_stage) for channel in channels])

    def _get_SNR(self, traces, times, sampling_rate, stored_noise_rms=None):
        """
        Same as `get_SNR` but for an array of traces (one per row) with a common sampling rate
        and additionally returns the maximum local peak to peak amplitude of each trace
        (in the coincidence window), which is also needed for ``max_a_norm``.

        Parameters
        ----------
        traces: 2D array of floats
            The traces, one per row
        times: 2D array of floats
            The times of each trace relative to its trace start time
        sampling_rate: float
            Sampling rate of the traces
        stored_noise_rms: array of floats, optional
            Noise RMS of each trace from pre-computed forced triggers. If None, the RMS is calculated from the noise window.

        Returns
        -------
        snrs: list of dict
            SNR dictionary for each trace
        noise_rms: array of floats
            noise root mean square (from the split trace) of each trace
        max_p2p: array of floats
            maximum local peak to peak amplitude of each trace
        """
        if self.__signal_window_start is not None:
            signal_window_start = self.__signal_window_start
        else:
            signal_window_start = np.take_along_axis(times, np.argmax(np.abs(traces), axis=-1)[:, None], axis=-1) - .5 * self.__signal_window_length
        signal_window_mask = (times > signal_window_start) & (times < signal_window_start + self.__signal_window_length)

        if self.__noise_window_start is not None and self.__noise_window_length is not None:
            noise_window_mask = (times > self.__noise_window_start) & (times < self.__noise_window_start + self.__noise_window_length)
            noise_window_length = self.__noise_window_length
        else:
            noise_window_mask = ~signal_window_mask
            noise_window_length = np.sum(noise_window_mask, axis=-1) / sampling_rate

        # Various definitions
        noise_int = np.sum(np.square(traces), axis=-1, where=noise_window_mask)
        has_noise_window = noise_window_length > 0
        if not np.all(has_noise_window):
            logger.warning("Noise window length is zero. This likely indicates that the tracelength is too small. Noise quantities can not be calcualted.")
        with np.errstate(divide='ignore'):
            noise_int = noise_int * np.where(has_noise_window, np.divide(self.__signal_window_length, noise_window_length), 1)

        if stored_noise_rms is not None:
            noise_rms = stored_noise_rms
        else:
            noise_rms = np.sqrt(np.mean(np.square(traces), axis=-1, where=noise_window_mask))

        # Calculating SNR - signal to noise ratio
        # (where the noise is zero, calculating an snr is not useful and all snrs are set to infinity)
        no_noise = (noise_rms == 0) | (noise_int == 0)
        if np.any(no_noise):
            logger.warning("RMS of noise is zero, calculating an snr is not useful. All snrs are set to infinity.")
        with np.errstate(divide='ignore', invalid='ignore'):
            signal_power = np.sum(np.square(traces), axis=-1, where=signal_window_mask) - noise_int
            integrated_power = np.sqrt(np.maximum(signal_power, 0) / noise_int)

            traces_max = np.max(traces, axis=-1, where=signal_window_mask, initial=-np.inf)
            traces_min = np.min(traces, axis=-1, where=signal_window_mask, initial=np.inf)
            peak_2_peak_amplitude = (traces_max - traces_min) / noise_rms / 2
            peak_amplitude = np.max(np.abs(traces), axis=-1, where=signal_window_mask, initial=0) / noise_rms

        snr_values = {
            'integrated_power': np.where(no_noise, np.inf, integrated_power),
            'peak_2_peak_amplitude': np.where(no_noise, np.inf, peak_2_peak_amplitude),
            'peak_amplitude': np.where(no_noise, np.inf, peak_amplitude)
        }

        # Calculate peak to peak voltage SNR using the RMS of the split waveform
        coincidence_window_size_bins = int(round(self.__coincidence_window_size * sampling_rate))
        if coincidence_window_size_bins < 2:
            logger.warning(f"Coincidence window size of {coincidence_window_size_bins} samples is too small for sampling rate {sampling_rate}.")

        noise_rms = trace_utilities.get_split_trace_noise_RMS(traces, segments=4, lowest=2)
        max_p2p = np.amax(trace_utilities.peak_to_peak_amplitudes(traces, coincidence_window_size_bins), axis=-1)

        snrs = []
        for i in range(len(traces)):
            snr = {key: values[i] for key, values in snr_values.items()}
            # only calculate when noise_rms is not zero (can happen in noiseless simulations)
            if noise_rms[i] != 0:
                snr['peak_2_peak_amplitude_split_noise_rms'] = max_p2p[i] / (2 * noise_rms[i])
            snrs.append(snr)

        if self.__debug:
            self._plot_debug(times, traces, signal_window_mask, noise_window_mask)

        return snrs, noise_rms, max_p2p

    @staticmethod
    def _plot_debug(times, traces, signal_window_mask, noise_window_mask):
        import matplotlib.pyplot as plt
        for t, trace, signal_mask, noise_mask in zip(times, traces, signal_window_mask, noise_window_mask):
            plt.figure()
            plt.plot(t[signal_mask], np.square(trace[signal_mask]))
            plt.plot(t[noise_mask], np.square(trace[noise_mask]), c = 'k', label = 'noise')
            plt.xlabel("Times [ns]")
            plt.ylabel("Power")
            plt.legend()

            plt.figure()
            plt.plot(t, trace)
            plt.fill_between(t, 1.1 * np.max(trace), 1.1 * np.min(trace), where=noise_mask, color='k', alpha=.2, label='noise window')
            plt.fill_between(t, 1.1 * np.max(trace), 1.1 * np.min(trace), where=signal_mask, color='r', alpha=.2, label='signal window')
            plt.legend()
            plt.show()

    @staticmethod
    def _group_channels(station):
        """ Groups the channels of a station by number of samples and sampling rate, so each group can be stacked into a 2D array """
        groups = {}
        for channel in station.iter_channels():
            groups.setdefault((channel.get_number_of_samples(), channel.get_sampling_rate()), []).append(channel)

        return groups.values()

    def get_max_a_norm(self, station):
        """
        Calculate the maximum peak to peak amplitude of the signal normalized by the noise level over all channels in a station.

        Parameters
        ----------
        station : Station
            The station object containing the channels.

        Returns
        -------
        maxaval : float
            The maximum peak to peak amplitude of the signal normalized by the noise level over all channels in the station.
        """
        maxaval = 0
        for channels in self._group_channels(station):
            traces = np.stack([channel.get_trace() for channel in channels])
            # skip empty traces (e.g. channels without signal in noiseless simulations), their normalized
            # trace would be NaN and a single NaN would mask the maximum of the whole group
            stds = np.std(traces, axis=-1, keepdims=True)
            has_signal = stds[:, 0] > 0
            if not np.any(has_signal):
                continue
            normalized_traces = traces[has_signal] / stds[has_signal]
            coincidence_window_size_bins = int(round(self.__coincidence_window_size * channels[0].get_sampling_rate()))
            if coincidence_window_size_bins < 2:
                logger.warning(f"Coincidence window size of {coincidence_window_size_bins} samples is too small for channels with sampling rate {channels[0].get_sampling_rate()}.")

            maxaval = max(maxaval, np.amax(trace_utilities.peak_to_peak_amplitudes(normalized_traces, coincidence_window_size_bins)))

        return maxaval


    @register_run()
    def run(self, evt, station, det, stored_noise = False, rms_stage = 'amp', snr_only=False):
        """
        Compute signal properties for all channels in a station.

        Parameters
        ----------
        evt : Event
        station : Station
        det : Detector
        stored_noise: bool
            Calculates noise from pre-computed forced triggers
        rms_stage: string, optional
            See functionality of det.get_noise_RMS. Only relevant if ``stored_noise==True``.
            Default "amp", i.e. use RMS voltage after amplifier stage.
        snr_only : bool, default ``False``
            If ``True``, only compute SNR statistics (faster);
            otherwise (default), also compute additional signal properties
            such as impulsivity, kurtosis and entropy.
        """

        t = time.time()
        max_amplitude_station = 0
        max_a_norm = 0
        for channels in self._group_channels(station):
            # all channels of a group are processed together as rows of 2D arrays
            traces = np.stack([channel.get_trace() for channel in channels])
            times = np.stack([channel.get_times() for channel in channels])
            envelopes = trace_utilities.get_hilbert_envelope(traces)
            max_amplitudes = np.max(np.abs(traces), axis=-1)
            max_envelopes = np.max(envelopes, axis=-1)
            signal_times = np.take_along_axis(times, np.argmax(envelopes, axis=-1)[:, None], axis=-1)[:, 0]

            # Use noise precalculated from forced triggers
            stored_noise_rms = self._get_stored_noise_rms(station.get_id(), channels, det, rms_stage) if stored_noise else None
            start_times = np.array([channel.get_trace_start_time() for channel in channels])
            snrs, noise_rms, max_p2p = self._get_SNR(
                traces, times - start_times[:, None], channels[0].get_sampling_rate(), stored_noise_rms)

            max_amplitude_station = max(max_amplitude_station, np.max(max_amplitudes))
            # as in get_max_a_norm, re-using the peak to peak amplitude from the SNR calculation
            # (empty traces are skipped, as a single NaN would mask the maximum of the whole group)
            stds = np.std(traces, axis=-1)
            has_signal = stds > 0
            if np.any(has_signal):
                max_a_norm = max(max_a_norm, np.max(max_p2p[has_signal] / stds[has_signal]))

            if not snr_only:
                # Calculate additional properties of the signal
                impulsivity = trace_utilities.get_impulsivity(traces, envelope=envelopes)
                root_power_ratio = trace_utilities.get_root_power_ratio(traces, times[0], noise_rms)
                entropy = np.array([trace_utilities.get_entropy(trace) for trace in traces])
                kurtosis = trace_utilities.get_kurtosis(traces)

            for i, channel in enumerate(channels):
                logger.debug("Event %s.%s, station.channel %s. %s: max amp = %.6g max amp env %.6g",
                            evt.get_run_number(), evt.get_id(), station.get_id(), channel.get_id(),
                            max_amplitudes[i], max_envelopes[i])

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(", ".join([f"{x:.6g}" for x in traces[i]]))

                channel[chp.signal_time] = signal_times[i]
                channel[chp.maximum_amplitude] = max_amplitudes[i]
                channel[chp.maximum_amplitude_envelope] = max_envelopes[i]
                channel[chp.P2P_amplitude] = np.max(traces[i]) - np.min(traces[i])
                channel[chp.SNR] = snrs[i]
                channel[chp.noise_rms] = noise_rms[i]

                if not snr_only:
                    channel[chp.impulsivity] = impulsivity[i]
                    channel[chp.root_power_ratio] = root_power_ratio[i]
                    channel[chp.entropy] = entropy[i]
                    channel[chp.kurtosis] = kurtosis[i]

        station[stnp.channels_max_amplitude] = max_amplitude_station
        station[stnp.channels_max_amplitude_norm] = max_a_norm
        self.__t = time.time() - t

    def end(self):
        from datetime import timedelta
        dt = timedelta(seconds = self.__t)
        logger.info("total time used by this module is {}".format(dt))
        return dt
