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
        channel, det
            Channel, Detector
        stored_noise: bool
            Calculates noise from pre-computed forced triggers
        rms_stage: string
            See functionality of det.get_noise_RMS

        Returns
        -------
        SNR: dict
            dictionary of various SNR parameters
        RMS: float
            noise root mean square of a channel
        """

        trace = channel.get_trace()
        times = channel.get_times() - channel.get_trace_start_time()

        if self.__signal_window_start is not None:
            signal_window_start = self.__signal_window_start
            signal_window_mask = (times > self.__signal_window_start) & (times < self.__signal_window_start + self.__signal_window_length)
        else:
            signal_window_start = times[np.argmax(np.abs(trace))] - .5 * self.__signal_window_length
            signal_window_mask = (times > signal_window_start) & (times < signal_window_start + self.__signal_window_length)
        if self.__noise_window_start is not None and self.__noise_window_length is not None:
            noise_window_mask = (times > self.__noise_window_start) & (times < self.__noise_window_start + self.__noise_window_length)
            noise_window_length = self.__noise_window_length
        else:
            noise_window_mask = ~signal_window_mask
            noise_window_length = len(trace[noise_window_mask]) / channel.get_sampling_rate()

        # Various definitions
        noise_int = np.sum(np.square(trace[noise_window_mask]))
        if(noise_window_length > 0):
            noise_int *= (self.__signal_window_length) / float(noise_window_length)
        else:
            logger.warning(f"Noise window length is zero. This likely indicates that the tracelength is too small. Noise quantities can not be calcualted.")

        if stored_noise:
            # we use the RMS from forced triggers
            noise_rms = det.get_noise_RMS(station_id, channel.get_id(), stage=rms_stage)
        else:
            noise_rms = np.sqrt(np.mean(np.square(trace[noise_window_mask])))

        if self.__debug:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(times[signal_window_mask], np.square(trace[signal_window_mask]))
            plt.plot(times[noise_window_mask], np.square(trace[noise_window_mask]), c = 'k', label = 'noise')
            plt.xlabel("Times [ns]")
            plt.ylabel("Power")
            plt.legend()

        # Calculating SNR - signal to noise ratio
        snr = {}
        if noise_rms == 0 or noise_int == 0:
            logger.info("RMS of noise is zero, calculating an snr is not useful. All snrs are set to infinity.")
            snr['peak_2_peak_amplitude'] = np.inf
            snr['peak_amplitude'] = np.inf
            snr['integrated_power'] = np.inf
        else:

            snr['integrated_power'] = np.sum(np.square(trace[signal_window_mask])) - noise_int
            if snr['integrated_power'] <= 0:
                logger.debug("Integrated signal {0} smaller than noise {1}, power snr 0".format(snr['integrated_power'], noise_int))
                snr['integrated_power'] = 0.
            else:
                snr['integrated_power'] /= (noise_int)
                snr['integrated_power'] = np.sqrt(snr['integrated_power'])

            # calculate amplitude values
            snr['peak_2_peak_amplitude'] = np.max(trace[signal_window_mask]) - np.min(trace[signal_window_mask])
            snr['peak_2_peak_amplitude'] /= noise_rms
            snr['peak_2_peak_amplitude'] /= 2

            snr['peak_amplitude'] = np.max(np.abs(trace[signal_window_mask])) / noise_rms

        # Calculate peak to peak voltage SNR using the RMS of the split waveform
        coincidence_window_size_bins = int(round(self.__coincidence_window_size * channel.get_sampling_rate()))
        if coincidence_window_size_bins < 2:
            logger.warning(f"Coincidence window size of {coincidence_window_size_bins} samples is too small for channel {channel.get_id()}.")

        noise_rms = trace_utilities.get_split_trace_noise_RMS(channel.get_trace(), segments=4, lowest=2)
        # only calculate when noise_rms is not zero (can happen in noiseless simulations)
        if noise_rms != 0:
            snr['peak_2_peak_amplitude_split_noise_rms'] = (
                np.amax(trace_utilities.peak_to_peak_amplitudes(channel.get_trace(), coincidence_window_size_bins)) / (2 * noise_rms))

        if self.__debug:
            plt.figure()
            plt.plot(times, trace)
            plt.fill_between(times, 1.1 * np.max(trace), 1.1 * np.min(trace), where=noise_window_mask, color='k', alpha=.2, label='noise window')
            plt.fill_between(times, 1.1 * np.max(trace), 1.1 * np.min(trace), where=signal_window_mask, color='r', alpha=.2, label='signal window')
            plt.legend()
            plt.show()

        return snr, noise_rms


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
        for channel in station.iter_channels():
            normalized_wf = channel.get_trace() / np.std(channel.get_trace())
            coincidence_window_size_bins = int(round(self.__coincidence_window_size * channel.get_sampling_rate()))
            if coincidence_window_size_bins < 2:
                logger.warning(f"Coincidence window size of {coincidence_window_size_bins} samples is too small for channel {channel.get_id()}.")

            thismax = np.amax(
                trace_utilities.peak_to_peak_amplitudes(normalized_wf, coincidence_window_size_bins))

            if thismax > maxaval:
                maxaval = thismax

        return maxaval


    @register_run()
    def run(self, evt, station, det, stored_noise = False, rms_stage = 'amp'):
        """
        Parameters
        ----------
        evt, station, det
            Event, Station, Detector
        stored_noise: bool
            Calculates noise from pre-computed forced triggers
        rms_stage: string
            See functionality of det.get_noise_RMS
        """

        t = time.time()
        max_amplitude_station = 0
        for channel in station.iter_channels():
            times = channel.get_times()
            trace = channel.get_trace()
            h = trace_utilities.get_hilbert_envelope(trace)
            max_amplitude = np.max(np.abs(trace))

            logger.info(f"Event {evt.get_run_number()}.{evt.get_id()}, station.channel "
                        f"{station.get_id()}. {channel.get_id()}: max amp = "
                        f"{max_amplitude:.6g} max amp env {h.max():.6g}")

            if logger.level >= logging.DEBUG:
                logger.debug(", ".join([f"{x:.6g}" for x in trace]))

            channel[chp.signal_time] = times[np.argmax(h)]
            max_amplitude_station = max(max_amplitude_station, max_amplitude)
            channel[chp.maximum_amplitude] = max_amplitude
            channel[chp.maximum_amplitude_envelope] = h.max()
            channel[chp.P2P_amplitude] = np.max(trace) - np.min(trace)

            # Calculate impulsivity of the signal
            channel[chp.impulsivity] = trace_utilities.get_impulsivity(trace)


            # Use noise precalculated from forced triggers
            signal_to_noise, noise_rms = self.get_SNR(
                station.get_id(), channel, det, stored_noise=stored_noise, rms_stage=rms_stage)

            channel[chp.SNR] = signal_to_noise
            channel[chp.noise_rms] = noise_rms
            channel[chp.root_power_ratio] = trace_utilities.get_root_power_ratio(trace, times, noise_rms)
            channel[chp.entropy] = trace_utilities.get_entropy(trace)
            channel[chp.kurtosis] = trace_utilities.get_kurtosis(trace)

        station[stnp.channels_max_amplitude] = max_amplitude_station
        station[stnp.channels_max_amplitude_norm] = self.get_max_a_norm(station)
        self.__t = time.time() - t

    def end(self):
        from datetime import timedelta
        dt = timedelta(seconds = self.__t)
        logger.info("total time used by this module is {}".format(dt))
        return dt
