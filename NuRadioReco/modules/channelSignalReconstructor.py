from NuRadioReco.modules.base.module import register_run
import numpy as np
from scipy import signal
import time

from NuRadioReco.utilities import units
from NuRadioReco.utilities import trace_utilities
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import stationParameters as stnp

import logging
logger = logging.getLogger('channelSignalReconstructor')


class channelSignalReconstructor:
    """
    Calculates basic signal parameters.

    """

    def __init__(self):
        self.__t = 0
        self.__conversion_factor_integrated_signal = trace_utilities.conversion_factor_integrated_signal
        self.begin()

    def begin(self, debug=False, signal_start=20 * units.ns, signal_stop=100 * units.ns,
              noise_start=150 * units.ns, noise_stop=350 * units.ns):
        """
        Parameters
        -----------
        signal_start: float
            Start time (relative to the trace start time) of the window in which signal quantities will be calculated, with time units
        debug: bool
            Set module to debug output
        signal_stop: float
            Stop time (relative to the trace start time) of the window in which signal quantities will be calculated, with time units
        noise_start: float
            Start time (relative to the trace start time) of the window in which noise quantities will be calculated, with time units
        noise_stop: float
            Stop time (relative to the trace start time) of the window in which noise quantities will be calculated, with time units
        """
        self.__signal_window_start = signal_start
        self.__signal_window_stop = signal_stop
        self.__noise_window_start = noise_start
        self.__noise_window_stop = noise_stop
        self.__debug = debug

    def get_SNR(self, station_id, channel, det, stored_noise=False, rms_stage=None):
        """
        Parameters
        -----------
        channel, det
            Channel, Detector
        stored_noise: bool
            Calculates noise from pre-computed forced triggers
        rms_stage: string
            See functionality of det.get_noise_RMS

        Returns
        ----------
        SNR: dict
            dictionary of various SNR parameters
        """

        trace = channel.get_trace()
        times = channel.get_times() - channel.get_trace_start_time()

        if (self.__noise_window_start >= self.__noise_window_stop):
            logger.error("Noise cannot end before noise starts")
        if self.__signal_window_start >= self.__signal_window_stop:
            logger.error("Signal cannot end before signal starts")

        noise_window_mask = (times > self.__noise_window_start) & (times < self.__noise_window_stop)
        signal_window_mask = (times > self.__signal_window_start) & (times < self.__signal_window_stop)

        # Various definitions
        noise_int = np.sum(np.square(trace[noise_window_mask]))
        noise_int *= (self.__signal_window_stop - self.__signal_window_start) / \
            float(self.__noise_window_stop - self.__noise_window_start)

        if stored_noise:
            # we use the RMS from forced triggers
            noise_rms = det.get_noise_RMS(station_id, channel.get_id(), stage=rms_stage)
        else:
            noise_rms = np.sqrt(np.mean(np.square(trace[noise_window_mask])))

        if self.__debug:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(times[signal_window_mask], np.square(trace[signal_window_mask]))
            plt.plot(times[noise_window_mask], np.square(trace[noise_window_mask]), c='k', label='noise')
            plt.xlabel("Times [ns]")
            plt.ylabel("Power")
            plt.legend()

        # Calculating SNR
        SNR = {}
        if (noise_rms == 0) or (noise_int == 0):
            logger.info("RMS of noise is zero, calculating an SNR is not useful. All SNRs are set to infinity.")
            SNR['peak_2_peak_amplitude'] = np.infty
            SNR['peak_amplitude'] = np.infty
            SNR['integrated_power'] = np.infty
        else:

            SNR['integrated_power'] = (np.sum(np.square(trace[signal_window_mask])) - noise_int)
            if SNR['integrated_power'] < noise_int:
                logger.debug("Integrated signal {0} smaller than noise {1}, power SNR 0".format(SNR['integrated_power'], noise_int))
                SNR['integrated_power'] = 0.
            else:

                SNR['integrated_power'] /= (noise_int / self.__signal_window_start)
                SNR['integrated_power'] = np.sqrt(SNR['integrated_power'])

            # calculate amplitude values
            SNR['peak_2_peak_amplitude'] = np.max(trace[signal_window_mask]) - np.min(trace[signal_window_mask])
            SNR['peak_2_peak_amplitude'] /= noise_rms
            SNR['peak_2_peak_amplitude'] /= 2

            SNR['peak_amplitude'] = np.max(np.abs(trace[signal_window_mask])) / noise_rms

        # SCNR
        SNR['Seckel_2_noise'] = 5

        if self.__debug:
            plt.figure()
            plt.plot(times, trace)
            plt.axvline(self.__noise_window_start, c='k', label='Noise Window')
            plt.axvline(self.__noise_window_stop, c='k', linestyle='--')
            plt.axvline(self.__signal_window_start, c='r', label='Signal Window')
            plt.axvline(self.__signal_window_stop, c='r', linestyle='--')
            plt.legend()
            plt.show()

        return SNR

    @register_run()
    def run(self, evt, station, det, stored_noise=False, rms_stage='amp'):
        """
        Parameters
        -----------
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
            h = np.abs(signal.hilbert(trace))
            max_amplitude = np.max(np.abs(trace))
            channel[chp.signal_time] = times[np.argmax(h)]
            max_amplitude_station = max(max_amplitude_station, max_amplitude)
            channel[chp.maximum_amplitude] = max_amplitude
            channel[chp.maximum_amplitude_envelope] = h.max()
            channel[chp.P2P_amplitude] = np.max(trace) - np.min(trace)

            # Use noise precalculated from forced triggers
            channel[chp.SNR] = self.get_SNR(station.get_id(), channel, det,
                                            stored_noise=stored_noise, rms_stage=rms_stage)

        station[stnp.channels_max_amplitude] = max_amplitude

        self.__t = time.time() - t

    def end(self):
        from datetime import timedelta
        logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        logger.info("total time used by this module is {}".format(dt))
        return dt
