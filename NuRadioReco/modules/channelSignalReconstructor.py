import numpy as np
from NuRadioReco.utilities import units
from scipy import signal
import time
import logging
logger = logging.getLogger('channelSignalReconstructor')


class channelSignalReconstructor:
    """
    Calculates basic signal parameters.
    """

    def __init__(self):
        self.__t = 0
        self.__conversion_factor_integrated_signal = 2.65441729 * 1e-3 * 1.e-9 * 6.24150934 * 1e18  # to convert V**2/m**2 * s -> J/m**2 -> eV/m**2
        self.__signal_window_start = 40 * units.ns
        self.__signal_window_stop = 100 * units.ns

    def begin(self):
        pass

    def run(self, evt, station, det, debug=False, rms_stage='amp'):
        t = time.time()
        max_amplitude_station = 0
        channels = station.get_channels()
        for channel in channels:
            times = channel.get_times()
            trace = channel.get_trace()
            h = np.abs(signal.hilbert(trace))
            max_amplitude = h.max()
            max_amplitude_station = max(max_amplitude_station, max_amplitude)
            channel['maximum_amplitude'] = max_amplitude

#             signal_window_mask = (times > self.__signal_window_start) & (times < self.__signal_window_stop)
#             noise_rms = (np.sum(trace[~signal_window_mask] ** 2) / np.sum(~signal_window_mask)) ** 0.5
#             channel['noise_rms'] = noise_rms
#             channel['SNR'] = max_amplitude / noise_rms / self.__SNR_normalization
            # we use the RMS from forced triggers
            RMS = det.get_noise_RMS(station.get_id(), channel.get_id(), stage=rms_stage)
            channel['SNR'] = max_amplitude / RMS
        station['channel_max_amplitude'] = max_amplitude

#             fig, ax = plt.subplots(1, 1)
#             ax.plot(times, trace)
#             ax.set_title("{:.2g} {:.2g} {:.2g} {:.2g}".format(max_amplitude, noise_rms, max_amplitude / noise_rms, trace[~signal_window_mask].std()))
#             plt.show()
#             print("channel {}: amp = {:.2g}, rms = {:.2g}".format(channel_name, channel['maximum_amplitude'], channel['noise_rms']))

#             trace_rolled = np.roll(trace, -np.argmax(np.abs(trace)))
#             ff = channel.get_frequencies()
#             spectrum = channel.get_frequency_spectrum()
#             spectrum_rolled = np.fft.rfft(trace_rolled, norm='ortho') * 2 ** 0.5
#             mask = (ff >= 100 * units.MHz) & (ff <= 500 * units.MHz)
#             fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
#             ax1.plot(times, trace_rolled / units.mV)
# #             ax2.plot(ff[mask] / units.MHz, np.abs(spectrum[mask]))
# #             ax3.plot(ff[mask] / units.MHz, np.rad2deg(np.unwrap(np.angle(spectrum[mask]))))
#             ax2.plot(ff[mask] / units.MHz, np.abs(spectrum_rolled[mask]))
#             ax3.plot(ff[mask] / units.MHz, np.rad2deg(np.unwrap(np.angle(spectrum_rolled[mask]))))
#             plt.tight_layout()
#             plt.show()
        self.__t = time.time() - t

    def end(self):
        from datetime import timedelta
        logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        logger.info("total time used by this module is {}".format(dt))
        return dt
