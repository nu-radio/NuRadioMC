import logging
import numpy as np
import radiotools.helper as hp

from scipy.signal import hilbert

from NuRadioReco.utilities import fft
from NuRadioReco.modules.base import module
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.framework.parameters import stationParameters, channelParameters
from NuRadioReco.modules.LOFAR.beamforming_utilities import mini_beamformer


logger = module.setup_logger(level=logging.DEBUG)


def find_snr_of_timeseries(timeseries, window_start=0, window_end=-1, noise_start=0, noise_end=-1):
    """
    Return the signal-to-noise ratio (SNR) of a given time trace. The additional parameters allow to
    select a window where the signal is and a separate window where there is only noise.

    Parameters
    ----------
    timeseries: list or np.ndarray
        The time trace
    window_start : int
    window_end : int
        We look for the peak inside the array `timeseries[window_start:window_end]`
    noise_start : int
    noise_end : int
        The array `timeseries[noise_start:noise_end]` is used to calculate the noise level

    Returns
    -------
    The SNR of the time trace
    """
    analytic_signal = hilbert(timeseries)
    amplitude_envelope = np.abs(analytic_signal)

    rms = np.sqrt(
        np.mean(amplitude_envelope[noise_start:noise_end] ** 2)
    )

    peak = np.max(
        amplitude_envelope[window_start:window_end]
    )

    return peak / rms


class stationPulseFinder:
    def __init__(self):
        self.logger = logging.getLogger('NuRadioReco.stationPulseFinder')

        self.__window_size = None
        self.__noise_away_from_pulse = None
        self.__snr_cr = None
        self.__min_good_channels = None

    def begin(self, window=500, noise_window=10000, cr_snr=3, good_channels=6):
        self.__window_size = window
        self.__noise_away_from_pulse = noise_window
        self.__snr_cr = cr_snr  # TODO: do we need different SNR levels for station and antenna
        self.__min_good_channels = good_channels

    def _signal_windows_polarisation(self, station, channel_positions, polarisation_ids=None):
        if polarisation_ids is None:
            polarisation_ids = [0, 1]

        frequencies = station.get_channel(station.get_channel_ids()[0]).get_frequencies()
        sampling_rate = station.get_channel(station.get_channel_ids()[0]).get_sampling_rate()

        direction_cartesian = hp.spherical_to_cartesian(
            station.get_parameter(stationParameters.zenith),
            station.get_parameter(stationParameters.azimuth)
        )

        values_per_pol = []

        for pol in polarisation_ids:
            all_traces = np.array([channel.get_frequency_spectrum() for channel in station.iter_channel_group(pol)])

            beamed_fft = mini_beamformer(all_traces, frequencies, channel_positions, direction_cartesian)
            beamed_timeseries = fft.freq2time(beamed_fft, sampling_rate, n=all_traces.shape[1])

            analytic_signal = hilbert(beamed_timeseries)
            amplitude_envelope = np.abs(analytic_signal)

            signal_window_start = int(
                np.argmax(amplitude_envelope) - self.__window_size / 2
            )
            signal_window_end = int(
                np.argmax(amplitude_envelope) + self.__window_size / 2
            )

            # find dominant polarization
            values_per_pol.append([np.max(amplitude_envelope), signal_window_start, signal_window_end])

        return np.asarray(values_per_pol)

    def _station_has_cr(self, station, channel_positions, signal_window=None, noise_window=None, polarisation_ids=None):
        if polarisation_ids is None:
            polarisation_ids = [0, 1]
        if signal_window is None:
            signal_window = [0, -1]
        if noise_window is None:
            noise_window = [0, -1]

        frequencies = station.get_channel(station.get_channel_ids()[0]).get_frequencies()
        sampling_rate = station.get_channel(station.get_channel_ids()[0]).get_sampling_rate()

        direction_cartesian = hp.spherical_to_cartesian(
            station.get_parameter(stationParameters.zenith),
            station.get_parameter(stationParameters.azimuth)
        )

        for pol in polarisation_ids:
            all_traces = np.array([channel.get_frequency_spectrum() for channel in station.iter_channel_group(pol)])

            beamed_fft = mini_beamformer(all_traces, frequencies, channel_positions, direction_cartesian)
            beamed_timeseries = fft.freq2time(beamed_fft, sampling_rate, n=all_traces.shape[1])

            snr = find_snr_of_timeseries(beamed_timeseries,
                                         window_start=signal_window[0], window_end=signal_window[1],
                                         noise_start=noise_window[0], noise_end=noise_window[1])
            if snr > self.__snr_cr:
                station.set_parameter(stationParameters.triggered, True)
                break  # no need to check second polarisation if CR found

    def _find_good_channels(self, station, signal_window=None, noise_window=None):
        if signal_window is None:
            signal_window = [0, -1]
        if noise_window is None:
            noise_window = [0, -1]

        good_channels = []
        for channel in station.iter_channels():
            snr = find_snr_of_timeseries(channel.get_trace(),
                                         window_start=signal_window[0], window_end=signal_window[1],
                                         noise_start=noise_window[0], noise_end=noise_window[1])
            if snr > self.__snr_cr:
                channel.set_parameter(channelParameters.SNR, snr)
                good_channels.append(channel.get_id())

        return good_channels

    @register_run()
    def run(self, event, detector):
        """
        Run a beamformer on all stations in `event` to search for significant pulses.

        Parameters
        ----------
        event : Event object
            The event on which to apply the pulse finding.
        detector : Detector object
            The detector related to the event.
        """
        for station in event.get_stations():
            position_array = [
                detector.get_absolute_position(station.get_id()) +
                detector.get_relative_position(station.get_id(), channel.get_id())
                for channel in station.iter_channel_group(0)
            ]  # the position are the same for every polarisation
            position_array = np.asarray(position_array)

            # FIXME: hardcoded polarisation values
            values_per_pol = self._signal_windows_polarisation(station, position_array, polarisation_ids=[0, 1])

            dominant_pol = np.argmax(values_per_pol[:, 0])
            pulse_window_start, pulse_window_end = values_per_pol[dominant_pol][1:]

            station.set_parameter(stationParameters.cr_dominant_polarisation, dominant_pol)

            signal_window = [int(pulse_window_start), int(pulse_window_end)]
            noise_window = [0, int(pulse_window_start - self.__noise_away_from_pulse)]

            self.logger.debug(f'Looking for signal in indices {signal_window}')
            self.logger.debug(f'Using {noise_window} as noise trace')

            self._station_has_cr(station, position_array,
                                 signal_window=signal_window,
                                 noise_window=noise_window
                                 )

            good_channels_station = self._find_good_channels(station,
                                                             signal_window=signal_window,
                                                             noise_window=noise_window
                                                             )

            if len(good_channels_station) < self.__min_good_channels:
                self.logger.error(f'There are only {len(good_channels_station)} antennas '
                                  f'with an SNR higher than {self.__snr_cr}, while there '
                                  f'are at least {self.__min_good_channels} required')
                station.set_parameter(stationParameters.triggered, False)  # stop from further processing

    def end(self):
        pass
