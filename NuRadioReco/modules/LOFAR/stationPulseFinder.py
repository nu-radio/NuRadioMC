import logging
import numpy as np
import radiotools.helper as hp

from scipy.signal import hilbert

from NuRadioReco.utilities import fft
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.framework.parameters import stationParameters, channelParameters, showerParameters
from NuRadioReco.modules.LOFAR.beamforming_utilities import mini_beamformer


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
    """
    Look for significant pulses in every station. The module uses beamforming to enhance the sensitivity in
    direction estimated from the LORA particle data. It also identifies the channels which have an SNR good
    enough to use for direction fitting later.
    """

    def __init__(self):
        self.logger = logging.getLogger('NuRadioReco.stationPulseFinder')

        self.__window_size = None
        self.__noise_away_from_pulse = None
        self.__snr_cr = None
        self.__min_good_channels = None

        self.direction_cartesian = None  # The zenith and azimuth pointing towards where to beamform.

    def begin(self, window=500, noise_window=10000, cr_snr=3, good_channels=6, logger_level=logging.WARNING):
        """
        Sets the window size to use for pulse finding, as well as the number of samples away from the pulse
        to use for noise measurements. The function also defines what an acceptable SNR is to consider a
        cosmic-ray signal to be in the trace, as well as the number of good channels a station should have
        to be kept for further processing.

        Parameters
        ----------
        window : int, default=500
            Size of the window to look for pulse
        noise_window : int, default=10000
            The trace used for noise characterisation goes from sample 0 to the start of the pulse searching
            window minus this number.
        cr_snr : float, default=3
            The minimum SNR a channel should have to be considered having a CR signal.
        good_channels : int, default=6
            The minimum number of good channels a station should have in order be "triggered".
        logger_level : int, default=logging.WARNING
            The logging level to use for the module.
        """
        # TODO: find window size used in PyCRTools
        self.__window_size = window
        self.__noise_away_from_pulse = noise_window
        self.__snr_cr = cr_snr
        self.__min_good_channels = good_channels

        self.logger.setLevel(logger_level)

    def _signal_windows_polarisation(self, station, channel_positions, channel_ids_per_pol):
        """
        Considers the channel groups given by `channel_ids_per_pol` one by one and beamforms the traces
        in the direction of `stationPulseFinder.direction_cartesian`. It then calculates the maximum of the
        amplitude envelope, and saves the corresponding index with the indices for pulse finding in a tuple,
        which it returns.

        Parameters
        ----------
        station : Station object
            The station to analyse.
        channel_positions : np.ndarray
            The array of channels positions, to be extracted from the detector description.
        channel_ids_per_pol : list[list]
            A list of channel IDs grouped per polarisation. The IDs in each sublist will be used together for
            beamforming in the direction given by `beamform_direction`.

        Returns
        -------
        tuple of floats
            * The index in `channel_ids_per_pol` which contains the strongest signal
            * The start index of the pulse window
            * The stop index of the pulse window
        """
        # Assume all the channels have the same frequency content and sampling rate
        frequencies = station.get_channel(channel_ids_per_pol[0][0]).get_frequencies()
        sampling_rate = station.get_channel(channel_ids_per_pol[0][0]).get_sampling_rate()

        values_per_pol = []

        for channel_ids in channel_ids_per_pol:
            all_traces = np.array([station.get_channel(channel).get_frequency_spectrum() for channel in channel_ids])

            beamed_fft = mini_beamformer(all_traces, frequencies, channel_positions, self.direction_cartesian)
            beamed_timeseries = fft.freq2time(beamed_fft, sampling_rate, n=all_traces.shape[1])

            analytic_signal = hilbert(beamed_timeseries)
            amplitude_envelope = np.abs(analytic_signal)

            signal_window_start = int(
                np.argmax(amplitude_envelope) - self.__window_size / 2
            )
            signal_window_end = int(
                np.argmax(amplitude_envelope) + self.__window_size / 2
            )

            values_per_pol.append([np.max(amplitude_envelope), signal_window_start, signal_window_end])

        values_per_pol = np.asarray(values_per_pol)
        dominant = np.argmax(values_per_pol[:, 0])
        window_start, window_end = values_per_pol[dominant][1:]

        return dominant, window_start, window_end

    def _station_has_cr(self, station, channel_positions, channel_ids_per_pol, signal_window=None, noise_window=None):
        """
        Beamform the station towards the direction given by `stationPulseFinder.direction_cartesian`
        and check if there is any significant signal in the trace. If this is the case,
        the `stationParameter.triggered` value is set to `True`.

        Parameters
        ----------
        station : Station object
            The station to process
        channel_positions : np.ndarray
            The array of channels positions, to be extracted from the detector description
        channel_ids_per_pol : list[list]
            A list of channel IDs grouped per polarisation.
        signal_window : array-like, default=[0, -1]
            A list containing the first and last index of the trace where to look for a pulse
        noise_window : array-like, default=[0, -1]
            A list containing the first and last index of the trace to use for noise characterisation
        """
        if signal_window is None:
            signal_window = [0, -1]
        if noise_window is None:
            noise_window = [0, -1]

        frequencies = station.get_channel(channel_ids_per_pol[0][0]).get_frequencies()
        sampling_rate = station.get_channel(channel_ids_per_pol[0][0]).get_sampling_rate()

        for channel_ids in channel_ids_per_pol:
            all_traces = np.array([station.get_channel(channel).get_frequency_spectrum() for channel in channel_ids])

            beamed_fft = mini_beamformer(all_traces, frequencies, channel_positions, self.direction_cartesian)
            beamed_timeseries = fft.freq2time(beamed_fft, sampling_rate, n=all_traces.shape[1])

            snr = find_snr_of_timeseries(beamed_timeseries,
                                         window_start=signal_window[0], window_end=signal_window[1],
                                         noise_start=noise_window[0], noise_end=noise_window[1])
            if snr > self.__snr_cr:
                station.set_parameter(stationParameters.triggered, True)
                return True  # no need to check second polarisation if CR found

        return False

    def _find_good_channels(self, station, signal_window=None, noise_window=None):
        """
        Loop over all channels in the station and return an array which contains booleans indicating whether
        the channel at that index has an SNR higher than the minimum required one
        (set in the `stationPulseFinder.begin()` function).

        Parameters
        ----------
        station : Station object
            The station to process
        signal_window : array-like, default=[0, -1]
            A list containing the first and last index of the trace where to look for a pulse
        noise_window : array-like, default=[0, -1]
            A list containing the first and last index of the trace to use for noise characterisation
        """
        if signal_window is None:
            signal_window = [0, -1]
        if noise_window is None:
            noise_window = [0, -1]

        good_channels = []
        for channel in station.iter_channels():
            snr = find_snr_of_timeseries(channel.get_trace(),
                                         window_start=signal_window[0], window_end=signal_window[1],
                                         noise_start=noise_window[0], noise_end=noise_window[1])

            channel.set_parameter(channelParameters.SNR, snr)
            if snr > self.__snr_cr:
                good_channels.append(channel.get_id())

        return good_channels

    def _check_station_triggered(self, station, channel_positions, channel_ids_per_pol, signal_window, noise_window):
        """
        Check if the station has been triggered by a radio signal. This functions verifies there is a significant
        pulse in the radio signal after beamforming, and also checks if there enough channels (i.e. LOFAR dipoles)
        that contain a good signal.

        Parameters
        ----------
        station : Station object
            The station to analyse
        """
        self.logger.debug(f'Looking for signal in indices {signal_window}')
        self.logger.debug(f'Using {noise_window} as noise trace')

        cr_found = self._station_has_cr(
            station, channel_positions, channel_ids_per_pol, signal_window=signal_window, noise_window=noise_window
        )

        if cr_found:
            good_channels_station = self._find_good_channels(
                station, signal_window=signal_window, noise_window=noise_window
            )

            self.logger.debug(f'Station {station.get_id()} has {len(good_channels_station)} good antennas')
            if len(good_channels_station) < self.__min_good_channels:
                self.logger.warning(f'There are only {len(good_channels_station)} antennas '
                                    f'with an SNR higher than {self.__snr_cr}, while there '
                                    f'are at least {self.__min_good_channels} required')
                station.set_parameter(stationParameters.triggered, False)  # stop from further processing

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
        zenith = event.get_hybrid_information().get_hybrid_shower("LORA").get_parameter(showerParameters.zenith)
        azimuth = event.get_hybrid_information().get_hybrid_shower("LORA").get_parameter(showerParameters.azimuth)

        self.direction_cartesian = hp.spherical_to_cartesian(
            zenith, azimuth
        )

        for station in event.get_stations():
            station_id = station.get_id()

            # Get the channel IDs grouped per polarisation (i.e. dipole orientation)
            # -> take these from Event, cause some might already be thrown out!
            station_even_list = []
            station_odd_list = []
            for channel in station.iter_channels():
                if channel.get_id() == int(channel.get_group_id()[1:]):
                    station_even_list.append(channel.get_id())
                else:
                    station_odd_list.append(channel.get_id())

            # Find the antenna positions by only looking at the channels from a given polarisation
            position_array = [
                detector.get_absolute_position(station_id) +
                detector.get_relative_position(station_id, channel_id)
                for channel_id in station_even_list
            ]
            position_array = np.asarray(position_array)

            ant_same_orientation = [station_even_list, station_odd_list]

            # Find polarisation with max envelope amplitude and calculate pulse search window from it
            dominant_pol, pulse_window_start, pulse_window_end = self._signal_windows_polarisation(
                station, position_array, ant_same_orientation
            )

            # Save the antenna orientation which contains the strongest pulse
            station.set_parameter(stationParameters.cr_dominant_polarisation,
                                  detector.get_antenna_orientation(station_id, ant_same_orientation[dominant_pol][0]))

            # Check if the station has a strong enough signal
            signal_window = [int(pulse_window_start), int(pulse_window_end)]
            noise_window = [0, int(pulse_window_start - self.__noise_away_from_pulse)]

            self._check_station_triggered(station, position_array, ant_same_orientation, signal_window, noise_window)

    def end(self):
        pass
