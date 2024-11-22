import logging
import numpy as np
import radiotools.helper as hp

from scipy.signal import hilbert, resample

from NuRadioReco.utilities import fft
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.framework.parameters import stationParameters, channelParameters, showerParameters
from NuRadioReco.modules.LOFAR.beamforming_utilities import mini_beamformer


def find_snr_of_timeseries(timeseries, sampling_rate=None, window_start=0, window_end=-1, noise_start=0, noise_end=-1,
                           resample_factor=1, full_output=False):
    r"""
    Return the signal-to-noise ratio (SNR) of a given time trace, defined as

    ..math ::
        \frac{max( | Hilbert(timeseries[window]) | )}{ STD( Hilbert(timeseries[noise]) ) }

    The signal window and noise window are controlled through the extra parameters to the function.

    Parameters
    ----------
    timeseries: array-like
        The time trace
    sampling_rate : float
        The sampling rate of the time trace (only needed if full_output=True)
    window_start : int
    window_end : int
        We look for the peak inside the resampled array `timeseries[window_start:window_end]`
    noise_start : int
    noise_end : int
        The array `timeseries[noise_start:noise_end]` is used to calculate the noise level
    resample_factor : int, default=1
        Factor with which the timeseries will be resampled, needs to be integer > 0
    full_output : bool, default=False
        If True, also the peak of the envelope, RMS and signal time are returned

    Returns
    -------
    The SNR of the time trace
    """
    resampled_window = resample(timeseries[window_start:window_end], (window_end - window_start) * resample_factor)
    analytic_signal = hilbert(resampled_window)
    amplitude_envelope = np.abs(analytic_signal)

    peak = np.max(
        amplitude_envelope
    )

    if full_output:
        resampled_max = np.argmax(analytic_signal)
        resampled_max_time = resampled_max / sampling_rate / resample_factor
        window_start_time = window_start / sampling_rate
        signal_time = window_start_time + resampled_max_time

    # (AC) The Hilbert envelope has a mean that is nonzero 
    # the stddev only takes the variations around the mean
    # whereas the (real) RMS takes the square of all values (incl the mean), then sqrt
    # TEST if this SNR definition gives good results! 
    # This is what seems to have been done in PyCRTools. See pulseenvelope.py:233
    # and mMath.cc function hMaxSNR  

    std = np.std(
        np.abs(hilbert(timeseries[noise_start:noise_end]))
    )

    rms = np.sqrt(
       np.mean(
           np.abs(hilbert(timeseries[noise_start:noise_end])) ** 2
       )
    )

    if full_output:
        return peak / std, peak, rms, signal_time

    return peak / std


class stationPulseFinder:
    """
    Look for significant pulses in every station. The module uses beamforming to enhance the sensitivity in
    direction estimated from the LORA particle data. It also identifies the channels which have an SNR good
    enough to use for direction fitting later.
    """

    def __init__(self):
        self.logger = logging.getLogger('NuRadioReco.stationPulseFinder')

        self.__window_size = None
        self.__noise_window_size = None
        self.__snr_cr = None
        self.__min_good_channels = None

        self.direction_cartesian = None  # The zenith and azimuth pointing towards where to beamform.

    def begin(self, window=256, noise_window=10000, cr_snr=6.5, good_channels=6, logger_level=logging.NOTSET):
        """
        Sets the window size to use for pulse finding, as well as the number of samples away from the pulse
        to use for noise measurements. The function also defines what an acceptable SNR is to consider a
        cosmic-ray signal to be in the trace, as well as the number of good channels a station should have
        to be kept for further processing.

        Parameters
        ----------
        window : int, default=256
            Size of the window to look for pulse
        noise_window : int, default=10000
            The trace used for noise characterisation goes from sample 0 to the start of the pulse searching
            window minus this number.
        cr_snr : float, default=3
            The minimum SNR a channel should have to be considered having a CR signal.
        good_channels : int, default=6
            The minimum number of good channels a station should have in order be "triggered".
        logger_level : int, default=logging.NOTSET
            Use this parameter to override the logging level for this module.
        """
        # TODO: find window size used in PyCRTools
        self.__window_size = window
        self.__noise_window_size = noise_window
        self.__snr_cr = cr_snr
        self.__min_good_channels = good_channels

        self.logger.setLevel(logger_level)

    def _signal_windows_polarisation(self, station, channel_positions, channel_ids_per_pol):
        """
        Considers the channel groups given by `channel_ids_per_pol` one by one and beamforms the traces
        in the direction of `stationPulseFinder.direction_cartesian`. It then calculates the maximum of the
        amplitude envelope without resampling the signal, and saves the corresponding index with the indices
        for pulse finding in a tuple, which it returns.

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

        # the first few samples are tapered with half-Hann, which would blow up the SNR
        noise_window_start = 10000
        noise_window_end = noise_window_start + self.__noise_window_size

        for i, channel_ids in enumerate(channel_ids_per_pol):
            all_spectra = np.array([station.get_channel(channel).get_frequency_spectrum() for channel in channel_ids])
            beamed_fft = mini_beamformer(all_spectra, frequencies, channel_positions, self.direction_cartesian)
            beamed_timeseries = fft.freq2time(beamed_fft, sampling_rate,
                                              n=station.get_channel(channel_ids[0]).get_trace().shape[0])

            analytic_signal = hilbert(beamed_timeseries)
            amplitude_envelope = np.abs(analytic_signal)
            signal_window_start = int(
                np.argmax(amplitude_envelope) - self.__window_size / 2
            )
            signal_window_end = int(
                np.argmax(amplitude_envelope) + self.__window_size / 2
            )

            snr = find_snr_of_timeseries(beamed_timeseries,
                                         window_start=signal_window_start, window_end=signal_window_end,
                                         noise_start=noise_window_start, noise_end=noise_window_end)

            if snr > self.__snr_cr:
                station.set_parameter(stationParameters.triggered, True)
            else:
                station.set_parameter(stationParameters.triggered, False)

            values_per_pol.append([snr, signal_window_start, signal_window_end])
            # SNR is technically better than just the max(envelope) as a measure for strongest polarization
        values_per_pol = np.asarray(values_per_pol)
        dominant = np.argmax(values_per_pol[:, 0])
        window_start, window_end = values_per_pol[dominant][1:]

        # Save window of strongest polarisation in all channels
        for channel in station.iter_channels():
            channel.set_parameter(channelParameters.signal_regions, [int(window_start), int(window_end)])
            channel.set_parameter(channelParameters.noise_regions, [int(noise_window_start), int(noise_window_end)])

        return dominant

    def _find_good_channels(self, station):
        """
        Loop over all channels in the station and return an array which contains booleans indicating whether
        the channel at that index has an SNR higher than the minimum required one
        (set in the `stationPulseFinder.begin()` function).

        Parameters
        ----------
        station : Station object
            The station to process
        """
        good_channels = []
        for channel in station.iter_channels():
            signal_window = channel.get_parameter(channelParameters.signal_regions)
            noise_window = channel.get_parameter(channelParameters.noise_regions)

            self.logger.debug(f'Channel {channel.get_id()}: looking for signal in indices {signal_window}')
            self.logger.debug(f'Channel {channel.get_id()}: using {noise_window} as noise trace')

            snr, peak, rms, signal_time = find_snr_of_timeseries(channel.get_trace(),
                                                    sampling_rate=channel.get_sampling_rate(),
                                                    window_start=signal_window[0], window_end=signal_window[1],
                                                    noise_start=noise_window[0], noise_end=noise_window[1],
                                                    resample_factor=16, full_output=True)

            channel.set_parameter(channelParameters.SNR, snr)
            channel.set_parameter(channelParameters.noise_rms, rms)
            channel.set_parameter(channelParameters.signal_time, signal_time)
            channel.set_parameter(channelParameters.maximum_amplitude_envelope, peak)
            channel.set_parameter(channelParameters.maximum_amplitude, np.max(channel.get_trace()))

            if snr > self.__snr_cr:
                good_channels.append(channel.get_id())

        return good_channels

    def _check_station_triggered(self, station):
        """
        Check if the station has been triggered by a radio signal. This functions verifies there is a significant
        pulse in the radio signal after beamforming, and also checks if there enough channels (i.e. LOFAR dipoles)
        that contain a good signal.

        Parameters
        ----------
        station : Station object
            The station to analyse
        """
        if station.get_parameter(stationParameters.triggered):
            good_channels_station = self._find_good_channels(
                station
            )

            self.logger.debug(f'Station {station.get_id()} has {len(good_channels_station)} good antennas')
            if len(good_channels_station) < self.__min_good_channels:
                self.logger.warning(f'Station {station.get_id()} has only {len(good_channels_station)} antennas '
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
                if channel.get_id() == channel.get_group_id():
                    station_even_list.append(channel.get_id())
                else:
                    station_odd_list.append(channel.get_id())

            # Find the antenna positions by only looking at the channels from a given polarisation
            position_array = [
                # detector.get_absolute_position(station_id) +    
                # only use the relative position since the absolute position would introduce a time shift 
                # in the beamformed timeseries which would lead to a time shift in the signal window.
                detector.get_relative_position(station_id, channel_id)
                for channel_id in station_even_list
            ]
            position_array = np.asarray(position_array)

            # Find polarisation with max envelope amplitude and calculate pulse search window from it
            dominant_pol = self._signal_windows_polarisation(
                station, position_array, [station_even_list, station_odd_list]
            )

            # Save the antenna orientation which contains the strongest pulse
            if dominant_pol == 0:
                dominant_orientation = detector.get_antenna_orientation(station_id, station_even_list[0])
            elif dominant_pol == 1:
                dominant_orientation = detector.get_antenna_orientation(station_id, station_odd_list[0])
            else:
                raise ValueError(f"Dominant polarisation {dominant_pol} not recognised")

            station.set_parameter(stationParameters.cr_dominant_polarisation, dominant_orientation)

            # Go over all the channels to check if individual SNR is strong enough
            self._check_station_triggered(
                station
            )

    def end(self):
        pass
