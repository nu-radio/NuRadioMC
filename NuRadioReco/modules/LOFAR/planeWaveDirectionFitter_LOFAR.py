"""
This module has been adapted from pycrtools.modules.tasks.directionfitplanewave and
NuRadioReco.modules.LOFAR.beamformingDirectionFitter_LOFAR

.. moduleauthor:: Philipp Laub <philipp.laub@fau.de>
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import radiotools.helper as hp

from scipy.signal import resample

from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import stationParameters, channelParameters, showerParameters
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.modules.LOFAR.beamforming_utilities import geometric_delay_far_field, lightspeed


class planeWaveDirectionFitter:
    """
    Fits the direction per station using timing differences of channels under the assumption of an incoming plane wave.
    """

    def __init__(self):
        self.logger = logging.getLogger("NuRadioReco.planeWaveDirectionFitter")

        self.__cr_snr = None
        self.__logger_level = None
        self.__debug = None
        self.__window_size = None
        self.__ignore_non_horizontal_array = None
        self.__rmsfactor = None
        self.__min_amp = None
        self.__max_iter = None

    def begin(self, max_iter=10, cr_snr=3, min_amp=None, rmsfactor=2.0, force_horizontal_array=True, window_size=256,
              debug=False, logger_level=logging.WARNING):
        """
        Set the parameters for the plane wave fit.

        Parameters
        ----------
        max_iter : int, default=10
            The maximum number of iterations to use during the fitting procedure.
        cr_snr : float, default=3
            The minimum SNR a channel should have to be considered having a cosmic ray signal.
            Ignored if min_amp is not None.
        min_amp : float, default=0.001
            The minimum amplitude a channel should have to be considered having a cosmic ray signal.
            Set to None if you want to use the SNR instead.
        rmsfactor : float, default=2.0
            How many sigma (times RMS) above the average can a delay deviate from the expected timelag
            (from latest fit iteration) before it is considered bad and removed as outlier.
        force_horizontal_array : bool, default=True
            Set to True when you know the array is non-horizontal (z > 0.5) but want to use the
            horizontal approximation anyway. Recommended to set to True.
        window_size : int, default=256
            The size of the window to use for the pulse finding.
        debug : bool, default=False
            Set to True to enable debug plots.
        logger_level : int, default=logging.WARNING
            The logging level to use for the module.
        """
        self.__max_iter = max_iter
        self.__cr_snr = cr_snr
        self.__min_amp = min_amp
        self.__rmsfactor = rmsfactor
        self.__ignore_non_horizontal_array = force_horizontal_array
        self.__window_size = window_size
        self.__debug = debug
        self.__logger_level = logger_level
        self.logger.setLevel(logger_level)

    @staticmethod
    def _get_timelags(station, channel_ids_dominant_pol, resample_factor=16):
        """
        Get timing differences between signals in antennas with respect to some reference antenna (the first one
        in the list of ids). The peak is determined using the Hilbert envelope after resampling the trace with
        `resample_factor`.

        Parameters
        ----------
        station : Station object
            The station for which to get the time lags
        channel_ids_dominant_pol : list of int
            The list of channel ids to calculate the time lags for (usually the dominant polarisation)
        resample_factor : int, default=16
            The resample factor to use when calculating the peak

        Returns
        -------
        timelags : np.ndarray
            The timelags (in internal units) for each channel in the list, with respect to the first one
        """
        # Determine the signal time
        timelags = []
        for channel_id in channel_ids_dominant_pol:
            channel = station.get_channel(channel_id)
            pulse_window_start, pulse_window_end = channel.get_parameter(channelParameters.signal_regions)
            resampled_window = resample(channel.get_trace()[pulse_window_start:pulse_window_end],
                                        (pulse_window_end - pulse_window_start) * resample_factor)
            resampled_max = np.argmax(resampled_window)
            resampled_max_time = resampled_max / channel.get_sampling_rate() / resample_factor
            window_start_time = pulse_window_start / channel.get_sampling_rate()
            timelags.append(window_start_time + resampled_max_time)

        timelags -= timelags[0]  # get timelags wrt 1st antenna

        return np.asarray(timelags)

    @staticmethod
    def _direction_horizontal_array(positions: np.ndarray, times: np.ndarray, ignore_z_coordinate=False):
        r"""
        --- adapted from pycrtools.modules.scrfind ---
        Given N antenna positions, and (pulse) arrival times for each antenna,
        get a direction of arrival (azimuth, zenith) assuming a source at infinity (plane wave).

        Here, we find the direction assuming all antennas are placed in the z=0 plane.
        If all antennas are co-planar, the best-fitting solution can be found using a 2D-linear fit.
        We find the best-fitting A and B in:

        .. math::

            t = A x + B y + C

        where t is the array of times; x and y are arrays of coordinates of the antennas.
        The C is the overall time offset in the data, that has to be subtracted out.
        The optimal value of C has to be determined in the fit process (it's not just the average time,
        nor the time at antenna 0).

        This is done using :mod:`numpy.linalg.lstsq`.

        The (azimuth, zenith) follows from:

        .. math::

            A = \sin(\mathrm{zenith}) \cos(\mathrm{azimuth})

            B = \sin(\mathrm{zenith}) \sin(\mathrm{azimuth})

        
        Parameters
        ----------
        positions : np.ndarray
            Positions (x,y,z) of the antennas (shape: (N_antennas, 3))
        times : array, float
            Measured pulse arrival times for all antennas

        Returns
        -------
        zenith : float
            Zenith in the [0, 2pi] interval (given in internal units)
        azimuth : float
            Azimuth in the [0, 2pi] interval (given in internal units)
        """

        # make x, y arrays out of the input position array
        x = positions[:, 0]
        y = positions[:, 1]

        # now a crude test for nonzero z-input, |z| > 0.5
        z = positions[:, 2]
        if not ignore_z_coordinate and max(abs(z)) > 0.5:
            raise ValueError("Input values of z are nonzero ( > 0.5) !")

        M = np.vstack([x, y, np.ones(len(x))]).T  # says the linalg.lstsq doc

        A, B, C = np.linalg.lstsq(M, lightspeed * times, rcond=None)[0]

        zenith = np.arcsin(np.sqrt(A**2 + B**2))  # TODO: this can result in RuntimeWarning - why?
        azimuth = np.arctan2(-B, -A)  # note minus sign as we want the direction of the _incoming_ vector (from the sky, not towards it)
        
        return np.mod(zenith * units.rad, 360 * units.deg), np.mod(azimuth * units.rad, 360 * units.deg)

    @register_run()
    def run(self, event, detector):
        """
        Run the plane wave fit for the given event and detector.

        Parameters
        ----------
        event : Event object
            The event for which to run the plane wave fit.
        detector : Detector object
            The detector for which to run the plane wave fit.
        """
        # converter = voltageToEfieldConverter()
        # logging.getLogger('voltageToEfieldConverter').setLevel(self.__logger_level)
        # converter.begin()

        for station in event.get_stations():
            if not station.get_parameter(stationParameters.triggered):
                self.logger.debug(f"Station CS{station.get_id():03d} did not trigger, skipping...")
                continue
            self.logger.debug(f"Running over station CS{station.get_id():03d}")

            # get LORA initial guess for the direction
            zenith = event.get_hybrid_information().get_hybrid_shower("LORA").get_parameter(showerParameters.zenith)
            azimuth = event.get_hybrid_information().get_hybrid_shower("LORA").get_parameter(showerParameters.azimuth)

            # Get all group IDs which are still present in the station
            station_channel_group_ids = set([channel.get_group_id() for channel in station.iter_channels()])

            # Get the dominant polarisation orientation as calculated by stationPulseFinder
            dominant_orientation = station.get_parameter(stationParameters.cr_dominant_polarisation)

            # Collect the positions of 'good' antennas
            good_channel_pair_ids = np.zeros((len(station_channel_group_ids), 2), dtype=int)
            relative_position_array = np.zeros((len(station_channel_group_ids), 3))
            good_amp_or_snr = np.zeros(len(station_channel_group_ids), dtype=bool)
            for ind, channel_group_id in enumerate(station_channel_group_ids):
                relative_position_array[ind] = detector.get_relative_position(station.get_id(), channel_group_id)
                for channel in station.iter_channel_group(channel_group_id):
                    if np.all(detector.get_antenna_orientation(station.get_id(), channel.get_id()) == dominant_orientation):
                        good_channel_pair_ids[ind, 0] = channel.get_id()
                    else:
                        good_channel_pair_ids[ind, 1] = channel.get_id()

                # Check if dominant channel has acceptable SNR or acceptable amplitude (if desired)
                channel = station.get_channel(good_channel_pair_ids[ind, 0])
                if self.__min_amp is None:
                    if channel.get_parameter(channelParameters.SNR) > self.__cr_snr:
                        good_amp_or_snr[ind] = True
                else:
                    if np.max(np.abs(channel.get_trace())) >= self.__min_amp:
                        good_amp_or_snr[ind] = True

            station.set_parameter(stationParameters.zenith, zenith)
            station.set_parameter(stationParameters.azimuth, azimuth)

            num_good_antennas = np.sum(good_amp_or_snr)
            mask_good_antennas = np.full(num_good_antennas, True)

            good_antennas = good_channel_pair_ids[good_amp_or_snr]
            position_array = relative_position_array[good_amp_or_snr]

            # iteratively do the plane wave fit and remove outliers (controlled by rmsfactor)
            # until the number of good antennas remains constant
            niter = 0

            while niter < self.__max_iter:  # TODO: maybe add additional condition?
                niter += 1
                # if only three antennas (or less) remain, fit should not be trusted as it always has a solution (fails)
                if num_good_antennas < 4:
                    self.logger.warning(f"Only {num_good_antennas:d} good antennas remaining!")
                    self.logger.error(f"Too few good antennas for direction fit!")
                    break

                # update arrays to use only previously found "good" antennas:
                position_array = position_array[mask_good_antennas]
                good_antennas = good_antennas[mask_good_antennas]

                # get time lags from the dominant antennas only
                times = self._get_timelags(station, good_antennas[:, 0])

                goodpositions = position_array
                goodtimes = times

                zenith, azimuth = self._direction_horizontal_array(goodpositions, goodtimes,
                                                                   self.__ignore_non_horizontal_array)

                # get residuals
                expected_delays = geometric_delay_far_field(
                    goodpositions, hp.spherical_to_cartesian(zenith / units.rad, azimuth / units.rad)
                )
                expected_delays -= expected_delays[0]  # get delays wrt 1st antenna

                residual_delays = goodtimes - expected_delays

                if np.isnan(zenith) or np.isnan(azimuth):
                    self.logger.warning('Plane wave fit returns NaN.')
                    bins = int((residual_delays.max() - residual_delays.min()) * lightspeed / (
                            position_array[:, 0].max() - position_array[:, 0].min()))
                    if bins < 1:
                        bins = 1
                    hist, edges = np.histogram(residual_delays, bins=bins)

                    max_time = np.argmax(hist)
                    self.logger.debug(f"histogram filled: {hist}")
                    self.logger.debug(f"edges: {edges}")
                    # fix for first and last bin
                    self.logger.debug(f"maximum at: {max_time}")

                    try:
                        upper = edges[max_time + 2]
                    except IndexError:
                        upper = edges[edges.shape[0] - 1]
                        self.logger.debug(f"upper exception")

                    try:
                        lower = edges[max_time]
                    except IndexError:
                        self.logger.debug(f"lower exception")
                        lower = edges[0]

                    self.logger.debug(f"selecting between lower {lower} and upper {upper}")
                    mask_good_antennas = (residual_delays > lower) & (residual_delays < upper)
                else:
                    # remove > k-sigma outliers and iterate
                    spread = np.std(residual_delays)
                    k = self.__rmsfactor
                    mask_good_antennas = abs(residual_delays - np.mean(residual_delays)) < k * spread
                    # gives subset of 'good_antennas' that is 'good' after this iteration

                self.logger.debug(f"station {station.get_id()}:")
                self.logger.debug(f"iteration {niter:d}:")
                self.logger.debug(f'azimuth = {np.rad2deg(azimuth):.3f}, zenith = {np.rad2deg(zenith):.3f}')
                self.logger.debug(f'number of good antennas = {num_good_antennas:d}')

                # Debug plots
                if self.__debug:
                    import matplotlib as mpl

                    inner = [['times'],
                             ['expected']]
                    outer = [[inner, 'residuals'],
                             ['traces', 'traces']]

                    fig, axd = plt.subplot_mosaic(outer, layout="constrained", figsize=(10, 12))
                    for channel in station.iter_channels(use_channels=good_antennas[:, 0]):
                        axd['traces'].plot(channel.get_trace() / units.mV)

                    # mark the signal window:
                    channel = station.get_channel(good_antennas[0, 0])  # should be all the same
                    pulse_window_start, pulse_window_end = channel.get_parameter(channelParameters.signal_regions)

                    axd['traces'].axvline(pulse_window_start, color='r')
                    axd['traces'].axvline(pulse_window_end, color='r')
                    axd['traces'].set_xlim(pulse_window_start - 500, pulse_window_end + 500)
                    axd['traces'].set_xlabel('Sample index')
                    axd['traces'].set_ylabel('Amplitude [mV]')
                    axd['traces'].set_title(f'Good traces used in iteration {niter}')

                    # Plot the timing residuals
                    datasets = [times, expected_delays]

                    norm1 = mpl.colors.Normalize(vmin=np.min(datasets), vmax=np.max(datasets))
                    norm2 = mpl.colors.Normalize(vmin=np.min(residual_delays), vmax=np.max(residual_delays))
                    cmap1 = mpl.colormaps.get_cmap('viridis')
                    cmap2 = mpl.colormaps.get_cmap('seismic')

                    axd['times'].scatter(position_array[:, 0], position_array[:, 1], c=times,
                                         norm=norm1, cmap=cmap1,
                                         label='Measured')
                    axd['times'].set_title("Time delays")
                    axd['times'].legend()

                    axd['expected'].scatter(position_array[:, 0], position_array[:, 1], c=expected_delays,
                                            norm=norm1, cmap=cmap1,
                                            label='Expected')
                    axd['expected'].legend()

                    axd['residuals'].scatter(position_array[:, 0], position_array[:, 1], c=residual_delays,
                                             norm=norm2, cmap=cmap2)
                    axd['residuals'].set_title("Residual time delays")

                    fig.colorbar(mpl.cm.ScalarMappable(norm=norm1, cmap=cmap1),
                                 ax=axd['expected'], orientation='horizontal',
                                 location='bottom',
                                 label='Time [ns]')

                    fig.colorbar(mpl.cm.ScalarMappable(norm=norm2, cmap=cmap2),
                                 ax=axd['residuals'], orientation='horizontal',
                                 location='bottom',
                                 label='Time [ns]')

                    fig.suptitle(f"Station {station.get_id()}")

                    for ax in ['expected', 'residuals']:
                        axd[ax].set_xlabel("Easting [m]")

                    axd['residuals'].yaxis.set_label_position("right")
                    axd['residuals'].set_ylabel("Northing [m]")
                    axd['residuals'].set_aspect('equal')

                    plt.show()
                    plt.close(fig)

                # Bookkeeping
                station.set_parameter(stationParameters.zenith, zenith)
                station.set_parameter(stationParameters.azimuth, azimuth)

                # if the next iteration has the same number of good antennae the while loop will be terminated
                if len(good_antennas[mask_good_antennas]) == num_good_antennas:
                    break
                else:
                    num_good_antennas = len(good_antennas[mask_good_antennas])

            azimuth = station.get_parameter(stationParameters.azimuth)
            zenith = station.get_parameter(stationParameters.zenith)

            self.logger.status(
                f"Azimuth (counterclockwise wrt to East) and zenith for station CS{station.get_id():03d}:")
            self.logger.status(f"{azimuth / units.deg}, {zenith / units.deg}")

            self.logger.status(f"Azimuth (clockwise wrt to North) and elevation for station CS{station.get_id():03d}:")
            self.logger.status(f"{90 - azimuth / units.deg}, {90 - zenith / units.deg}")

            station.set_parameter(stationParameters.cr_zenith, zenith)
            station.set_parameter(stationParameters.cr_azimuth, azimuth)

    def end(self):
        pass
