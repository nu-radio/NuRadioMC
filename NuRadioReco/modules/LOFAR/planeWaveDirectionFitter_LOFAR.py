"""
This module has been adapted from pycrtools.modules.tasks.directionfitplanewave and
NuRadioReco.modules.LOFAR.beamformingDirectionFitter_LOFAR

.. moduleauthor:: Philipp Laub <philipp.laub@fau.de>
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import radiotools.helper as hp


from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import stationParameters, channelParameters, showerParameters
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.modules.LOFAR.beamforming_utilities import geometric_delay_far_field, lightspeed


def average_direction(event, detector, mode='normal'):
    """
    Calculate the average direction for an event based on the plane wave directions of the individual stations.

    Parameters
    ----------
    event : Event object
        The event for which to calculate the average direction.
    detector : Detector object
        The detector for which to calculate the average direction.
    mode : str, default='normal'
        The mode to use for the calculation. Can be 'normal' (just raw mean) or 'weighted'
        (with number of good antennas as weight per station).

    Returns
    -------
    avg_zenith : float
        The average zenith angle for the event.
    avg_azimuth : float
        The average azimuth angle for the event.
    """
    zeniths = []
    azimuths = []
    num_good_antennas = []
    for station in event.get_stations():
        if station.get_parameter(stationParameters.triggered):
            flagged_channels = station.get_parameter(stationParameters.flagged_channels)
            num_good_antennas.append(
                detector.get_number_of_channels(station.get_id()) - len(flagged_channels)
            )
            zeniths.append(station.get_parameter(stationParameters.cr_zenith))
            azimuths.append(station.get_parameter(stationParameters.cr_azimuth))

    zeniths = np.array(zeniths)
    azimuths = np.array(azimuths)
    num_good_antennas = np.array(num_good_antennas)

    # Calculate the average direction: 
    if mode == 'normal':
        avg_zenith = np.mean(zeniths)
        avg_azimuth = np.mean(azimuths)
    elif mode == 'weighted':
        avg_zenith = np.sum(zeniths * num_good_antennas) / np.sum(num_good_antennas)
        avg_azimuth = np.sum(azimuths * num_good_antennas) / np.sum(num_good_antennas)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return avg_zenith, avg_azimuth


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
        self.__min_number_good_antennas = None

    def begin(self, max_iter=10, cr_snr=6.5, min_amp=None, rmsfactor=2.0, force_horizontal_array=True,
              debug=False, logger_level=logging.NOTSET, min_number_good_antennas=4):
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
        debug : bool, default=False
            Set to True to enable debug plots.
        logger_level : int, default=logging.WARNING
            The logging level to use for the module.
        min_number_good_antennas : int, default=4
            The minimum number of good antennas that should be present in a station to consider it for the fit.
        """
        self.__max_iter = max_iter
        self.__cr_snr = cr_snr
        self.__min_amp = min_amp
        self.__rmsfactor = rmsfactor
        self.__ignore_non_horizontal_array = force_horizontal_array
        self.__debug = debug
        self.__logger_level = logger_level
        self.logger.setLevel(logger_level)
        self.__min_number_good_antennas = min_number_good_antennas

    @staticmethod
    def _get_timelags(station, channel_ids_dominant_pol):
        """
        Get timing differences between signals in antennas with respect to some reference antenna (the first one
        in the list of ids). The peak is determined using the Hilbert envelope after resampling the trace with
        `resample_factor`.

        Parameters
        ----------
        station : Station object
            The station for which to get the time lags
        channel_ids_dominant_pol : list of int
            The list of channel ids to return the time lags for (usually the dominant polarisation)

        Returns
        -------
        timelags : np.ndarray
            The timelags (in internal units) for each channel in the list, with respect to the first one
        """
        # Get the signal time found by stationPulseFinder
        timelags = []
        for channel_id in channel_ids_dominant_pol:
            timelags.append(station.get_channel(channel_id).get_parameter(channelParameters.signal_time))

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
        for station in event.get_stations():
            if not station.get_parameter(stationParameters.triggered):
                self.logger.debug(f"Station CS{station.get_id():03d} did not trigger, skipping...")
                continue
            self.logger.debug(f"Running over station CS{station.get_id():03d}")

            # get LORA initial guess for the direction
            lora_zenith = event.get_hybrid_information().get_hybrid_shower("LORA").get_parameter(showerParameters.zenith)
            lora_azimuth = event.get_hybrid_information().get_hybrid_shower("LORA").get_parameter(showerParameters.azimuth)

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

            num_good_antennas = np.sum(good_amp_or_snr)
            mask_good_antennas = np.full(num_good_antennas, True)

            # the dominant antennas are good_antennas[:, 0]
            good_antennas = good_channel_pair_ids[good_amp_or_snr]
            position_array = relative_position_array[good_amp_or_snr]

            # iteratively do the plane wave fit and remove outliers (controlled by rmsfactor)
            # until the number of good antennas remains constant
            niter = 0
            zenith, azimuth = lora_zenith, lora_azimuth
            while niter < self.__max_iter:  # TODO: maybe add additional condition?
                niter += 1
                # if only three antennas (or less) remain, fit should not be trusted as it always has a solution (fails)
                if num_good_antennas < self.__min_number_good_antennas:
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

                # Debug plots if required
                if self.__debug:
                    self.debug_plots(
                        event, expected_delays, good_antennas, niter, position_array, residual_delays, station, times
                    )

                if np.isnan(zenith) or np.isnan(azimuth):
                    self.logger.error(
                        'Plane wave fit returns NaN. I will try to recover by setting zenith and azimuth '
                        'to the LORA estimate and recalculating the residual delays.'
                    )

                    zenith = lora_zenith
                    azimuth = lora_azimuth

                    expected_delays = geometric_delay_far_field(
                        goodpositions, hp.spherical_to_cartesian(zenith / units.rad, azimuth / units.rad)
                    )
                    expected_delays -= expected_delays[0]
                    residual_delays = goodtimes - expected_delays

                    bins = int(
                        (residual_delays.max() - residual_delays.min()) * lightspeed /
                        (position_array[:, 0].max() - position_array[:, 0].min())
                    )
                    hist, edges = np.histogram(residual_delays, bins=max(bins, 1))

                    max_time = np.argmax(hist)
                    self.logger.debug(f"histogram filled: {hist}")
                    self.logger.debug(f"edges: {edges}")
                    self.logger.debug(f"maximum at: {max_time}")

                    upper = edges[min(max_time + 2, len(edges) - 1)]
                    lower = edges[max(max_time - 1, 0)]

                    self.logger.debug(f"Selecting between lower {lower} and upper {upper}")
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

                # if the next iteration has the same number of good antennae the while loop will be terminated
                if len(good_antennas[mask_good_antennas]) == num_good_antennas:
                    break
                else:
                    num_good_antennas = len(good_antennas[mask_good_antennas])

            self.logger.status(
                f"Azimuth (counterclockwise wrt to East) and zenith for station CS{station.get_id():03d}:"
            )
            self.logger.status(f"{azimuth / units.deg}, {zenith / units.deg}")

            self.logger.status(
                f"Azimuth (clockwise wrt to North) and elevation for station CS{station.get_id():03d}:"
            )
            self.logger.status(f"{90 - azimuth / units.deg}, {90 - zenith / units.deg}")

            # Set stationParameters.zenith/azimuth because voltageToEfieldConverter uses these to convert
            # NOTE: these can be the LORA direction (in case the fit failed)
            station.set_parameter(stationParameters.zenith, zenith)
            station.set_parameter(stationParameters.azimuth, azimuth)

            # Only set reconstructed direction if it is not identical to the LORA direction
            if not (zenith == lora_zenith and azimuth == lora_azimuth):
                self.logger.info(
                    f"The fit for station CS{station.get_id():03d} seems to have failed."
                    f"I will not set the cr_zenith and cr_azimuth station parameters, but you can"
                    f"still unfold the voltages to electric fields with the LORA direction as this"
                    f"is saved in the zenith and azimuth station parameters."
                )
                station.set_parameter(stationParameters.cr_zenith, zenith)
                station.set_parameter(stationParameters.cr_azimuth, azimuth)

            # flag channels that were not used in the fit
            station_flagged_channels = station.get_parameter(stationParameters.flagged_channels)

            for channel_id in good_channel_pair_ids.flatten().tolist():
                if channel_id not in good_antennas.flatten():
                    # TODO: this flag is not always correct, as channels excluded by SNR are also flagged
                    station_flagged_channels[channel_id].append("planewavefit_timing_outlier")

            station.set_parameter(stationParameters.flagged_channels, station_flagged_channels)

    @staticmethod
    def debug_plots(
            event, expected_delays, good_antennas, niter, position_array, residual_delays, station, times
    ):
        """
        Create debug plots for the plane wave fit.
        """
        planeWaveDirectionFitter.__debug_mosaic(
            event, expected_delays, good_antennas, niter, position_array, residual_delays, station, times
        )
        planeWaveDirectionFitter.__debug_residuals(
            event, good_antennas, residual_delays, station, niter
        )

    @staticmethod
    def __debug_residuals(event, good_antennas, residual_delays, station, niter):
        """
        Show the residuals per antenna and mark SNR
        """
        fig, ax = plt.subplots()

        antenna_SNRs = np.zeros(len(good_antennas))
        for i, antenna in enumerate(good_antennas[:, 0]):
            channel = station.get_channel(antenna)
            antenna_SNRs[i] = channel.get_parameter(channelParameters.SNR)

        plt.scatter(np.arange(len(residual_delays)), residual_delays, marker='o', c=antenna_SNRs)

        # add colorbar
        plt.colorbar(label='SNR')

        ax.set_xlabel('Antenna')
        ax.set_ylabel('Residual time [ns]')
        ax.set_title(f'Residuals for station {station.get_id()}')

        fig.savefig(
            f"pipeline_planewavefit_residuals_CS{station.get_id():03d}_iteration{niter}_{event.get_id()}.png",
            dpi=250, bbox_inches='tight'
        )
        # fig.savefig(
        #     f"pipeline_planewavefit_residuals_CS{station.get_id():03d}_iteration{niter}_{event.get_id()}.svg",
        #     dpi=250, bbox_inches='tight'
        # )
        plt.close(fig)

    @staticmethod
    def __debug_mosaic(event, expected_delays, good_antennas, niter, position_array, residual_delays, station, times):
        """
        Plot the timings, as well as the residuals and the traces used for the fit.
        """
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

        fig.savefig(
            f"pipeline_planewavefit_debug_CS{station.get_id():03d}_iteration{niter}_{event.get_id()}.png",
            dpi=250, bbox_inches='tight'
        )
        # fig.savefig(
        #     f"pipeline_planewavefit_debug_CS{station.get_id():03d}_iteration{niter}_{event.get_id()}.svg",
        #     dpi=250, bbox_inches='tight'
        # )

        plt.close(fig)

    def end(self):
        pass
