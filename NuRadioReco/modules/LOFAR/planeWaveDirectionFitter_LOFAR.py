import logging
import numpy as np
import matplotlib.pyplot as plt
import radiotools.helper as hp

from scipy import constants
from scipy.signal import hilbert

from NuRadioReco.utilities import fft
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import stationParameters, channelParameters, showerParameters
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.modules.LOFAR.beamforming_utilities import mini_beamformer

lightspeed = constants.c / 1.0003 * (units.m / units.s)
halfpi = np.pi / 2.


def normalize(arr):
    return arr / np.abs(arr)

# adapted from pycrtools.modules.tasks.directionfitplanewave and NuRadioReco.modules.LOFAR.beamformingDirectionFitter_LOFAR

class planeWaveDirectionFitter:
    """
    Fits the direction per station using timing differences of channels under the assumption of an incoming plane wave.
    """

    def __init__(self):
        self.logger = logging.getLogger("NuRadioReco.planeWaveDirectionFitter")

        self.__cr_snr = None


    def begin(self, max_iter=10, cr_snr=3, min_amp=None, rmsfactor=2.0, ignoreNonHorizontalArray=True, window_size=800, debug=False, logger_level=logging.WARNING):
        """
        Set the parameters for the plane wave fit.

        Parameters
        ----------
        max_iter : int, default=10
            The maximum number of iterations to use during the fitting procedure.
        cr_snr : float, default=3
            The minimum SNR a channel should have to be considered having a cosmic ray signal. Ignored if min_amp is not None.
        min_amp : float, default=0.001
            The minimum amplitude a channel should have to be considered having a cosmic ray signal. Set to None if you want to use the SNR instead.
        rmsfactor : float, default=2.0
            How many sigma (times RMS) above the average can a delay deviate from the expected timelag (from latest fit iteration) before it is considered bad and removed as outlier.
        ignoreNonHorizontalArray : bool, default=True
            Set to True when you know the array is non-horizontal (z > 0.5) but want to use the horizontal approximation anyway. Recommended to set to True.
        window_size : int, default=800
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
        self.__ignoreNonHorizontalArray = ignoreNonHorizontalArray
        self.__window_size = window_size
        self.__debug = debug
        self.__logger_level = logger_level
        self.logger.setLevel(logger_level)

    def _signal_windows_polarisation(self, station, channel_positions, channel_ids_per_pol, zenith, azimuth):
        """
        Considers the channel groups given by `channel_ids_per_pol` one by one and beamforms the traces
        in the direction of `stationPulseFinder.direction_cartesian`. It then calculates the maximum of the
        amplitude envelope, and saves the corresponding index with the indices for pulse finding in a tuple,
        which it returns. From stationPulseFinder.py.

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
        direction_cartesian = hp.spherical_to_cartesian(
            zenith, azimuth
        )


        # Assume all the channels have the same frequency content and sampling rate
        frequencies = station.get_channel(channel_ids_per_pol[0][0]).get_frequencies()
        sampling_rate = station.get_channel(channel_ids_per_pol[0][0]).get_sampling_rate()

        values_per_pol = []

        for i, channel_ids in enumerate(channel_ids_per_pol):
            all_spectra = np.array([station.get_channel(channel).get_frequency_spectrum() for channel in channel_ids])
            beamed_fft = mini_beamformer(all_spectra, frequencies, channel_positions, direction_cartesian)
            beamed_timeseries = fft.freq2time(beamed_fft, sampling_rate, n=station.get_channel(channel_ids[0]).get_trace().shape[0])

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

        return int(dominant), int(window_start), int(window_end)
    

    def _get_timelags(self, station, channel_ids_per_pol, positions, zenith, azimuth):
        """
        Get timing differences between signals in antennas with respect to some reference antenna (here the first one).

        Parameters
        ----------
        station : Station object
            The station for which to get the time lags.
        """
        # Get the dominant polarisation and the pulse window
        dominant_pol, pulse_window_start, pulse_window_end = self._signal_windows_polarisation(
            station=station,
            channel_positions=positions,
            channel_ids_per_pol=channel_ids_per_pol,
            zenith=zenith,
            azimuth=azimuth
        )
        # Collect the traces
        traces = [station.get_channel(id).get_trace() for id in channel_ids_per_pol[dominant_pol]]
        times = station.get_channel(channel_ids_per_pol[dominant_pol][0]).get_times()

        # Determine the signal time
        indices_max_trace = []
        for trace in traces:
            trace[:pulse_window_start] = 0
            trace[pulse_window_end:] = 0
            indices_max_trace.append(np.argmax(np.abs(trace)))
        timelags = np.array([times[index] for i, index in enumerate(indices_max_trace)])
        timelags -= timelags[0] # get timelags wrt 1st antenna

        return timelags - timelags[0]


    def _directionForHorizontalArray(self, positions:np.ndarray, times:np.ndarray, ignoreZCoordinate=False):
        """
        --- adapted from pycrtools.modules.scrfind ---
        Given N antenna positions, and (pulse) arrival times for each antenna,
        get a direction of arrival (az, el) assuming a source at infinity (plane wave).

        Here, we find the direction assuming all antennas are placed in the z=0 plane.
        If all antennas are co-planar, the best-fitting solution can be found using a 2D-linear fit.
        We find the best-fitting A and B in:

        .. math::

            t = A x + B y + C

        where t is the array of times; x and y are arrays of coordinates of the antennas.
        The C is the overall time offset in the data, that has to be subtracted out.
        The optimal value of C has to be determined in the fit process (it's not just the average time, nor the time at antenna 0).

        This is done using :mod:`numpy.linalg.lstsq`.

        The (az, el) follows from:

        .. math::

            A = \cos(\mathrm{el}) \cos(\mathrm{az})

            B = \cos(\mathrm{el}) \sin(\mathrm{az})

        
        Parameters
        ----------
        positions : np.ndarray
            Positions (x,y,z) of the antennas (shape: (N_antennas, 3))
        times : array, float
            Pulse arrival times for all antennas

        Returns
        -------
        (az, el) : in radians, and seconds-squared.

        """

        # make x, y arrays out of the input position array
    #    N = len(positions)
        x = positions[:,0]
        y = positions[:,1]

        # now a crude test for nonzero z-input, |z| > 0.5
        z = positions[:,2]
        if not ignoreZCoordinate and max(abs(z)) > 0.5:
            raise ValueError("Input values of z are nonzero ( > 0.5) !")
            return (-1, -1)

        M = np.vstack([x, y, np.ones(len(x))]).T  # says the linalg.lstsq doc

        (A, B, C) = np.linalg.lstsq(M, lightspeed * (times), rcond=None)[0]

        el = np.arccos(np.sqrt(A * A + B * B))
        az = halfpi - np.arctan2(-B, -A)  # note minus sign as we want the direction of the _incoming_ vector (from the sky, not towards it)
        # note: Changed to az = 90_deg - phi
        return (az, el)


    def _timeDelaysFromDirection(self, positions, direction):
        """
        --- adapted from pycrtools.modules.scrfind ---
        Get time delays for antennas at given position for a given direction.
        Time delays come out as an np-array.

        Required arguments:

        =========== =================================================
        Parameter   Description
        =========== =================================================
        *positions* ``(np-array x1, y1, z1, x2, y2, z2, ...)``
        *direction* (az, el) in radians.
        =========== =================================================
        """
        # convert position array into shape used by original implementation:
        positions = np.copy(positions).flatten()
        n = int(len(positions) / 3)
        phi = np.pi/2 - direction[0]  # warning, 90 degree? -- Changed to az = 90_deg - phi
        theta = np.pi/2 - direction[1]  # theta as in standard spherical coords, while el=90 means zenith...

        cartesianDirection = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        timeDelays = np.zeros(n)
        for i in range(n):
            thisPosition = positions[3 * i:3 * (i + 1)]
            timeDelays[i] = - (1 / lightspeed) * np.dot(cartesianDirection, thisPosition)  # note the minus sign! Signal vector points down from the sky.

        return timeDelays

    
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
                continue
            
            # get LORA inital guess for the direction
            zenith = event.get_hybrid_information().get_hybrid_shower("LORA").get_parameter(showerParameters.zenith)
            azimuth = event.get_hybrid_information().get_hybrid_shower("LORA").get_parameter(showerParameters.azimuth)

            # Get all group IDs which are still present in the station
            station_channel_group_ids = set([channel.get_group_id() for channel in station.iter_channels()])

            # collect the positions of 'good' antennas
            position_array = []
            good_antennas = []
            for group_id in station_channel_group_ids:
                channels = [channel for channel in station.iter_channel_group(group_id)]

                good_amp = False
                good_snr = False
                # Only use channels with acceptable amplitude (if desired)
                if self.__min_amp is not None:
                    for channel in channels:
                        if np.max(np.abs(channel.get_trace())) >= self.__min_amp:
                            good_amp = True
                else: # Only use channels with acceptable SNR
                    for channel in channels:
                        if channel.get_parameter(channelParameters.SNR) > self.__cr_snr:
                            good_snr = True

                if good_snr or good_amp:
                    position_array.append( 
                        detector.get_absolute_position(station.get_id()) + 
                        detector.get_relative_position(station.get_id(), channels[0].get_id())
                    ) # positions are the same for every polarization, array of [easting, northing, altitude] ([x, y, z])

                    good_antennas.append((channels[0].get_id(), channels[1].get_id()))

            station.set_parameter(stationParameters.zenith, zenith)
            station.set_parameter(stationParameters.azimuth, azimuth)
            
            good_antennas = np.array(good_antennas, dtype=object)
            mask_good_antennas = np.full(good_antennas.shape[0], True)
            num_good_antennas = good_antennas.shape[0]
            position_array = np.array(position_array)

            # iteratively do the plane wave fit and remove outliers (controlled by rmsfactor) until the number of good antennas remains constant
            niter = 0
            fit_failed = False

            while niter < self.__max_iter: # TODO: maybe add additional condition?
                niter += 1
                # if fit only remains with three antennas (or less) it should not be trusted as it always has a solution (fails)
                if num_good_antennas < 4:
                    self.logger.warning(f"Only {num_good_antennas:d} good antennas remaining!")
                    self.logger.error(f"Too few good antennas for direction fit!")
                    fit_failed = True
                    break
                
                # update arrays to use only previously found "good" antennas:
                position_array = position_array[mask_good_antennas]
                good_antennas = good_antennas[mask_good_antennas]
                mask_good_antennas = np.full(good_antennas.shape[0], True)

                # get timelags
                channel_ids_per_pol = [[channel_group[0] for channel_group in good_antennas], [channel_group[1] for channel_group in good_antennas]]
                times = self._get_timelags(station, channel_ids_per_pol, position_array, zenith, azimuth)

                goodpositions = position_array
                goodtimes = times


                (az, el) = self._directionForHorizontalArray(goodpositions, goodtimes, self.__ignoreNonHorizontalArray)

                if np.isnan(el) or np.isnan(az):
                    self.logger.warning('Plane wave fit returns NaN. Setting elevation to 0.0')
                    el = np.deg2rad(40)  
                    fit_failed = True
                else:
                    fit_failed = False

                # get residuals
                expectedDelays = self._timeDelaysFromDirection(goodpositions, (az, el))
                expectedDelays -= expectedDelays[0] #get delays wrt 1st antenna
                
                residual_delays = goodtimes  - expectedDelays


                if fit_failed:
                    bins = int((residual_delays.max()-residual_delays.min())*lightspeed/(position_array[:,0].max()-position_array[:,0].min()))
                    if bins < 1:
                        bins = 1
                    hist, edges = np.histogram(residual_delays,bins=bins)

                    max_time = np.argmax(hist)
                    self.logger.info(f"histogram filled: {hist}")
                    self.logger.info(f"edges: {edges}")
                    # fix for first and last bin
                    self.logger.info(f"maximum at: {max_time}")
                    try:
                        upper = edges[max_time+2]
                    except:
                        upper = edges[edges.shape[0]-1]
                        self.logger.info(f"upper exception")
                    try:
                        lower = edges[max_time]

                    except:
                        self.logger.info(f"lower exception")
                        lower = edges[0]

                    self.logger.info(f"selecting between lower {lower} and upper {upper}")
                    mask_good_antennas = (residual_delays > lower) & (residual_delays < upper)
                else:
                    # remove > k-sigma outliers and iterate
                    spread = np.std(residual_delays)
                    k = self.__rmsfactor
                    mask_good_antennas = abs(residual_delays - np.mean(residual_delays)) < k * spread
                    # gives subset of 'good_antennas' that is 'good' after this iteration
                
                self.logger.info(f"station {station.get_id()}:")
                self.logger.info(f"iteration {niter:d}:")
                self.logger.info(f'az = {np.rad2deg(az):.3f}, el = {np.rad2deg(el):.3f}')
                self.logger.info(f'number of good antennas = {num_good_antennas:d}')
                
                azimuth = np.mod(90 * units.deg - np.rad2deg(az) * units.deg, 360 * units.deg)
                zenith = np.mod(90 * units.deg - np.rad2deg(el) * units.deg, 360 * units.deg)

                # debug plots:
                if self.__debug:
                    fig = plt.figure(figsize=(15, 5))
                    ax1 = fig.add_subplot(131)
                    scatter1 = ax1.scatter(position_array[:,0], position_array[:,1], c=times)
                    ax1.set_title("Time delays, measured")
                    cbar1 = fig.colorbar(scatter1)
                    ax2 = fig.add_subplot(132)
                    scatter2 = ax2.scatter(position_array[:,0], position_array[:,1], c=expectedDelays)
                    ax2.set_title("Time delays, expected")
                    cbar2 = fig.colorbar(scatter2)
                    ax3 = fig.add_subplot(133)
                    scatter3 = ax3.scatter(position_array[:,0], position_array[:,1], c=residual_delays)
                    ax3.set_title("Time delays, residual")
                    cbar3 = fig.colorbar(scatter3, label='time [ns]')
                    fig.suptitle(f"Station {station.get_id()}")
                    for ax in [ax1, ax2, ax3]:
                        ax.set_xlabel("Easting [m]")
                    ax1.set_ylabel("Northing [m]")
                    plt.show()
                    plt.close(fig)

                # Bookkeeping
                station.set_parameter(stationParameters.zenith, zenith)
                station.set_parameter(stationParameters.azimuth, azimuth)

                
                # if the next iteration has the same number of good antenneas the while loop will be terminated
                if len(good_antennas[mask_good_antennas]) == num_good_antennas:
                    break
                else:
                    num_good_antennas = len(good_antennas[mask_good_antennas])
                

            azimuth = np.mod(90 * units.deg - np.rad2deg(az) * units.deg, 360 * units.deg)
            zenith = np.mod(90 * units.deg - np.rad2deg(el) * units.deg, 360 * units.deg)

            self.logger.info(f"Azimuth (counterclockwise wrt to East) and zenith for station CS{station.get_id():03d}:")
            self.logger.info(azimuth / units.deg, zenith / units.deg)

            self.logger.info(f"Azimuth (wrt to North) and elevation for station CS{station.get_id():03d}:")
            self.logger.info((90 - azimuth / units.deg, 90 - zenith / units.deg))

            station.set_parameter(stationParameters.cr_zenith, zenith)
            station.set_parameter(stationParameters.cr_azimuth, azimuth)
                


    def end(self):
        pass