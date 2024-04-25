import logging
import numpy as np
import matplotlib.pyplot as plt
import radiotools.helper as hp

from scipy import constants
from scipy.signal import hilbert
from scipy.optimize import minimize, fmin_powell, Bounds, least_squares

from NuRadioReco.utilities import fft
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import stationParameters, channelParameters, showerParameters
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.modules.voltageToEfieldConverter import voltageToEfieldConverter

import traceback

lightspeed = constants.c / 1.0003 * units.m / units.s
halfpi = np.pi / 2.

debug = True

# adapted from pycrtools.modules.tasks.directionfitplanewave and NuRadioReco.modules.LOFAR.beamformingDirectionFitter_LOFAR

class planeWaveDirectionFitter:
    """
    Fits the direction per station using timing differences of channels under the assumption of an incoming plane wave.
    """

    def __init__(self):
        self.logger = logging.getLogger("NuRadioReco.planeWaveDirectionFitter")

        self.__cr_snr = None

    def begin(self, max_iter=10, cr_snr=3, rmsfactor=2.0, assumeHorizontalArray=False, ignoreNonHorizontalArray=True, logger_level=logging.WARNING):
        """
        Set the parameters for the plane wave fit.

        Parameters
        ----------
        max_iter : int, default=10
            The maximum number of iterations to use during the fitting procedure.
        cr_snr : float, default=3
            The minimum SNR a channel should have to be considered having a cosmic ray signal.
        rmsfactor : float, default=2.0
            How many sigma (times RMS) above the average can a delay deviate from the expected timelag (from latest fit iteration) before it is considered bad and removed as outlier.
        assumeHorizontalArray: bool, default=False
            Whether a horizontal antenna array is assumed and whether to use the plane wave fit for a horizontal array (z=0).
        ignoreNonHorizontalArray : bool, default=True
            Set to True when you know the array is non-horizontal (z > 0.5) but want to use the horizontal approximation anyway 
        logger_level : int, default=logging.WARNING
            The logging level to use for the module.
        """
        self.__max_iter = max_iter
        self.__cr_snr = cr_snr
        self.__rmsfactor = rmsfactor
        self.__assumeHorizontalArray = assumeHorizontalArray
        self.__ignoreNonHorizontalArray = ignoreNonHorizontalArray
        self.__logger_level = logger_level
        self.logger.setLevel(logger_level)

    def _get_timelags(self, station):
        """
        Get timing differences between signals in antennas with respect to some reference antenna (here the first one).

        Parameters
        ----------
        station : Station object
            The station for which to get the time lags.
        """

        # # Determine dominant polarisation in Efield by looking for strongest signal in 5 randomly selected traces
        # random_traces = np.random.choice(station.get_electric_fields(), size=5)
        # dominant_pol_traces = []
        # for trace in random_traces:
        #     trace_envelope = np.abs(hilbert(trace.get_trace(), axis=0))
        #     dominant_pol_traces.append(np.argmax(np.max(trace_envelope, axis=1)))
        # dominant_pol = np.argmax(np.bincount(dominant_pol_traces))
        # self.logger.debug(f"Dominant polarisation is {dominant_pol}")

        # Collect the Efield traces
        traces = np.array([trace.get_trace() for trace in station.get_electric_fields()])

        times = station.get_electric_fields()[0].get_times()

        # So far, the time, where the hilbert envelope of the trace is maximal is used as the time of the signal. 
        # TODO: add 'better' method for finding the signal pulse time
        indices_max_trace = []#np.array([np.argmax(np.abs(hilbert(trace[i]))) for trace in traces])
        for trace in traces:
            dominant_pol = np.argmax(np.max(np.abs(hilbert(trace, axis=0)), axis=1))
            indices_max_trace.append(np.argmax(np.abs(hilbert(trace[dominant_pol]))))

        timelags = np.array([times[index] for index in indices_max_trace])
        timelags -= timelags[0] # get timelags wrt 1st antenna

        return timelags

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

        if debug:
            print(f"M dim: {M.shape}, times dim: {times.shape}")

        (A, B, C) = np.linalg.lstsq(M, lightspeed * times, rcond=None)[0]

        el = np.arccos(np.sqrt(A * A + B * B))
        az = halfpi - np.arctan2(-B, -A)  # note minus sign as we want the direction of the _incoming_ vector (from the sky, not towards it)
        # note: Changed to az = 90_deg - phi
        return (az, el)

    
    def _directionForNonHorizontalArray(self, positions:np.ndarray, times:np.ndarray, station):
        """
        --- adapted from pycrtools.modules.scrfind, extended to non-horizontal array ---
        Given N antenna positions, and (pulse) arrival times for each antenna,
        get a direction of arrival aimuthz (az) and elevation (el) assuming a source at infinity (plane wave).

        We find the best-fitting az and el using:

        .. math::

            ct = A x + B y + C z + D

        where t is the array of times; x, y and z are arrays of coordinates of the antennas.
        A, B and C are coefficients that are given as follows:

        .. math::

            A = \cos(\mathrm{el}) \cos(\mathrm{az})

            B = \cos(\mathrm{el}) \sin(\mathrm{az})

            C = \sin(\mathrm{el})

        The D is the overall time offset in the data, that has to be subtracted out.
        The optimal value of D has to be determined in the fit process (it's not just the average time, nor the time at antenna 0).

        
        Parameters
        ----------
        positions : np.ndarray
            Positions (x,y,z) of the antennas (shape: (N_antennas, 3))
        times : array, float
            Pulse arrival times for all antennas
        station : station-object
            Station for which to fit direction

        Returns
        -------
        (az, el) : in radians, and seconds-squared.

        """
        # make x, y arrays out of the input position array
        # N = len(positions)


        x = positions[:,0]
        y = positions[:,1]
        z = positions[:,2]

        def func_to_minimize(params, x, y, z, times):
            az = params[0] #% (2 * np.pi)
            el = params[1] #% (2 * np.pi)
            D = params[2]
            A = - np.cos(el) * np.cos(az)
            B = - np.cos(el) * np.sin(az)
            C = - np.sin(el)
            return A * x + B * y + C * z + D - times * lightspeed

        start_zenith = station.get_parameter(stationParameters.zenith) / units.rad
        start_azimuth = station.get_parameter(stationParameters.azimuth) / units.rad

        start_el = np.pi/2 - start_zenith
        start_az_prime = start_azimuth

        output = least_squares(
            fun=func_to_minimize, 
            x0=np.array([start_az_prime, start_el, 10]),
            args=(x, y, z, times),
            method="lm",
            # loss="cauchy",
            # bounds=np.array([(0., 2*np.pi), (0., 2*np.pi), (None, None)]),
            )

        if debug:
            print(f'fit params: {output.x}')

        az_prime = output.x[0]
        el = np.mod(output.x[1], 2*np.pi)

        az = np.mod(np.pi/2 - az_prime, 2. * np.pi)  # note minus sign as we want the direction of the _incoming_ vector (from the sky, not towards it)
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
        phi = halfpi - direction[0]  # warning, 90 degree? -- Changed to az = 90_deg - phi
        theta = halfpi - direction[1]  # theta as in standard spherical coords, while el=90 means zenith...

        cartesianDirection = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        timeDelays = np.zeros(n)
        for i in range(n):
            thisPosition = positions[3 * i:3 * (i + 1)]
            timeDelays[i] = - (1 / lightspeed) * np.dot(cartesianDirection, thisPosition)  # note the minus sign! Signal vector points down from the sky.

        return timeDelays


    def _plane_wave_fit(self, positions, times):
        """ 
        TODO ADD DOCUMENTATION!
        """
        pass
    
    @register_run()
    def run(self, event, detector):
        """ 
        TODO ADD DOCUMENTATION
        """
        converter = voltageToEfieldConverter()
        logging.getLogger('voltageToEfieldConverter').setLevel(self.__logger_level)
        converter.begin()

        for station in event.get_stations():
            if not station.get_parameter(stationParameters.triggered):
                continue

            zenith = event.get_hybrid_information().get_hybrid_shower("LORA").get_parameter(showerParameters.zenith)
            azimuth = event.get_hybrid_information().get_hybrid_shower("LORA").get_parameter(showerParameters.azimuth)

            # Get all group IDs which are still present in the station
            station_channel_group_ids = set([channel.get_group_id() for channel in station.iter_channels()])

            position_array = []
            good_antennas = []
            for group_id in station_channel_group_ids:
                channels = [channel for channel in station.iter_channel_group(group_id)]

                # Only use channels with acceptable SNR
                good_snr = False
                for channel in channels:
                    if channel.get_parameter(channelParameters.SNR) > self.__cr_snr:
                        good_snr = True

                if good_snr:
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


            niter = 0
            fit_failed = False

            while niter < self.__max_iter: # TODO: maybe add additional condition
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
                mask_converter_successful = np.full(good_antennas.shape[0], True)

                # Make sure all the previously calculated Efields are removed
                station.set_electric_fields([])

                # Unfold antenna response for good antennas
                for j, ant in enumerate(good_antennas):
                    try:
                        converter.run(event, station, detector, use_channels=ant) # TODO: proper elegant solutions
                    except:
                        mask_converter_successful[j] = False
                        traceback.print_exc()
                
                # use only those antennas and positions for which the converter did not throw an error:
                good_antennas = good_antennas[mask_converter_successful]
                num_good_antennas = good_antennas.shape[0]
                position_array = position_array[mask_converter_successful]

                if num_good_antennas < 3:
                    self.logger.error(f"Too few antennas made it past the converter!")
                    break

                # get timelags
                times = self._get_timelags(station) # TODO: check if additional masking is needed or if the converter does this automatically
                
                goodpositions = position_array#[mask_converter_successful]
                goodtimes = times#[mask_converter_successful]
                if debug:
                    print(f"num_good_antennas = {num_good_antennas}")
                    print(f"goodpositions.shape = {goodpositions.shape}")
                    print(f"goodtimes.shape = {goodtimes.shape}")

                if self.__assumeHorizontalArray:
                    (az, el) = self._directionForHorizontalArray(goodpositions, goodtimes, self.__ignoreNonHorizontalArray)
                else:
                    (az, el) = self._directionForNonHorizontalArray(goodpositions, goodtimes, station)

                if np.isnan(el) or np.isnan(az):
                    self.logger.warning('Plane wave fit returns NaN. Setting elevation to 0.0')
                    el = np.deg2rad(40)  
                    fit_failed = True
                else:
                    fit_failed = False

                # get residuals
                expectedDelays = self._timeDelaysFromDirection(goodpositions, (az, el))
                expectedDelays -= expectedDelays[0] #get delays wrt 1st antenna
                
                residual_delays = goodtimes - expectedDelays

                if debug:
                    print(f"residual delays: {residual_delays}")

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
                    mask_good_antennas = abs(residual_delays - np.mean(residual_delays)) < k * spread # TODO: check indizes
                    # gives subset of 'good_antennas' that is 'good' after this iteration
                
                self.logger.info(f"station {station.get_id()}:")
                self.logger.info(f"iteration {niter:d}:")
                self.logger.info(f'az = {np.rad2deg(az):.3f}, el = {np.rad2deg(el):.3f}')
                self.logger.info(f'number of good antennas = {num_good_antennas:d}')
                
                # TODO: double-check if following conversions are correct
                # Note: seem to be compatible with beamformingDirectionFitter_LOFAR
                azimuth = np.mod(90 * units.deg - np.rad2deg(az) * units.deg, 360 * units.deg)
                zenith = np.mod(90 * units.deg - np.rad2deg(el) * units.deg, 360 * units.deg)

                # Bookkeeping
                station.set_parameter(stationParameters.zenith, zenith)
                station.set_parameter(stationParameters.azimuth, azimuth)

                if debug:
                    print(f'station {station.get_id()}, iteration {niter}: zenith = {zenith:.3f}, azimuth = {azimuth:.3f}, number of good antennas: {num_good_antennas}')

                # if the next iteration has the same number of good antenneas the while loop will be terminated
                if len(good_antennas[mask_good_antennas]) == num_good_antennas:
                    break
                else:
                    num_good_antennas = len(good_antennas[mask_good_antennas])
                    # indicesOfGoodAntennas = indicesOfGoodAntennas[goodSubset]
                


            # TODO: double-check if following conversions are correct
            # Note: seem to be compatible with beamformingDirectionFitter_LOFAR
            azimuth = np.mod(90 * units.deg - np.rad2deg(az) * units.deg, 360 * units.deg)
            zenith = np.mod(90 * units.deg - np.rad2deg(el) * units.deg, 360 * units.deg)

            self.logger.info(f"Azimuth (counterclockwise wrt to East (hopefully)) and zenith for station CS{station.get_id():03d}:")
            self.logger.info(azimuth / units.deg, zenith / units.deg)

            self.logger.info(f"Azimuth (wrt to North) and elevation for station CS{station.get_id():03d}:")
            self.logger.info((90 - azimuth / units.deg, 90 - zenith / units.deg))

            station.set_parameter(stationParameters.cr_zenith, zenith)
            station.set_parameter(stationParameters.cr_azimuth, azimuth)
                


    def end(self):
        pass