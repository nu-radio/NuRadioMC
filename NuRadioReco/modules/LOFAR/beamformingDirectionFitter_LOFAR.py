import logging
import numpy as np
import matplotlib.pyplot as plt
import radiotools.helper as hp

from scipy import constants
from scipy.signal import hilbert
from scipy.optimize import minimize, fmin_powell, Bounds

from NuRadioReco.utilities import fft
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import stationParameters, channelParameters, showerParameters
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.modules.voltageToEfieldConverter import voltageToEfieldConverter
from NuRadioReco.modules.LOFAR.beamforming_utilities import beamformer

lightspeed = constants.c * units.m / units.s


def geometric_delays(ant_positions, sky):
    """
    Returns geometric delays in a matrix.

    Parameters
    ----------
    ant_positions : np.ndarray
        The antenna positions to use, formatted as a (nr_of_ant, 3) shaped array.
    sky : np.ndarray
        The unit vector pointing to the arrival direction, in cartesian coordinates.

    Returns
    -------
    delays : np.ndarray
    """
    delays = np.dot(ant_positions, sky)
    delays /= -1 * lightspeed
    return delays


class beamformingDirectionFitter:
    """
    Fits the direction per station using interferometry between all the channels with a good enough signal.
    """

    def __init__(self):
        self.logger = logging.getLogger("NuRadioReco.beamFormingDirectionFitter")

        self.__max_iter = None
        self.__cr_snr = None

    def begin(self, max_iter, cr_snr=3, logger_level=logging.WARNING):
        """
        Set the values for the fitting procedures.

        Parameters
        ----------
        max_iter : int
            The maximum number of iterations to use during the fitting procedure
        cr_snr : float, default=3
            The minimum SNR a channel should have to be considered having a CR signal.
        logger_level : int, default=logging.WARNING
            The logging level to use for the module.
        """
        self.__max_iter = max_iter
        self.__cr_snr = cr_snr

        self.logger.setLevel(logger_level)

    def _direction_fit(self, station, ant_positions):
        """
        Fit the arrival direction by iteratively beamforming the signal and maximising the peak of the time trace.

        Parameters
        ----------
        station : Station object
            The station for which to fit the direction
        ant_positions : np.ndarray, 2D
            The array of antenna positions, to be extracted from the detector description.
        """
        freq = station.get_electric_fields()[0].get_frequencies()

        # Determine dominant polarisation in Efield by looking for strongest signal in 5 randomly selected traces
        random_traces = np.random.choice(station.get_electric_fields(), size=5)
        dominant_pol_traces = []
        for trace in random_traces:
            trace_envelope = np.abs(hilbert(trace.get_trace(), axis=0))
            dominant_pol_traces.append(np.argmax(np.max(trace_envelope, axis=1)))
        dominant_pol = np.argmax(np.bincount(dominant_pol_traces))
        self.logger.debug(f"Dominant polarisation is {dominant_pol}")

        # Collect the Efield traces
        fft_traces = np.array([trace.get_frequency_spectrum()[dominant_pol]
                               for trace in station.get_electric_fields()])

        def negative_beamed_signal(direction):
            theta = direction[0]
            phi = direction[1]
            my_direction_cartesian = hp.spherical_to_cartesian(theta, phi)

            my_delays = geometric_delays(ant_positions, my_direction_cartesian)

            my_out = beamformer(fft_traces, freq, my_delays)
            timeseries = fft.freq2time(my_out, 200 * units.MHz)

            return -100 * np.max(timeseries ** 2)

        start_direction = np.array([station.get_parameter(stationParameters.zenith) / units.rad,
                                    station.get_parameter(stationParameters.azimuth) / units.rad])
        self.logger.debug(f"Initial guess for fit routine is {start_direction}")

        # TODO: adding Bounds changes result of fitting routine!
        # fit_result = minimize(negative_beamed_signal, start_direction,
        #                       method='Powell', bounds=Bounds((0, -np.pi), (np.pi / 2, np.pi), (True, True)),
        #                       options={'maxiter': self.__max_iter, 'xtol': 1.0, 'disp': False,
        #                                'direc': np.array([[0.1, 0], [0, 0.1]])})
        #
        # self.logger.debug(f"Fit returned the following message: {fit_result.message}")
        # if not fit_result.success:
        #     self.logger.warning(f"The fit failed with the following message: {fit_result.message}")
        #     raise RuntimeError
        #
        # fit_direction = fit_result.x

        all = fmin_powell(negative_beamed_signal, start_direction,
                          maxiter=self.__max_iter, xtol=1.0 * units.deg,
                          direc=np.array([[2.0 * units.deg, 0], [0, 2.0 * units.deg]]),
                          disp=False, full_output=True)
        fit_direction = all[0]

        self.logger.debug(f"Fit finished with following direc vector: {all[2]}")

        direction_cartesian = hp.spherical_to_cartesian(*fit_direction)

        delays = geometric_delays(ant_positions, direction_cartesian)
        out = beamformer(fft_traces, freq, delays)

        return fit_direction, out

    @register_run()
    def run(self, event, detector):
        """
        reconstruct signal arrival direction for all events through beam forming.
        https://arxiv.org/pdf/1009.0345.pdf

        Parameters
        ----------
        event: Event
            The event to run the module on
        detector: Detector
            The detector object

        """
        converter = voltageToEfieldConverter()
        converter.begin()

        for station in event.get_stations():
            if not station.get_parameter(stationParameters.triggered):
                # Not triggered means no reliable pulse found or not enough antennas to do the fit
                continue

            zenith = event.get_hybrid_information().get_hybrid_shower("LORA").get_parameter(showerParameters.zenith)
            azimuth = event.get_hybrid_information().get_hybrid_shower("LORA").get_parameter(showerParameters.azimuth)

            # Get all group IDs which are still present in the Station
            station_channel_group_ids = set([channel.get_group_id() for channel in station.iter_channels()])

            position_array = []
            good_antennas = []
            for group_id in station_channel_group_ids:
                channels = [channel for channel in station.iter_channel_group(group_id)]

                # Only use the channels with an acceptable SNR
                good_snr = False
                for channel in channels:
                    if channel.get_parameter(channelParameters.SNR) > self.__cr_snr:
                        good_snr = True

                if good_snr:
                    position_array.append(
                        detector.get_absolute_position(station.get_id()) +
                        detector.get_relative_position(station.get_id(), channels[0].get_id())
                    )  # the position are the same for every polarisation

                    good_antennas.append((channels[0].get_id(), channels[1].get_id()))

            station.set_parameter(stationParameters.zenith, zenith)
            station.set_parameter(stationParameters.azimuth, azimuth)

            direction_difference = np.asarray([90, 90]) * units.deg
            while direction_difference[0] > 0.5 * units.deg or direction_difference[1] > 0.5 * units.deg:
                # Make sure all the previously calculated Efields are removed
                station.set_electric_fields([])

                # Unfold antenna response for good antennas
                for ant in good_antennas:
                    converter.run(event, station, detector, use_channels=ant)

                # Perform the direction fit on the station
                try:
                    direction_fit, freq_spectrum = self._direction_fit(
                        station, position_array
                    )
                except RuntimeError:
                    self.logger.error(f"Direction fit could not be completed for station  CS{station.get_id():03d}")
                    break

                # Check if fit produced unphysical results
                if direction_fit[0] > 90 * units.deg:
                    self.logger.warning(f"Zenith angle {direction_fit[0] / units.deg} is larger than 90 degrees!")
                    self.logger.error(f"Direction fit could not be completed for station  CS{station.get_id():03d}")
                    break

                # See if the fit converged
                zenith_diff = np.abs(station.get_parameter(stationParameters.zenith) / units.rad - direction_fit[0])
                azimuth_diff = np.abs(station.get_parameter(stationParameters.azimuth) / units.rad - direction_fit[1])

                direction_difference = [zenith_diff, azimuth_diff]

                # Bookkeeping
                station.set_parameter(stationParameters.zenith, direction_fit[0])
                station.set_parameter(stationParameters.azimuth, direction_fit[1])

                self.logger.debug('Difference after another fit iteration is %s;' % direction_difference)
                self.logger.debug('Direction after this fit iteration is %s;' % direction_fit)

            self.logger.info(f"Azimuth (wrt to North) and elevation for station CS{station.get_id():03d}:")
            self.logger.info((90 - station.get_parameter(stationParameters.azimuth) / units.deg,
                              90 - station.get_parameter(stationParameters.zenith) / units.deg))

            station.set_parameter(stationParameters.cr_zenith, station.get_parameter(stationParameters.zenith))
            station.set_parameter(stationParameters.cr_azimuth, station.get_parameter(stationParameters.azimuth))

    def end(self):
        pass
