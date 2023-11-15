import logging
import numpy as np
import matplotlib.pyplot as plt
import radiotools.helper as hp

from scipy import constants
from scipy.optimize import fmin_powell

from NuRadioReco.utilities import fft
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import stationParameters
from NuRadioReco.modules.base import module
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.modules.voltageToEfieldConverter import voltageToEfieldConverter
from NuRadioReco.modules.LOFAR.beamforming_utilities import beamformer


logger = module.setup_logger(level=logging.DEBUG)


lightspeed = constants.c * units.m / units.s


def geometric_delays(ant_positions, sky):
    """Returns geometric delays in a matrix.

    :param ant_positions: antenna positions
    :type ant_positions: np.ndarray
    :param sky: unit vector pointing to the arrival direction in cartesian coordinates
    :type sky: np.ndarray
    :return: delays
    :type: np.ndarray
    """
    sky *= 1e6
    distance = np.sqrt(np.sum(sky ** 2))
    delays = np.sqrt(np.sum((sky - ant_positions) ** 2, axis=1)) - distance
    delays /= lightspeed
    return delays


class beamformingDirectionFitter:
    """
    Fits the direction using interferometry between desired channels.
    """

    def __init__(self):
        self.logger = logging.getLogger("NuRadioReco.beamFormingDirectionFitter")

        self.__zenith = []
        self.__azimuth = []
        self.__delta_zenith = []
        self.__delta_azimuth = []

        self.__max_iter = None

    def begin(self, max_iter):
        self.__max_iter = max_iter

    def _direction_fit(self, fft_traces, freq, ant_positions):
        def negative_beamed_signal(direction):
            theta = direction[0]
            phi = direction[1]
            direction_cartesian = hp.spherical_to_cartesian(theta, phi)

            delays = geometric_delays(ant_positions, direction_cartesian)

            out = beamformer(fft_traces, freq, delays)
            timeseries = fft.freq2time(out, 200 * units.MHz)  # TODO: is this really necessary?

            return -100 * np.max(timeseries ** 2)

        start_direction = np.array([self.__zenith[-1], self.__azimuth[-1]])
        fit_direction = fmin_powell(negative_beamed_signal,
                                    start_direction,
                                    maxiter=self.__max_iter, xtol=1.0)

        theta = fit_direction[0]
        phi = fit_direction[1]
        direction_cartesian = hp.spherical_to_cartesian(theta, phi)

        delays = geometric_delays(ant_positions, direction_cartesian)
        out = beamformer(fft_traces, freq, delays)

        return fit_direction, out

    @register_run()
    def run(self, evt, det):
        """
        reconstruct signal arrival direction for all events through beam forming.
        https://arxiv.org/pdf/1009.0345.pdf

        Parameters
        ----------
        evt: Event
            The event to run the module on
        station: Station
            The station to run the module on
        det: Detector
            The detector object

        """
        converter = voltageToEfieldConverter()
        converter.begin()

        for station in evt.get_stations():
            if not station.get_parameter(stationParameters.triggered):
                # Not triggered means to reliable pulse found or not enough antennas to do the fit
                continue
            # TODO: need to obtain good channels and only use those

            zenith = station.get_parameter(stationParameters.zenith)
            azimuth = station.get_parameter(stationParameters.azimuth)

            position_array = [
                det.get_absolute_position(station.get_id()) +
                det.get_relative_position(station.get_id(), channel.get_id())
                for channel in station.iter_channel_group(0)
            ]  # the position are the same for every polarisation

            # FIXME: hardcoded polarisation list
            for channel0, channel1 in zip(station.iter_channel_group(0), station.iter_channel_group(1)):
                converter.run(evt, station, det, use_channels=[channel0.get_id(), channel1.get_id()])

            # TODO: does the dominant polarisation needs to be updated during loop?
            dominant_pol = station.get_parameter(stationParameters.cr_dominant_polarisation)

            # The e-field from the converter has eR as [0] component -> do +1 in index
            e_field_traces_fft = np.array([trace.get_frequency_spectrum()[dominant_pol+1]
                                           for trace in station.get_electric_fields()])

            frequencies = station.get_electric_fields()[0].get_frequencies()

            self.__zenith.append(zenith)
            self.__azimuth.append(azimuth)

            direction_difference = np.asarray([100, 100])
            while direction_difference[0] > 0.1 * units.deg and direction_difference[1] > 0.1 * units.deg:
                direction_fit, freq_spectrum = self._direction_fit(
                    e_field_traces_fft, frequencies, position_array
                )

                zenith_diff = np.abs(self.__zenith[-1] - direction_fit[0])
                azimuth_diff = np.abs(self.__azimuth[-1] - direction_fit[1])

                direction_difference = np.asarray([zenith_diff, azimuth_diff])

                # Bookkeeping
                self.__zenith.append(direction_fit[0])
                self.__azimuth.append(direction_fit[1])

                self.__delta_zenith.append(zenith_diff)
                self.__delta_azimuth.append(azimuth_diff)

                self.logger.debug('Difference after another fit iteration is %s;' % direction_difference)
                self.logger.debug('Direction after this fit iteration is %s;' % direction_fit)

            print(station.get_id())
            print(self.__zenith)
            print(self.__azimuth)

            self.__zenith = []
            self.__azimuth = []
            self.__delta_zenith = []
            self.__delta_azimuth = []

    def end(self):
        pass
