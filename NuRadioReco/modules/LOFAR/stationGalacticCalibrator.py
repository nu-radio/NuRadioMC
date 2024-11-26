import os
import logging
import datetime

import numpy as np

from scipy.interpolate import interp1d
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy import units as u

from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units


def fourier_series(x, p):
    """
    Evaluates the partial Fourier series:

    .. math:: F(x) \\approx \\frac{a_{0}}{2} + \\sum_{n=1}^{\\mathrm{order}} a_{n} \\sin(nx) + b_{n} \\cos(nx)

    Here the coefficients :math:`a_{n}` are assumed to be the even elements of `p` and the :math:`b_{n}` coefficients
    the odd elements.
    """
    r = p[0] / 2
    order = int((len(p) - 1) / 2)
    for i in range(order):
        n = i + 1
        r += p[2 * i + 1] * np.sin(n * x) + p[2 * i + 2] * np.cos(n * x)
    return r


class stationGalacticCalibrator:
    """
    Apply the galactic calibration to all the channels, to each dipole polarization separately. This function
    assumes the traces have already been cleaned from any RFI. Both the absolute calibration using Galactic noise and
    the relative calibration between antenna's is applied.

    Parameters
    ----------
    experiment: str
        Reference to the antenna set parameters to use.

    Notes
    -----
    The absolute calibration makes use of a measured calibration curve, which encodes

    #. The conversion from ADC to Volts,
    #. As well as the gains and losses in the amplifiers and coax cables.

    The relative calibration makes sure all the antennas are calibrated to the same reference value. On the other
    hand, the calibration correlates this reference value to the Galactic noise in order to make the units
    physically meaningful.

    Further details are described in this `overview <https://arxiv.org/pdf/1311.1399.pdf>`_, and also this
    `paper <https://arxiv.org/pdf/1903.05988.pdf>`_ .
    """
    def __init__(self, experiment='LOFAR_LBA'):
        self.logger = logging.getLogger('NuRadioReco.LOFAR.stationGalacticCalibrator')

        self.__experiment = experiment

        self.__experiment_parameters = None
        self.__abs_calibration_curve = None
        self.__rel_calibration_coefficients = None

        self.begin()

    def begin(self, logger_level=logging.NOTSET):
        """
        Loads the experimental parameters (such as longitude and latitude) as well as the Galactic calibration
        curves and Fourier coefficients from the directory `NuRadioReco/utilities/data/`.

        Parameters
        ----------
        logger_level : int, default=logging.NOTSET
            Use this parameter to override the logging level for this module.
        """
        self.logger.setLevel(logger_level)

        # The files are stored in the data folder of the utilities module, which sits 3 folders up
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'utilities', 'data')

        # Get the experiment parameters such as latitude and longitude
        with open(os.path.join(data_dir, "experiment_parameters.txt"), "r") as f:
            all_experiment_parameters = f.readlines()

        for line in all_experiment_parameters:
            if line.startswith(self.__experiment):
                self.__experiment_parameters = line.split(", ")

        # Get absolute calibration curve
        self.__abs_calibration_curve = np.genfromtxt(
            os.path.join(
                data_dir, "galactic_calibration",
                f"{self.__experiment}_galactic_{self.__experiment_parameters[6]}_{self.__experiment_parameters[7]}.txt"
            ),
        )

        # Get fitted Fourier coefficients for relative calibration and store them based on polarisation group ID
        rel_calibration_file = np.genfromtxt(
            os.path.join(
                data_dir, "galactic_calibration",
                f"{self.__experiment}_Fourier_coefficients.txt",
            ),
            dtype=str,
            delimiter=', '
        )

        self.__rel_calibration_coefficients = {}
        for col in rel_calibration_file.T:
            group_id = int(col[0].split(" ")[1])
            coefficients = col[1:].astype('f8')

            self.__rel_calibration_coefficients[group_id] = coefficients

    def _get_absolute_calibration(self, frequencies):
        """
        Calculate the absolute calibration for a single trace, using the loaded calibration curve. The curve
        should be a 1-D array, containing the calibration values for frequencies from 0 MHz up to
        `len(calibration_curve)` MHz, in steps of 1 MHz.

        Parameters
        ----------
        frequencies: array-like
            The frequencies sampled in the trace.

        Returns
        -------
        ndarray
            The coefficient with which to multiply each frequency channel listed in `frequencies`.
        """
        # Set the calibration curve frequencies
        calibration_frequencies = np.arange(len(self.__abs_calibration_curve)) * units.MHz

        # Interpolate the curve between the frequency positions
        f = interp1d(calibration_frequencies, self.__abs_calibration_curve)

        # Apply the interpolation to the sampled frequencies and return
        return f(frequencies)

    def _get_relative_calibration(self, local_sidereal_time, channel):
        """
        Calculate the relative calibration correction factor for a channel, given the Fourier coefficients for the curve
        of the galactic noise power they observe as a function of the local sidereal time. This makes sure
        all channels are calibrated to the same reference power. It is not frequency dependent, unlike the absolute
        calibration.

        Parameters
        ----------
        local_sidereal_time : float
            The local sidereal time of the observation.
        channel : Channel object
            The channel for which to calculate the correction factor.

        Returns
        -------
        scale : float
            The correction factor to be multiplied with the trace.

        Notes
        -----
        The idea here is that most of the trace contains noise, therefore the power of the trace should be dominated
        by the Galactic noise. The Fourier coefficients are fitted to the variation of the observed sky noise as a
        function of sidereal time. Normalising the trace with respect to this curve ensures all the antennas are
        calibrated to observe the same value of the Galactic noise.
        """

        # Get channel parameters -> group_id = a(even_id), odd_id = even_id + 1
        channel_polarisation = channel.get_id() - int(channel.get_group_id()[1:])  # TODO: need clear function?
        channel_bandwidth = channel.get_sampling_rate() / channel.get_number_of_samples() / units.s  # 1 / period
        channel_power = np.sum(np.abs(channel.get_frequency_spectrum()) ** 2) * channel_bandwidth

        self.logger.debug(f"Channel power of channel {channel.get_id()} is {channel_power}")

        # Calculate Galactic power noise
        # the local sidereal time runs from 0 to 24 (it is calculated from the Earth angle), so normalise it to 2 * pi
        galactic_noise_power = fourier_series(local_sidereal_time / 24.0 * 2 * np.pi,
                                              self.__rel_calibration_coefficients[channel_polarisation])

        # Calculate the correction factor per antenna
        scale = galactic_noise_power / channel_power
        if scale == np.inf:
            scale = 0.0  # A channel without a signal will have 0 channel power, and result in np.inf
        return np.sqrt(scale)  # Correction is applied in time domain

    def _calibrate_channel(self, channel, timestamp):
        """
        Convenience function to apply the absolute and relative calibration to a single channel.

        Parameters
        ----------
        channel : Channel object
            The channel to be calibrated.
        timestamp : int
            The UNIX timestamp corresponding to the observation.
        """
        # Find the sidereal time for the experiment
        dt_object = datetime.datetime.utcfromtimestamp(timestamp)
        observing_location = EarthLocation(lat=float(self.__experiment_parameters[4]) * u.deg,
                                           lon=float(self.__experiment_parameters[5]) * u.deg)
        observing_time = Time(dt_object, scale="utc", location=observing_location)
        local_time = observing_time.sidereal_time("apparent").hour

        # Load the trace from the channel
        trace_fft = channel.get_frequency_spectrum()
        trace_frequencies = channel.get_frequencies()
        trace_sampling_rate = channel.get_sampling_rate()

        # Calibrate
        trace_fft *= self._get_absolute_calibration(trace_frequencies)
        trace_fft *= self._get_relative_calibration(local_time, channel)

        # Set the calibrated trace back in the channel
        channel.set_frequency_spectrum(trace_fft, trace_sampling_rate)

    @register_run()
    def run(self, event):
        """
        Run the calibration on all stations in `event`.

        Parameters
        ----------
        event : Event object
            The event on which to apply the Galactic calibration.
        """
        timestamp = event.get_id() + 1262304000
        for station in event.get_stations():
            for channel in station.iter_channels():
                self._calibrate_channel(channel, timestamp)

    def end(self):
        pass
