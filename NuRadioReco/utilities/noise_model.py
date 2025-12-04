import numpy as np
import scipy as scp
import matplotlib.pyplot as plt

from NuRadioReco.utilities import fft
import NuRadioReco.framework.channel
import NuRadioReco.framework.sim_station

import logging
logger = logging.getLogger('NuRadioReco.utilities.noise_model')


class NoiseModel:
    """
    Probabilistic description of band-limited noise in radio detectors. The noise is assumed to be a multivariate gaussian
    of dimension n_samples. The covariance matrix of the distribution can be calculated from the spectrum of the noise or
    directly from datasets containing only noise.

    The main purpose of this class is to calculate the likelihood of a signal  given a measured trace, which can be used
    for likelihood reconstruction as described in https://arxiv.org/abs/2510.21925. Additionally, the class can be used
    to estimate the Fisher information matrix for a parameterized signal model.

    Parameters
    ----------
        n_antennas : int
            Number of antennas
        n_samples : int
            Number of samples in each trace
        sampling_rate : float
            The sampling rate of the antennas
        matrix_inversion_method : str, optional
            If the covariance matrix is not full rank, the inverse, which is used to calculate the multivariate normal,
            does not exist. This is the case if any frequency amplitudes in the spectra are zero, which corresponds to
            a distribution in a sub-space which has lower number of dimensions than n_samples, i.e. some dimensions are
            degenerate.  The correct mathematical way to describe the degenerate case is by using the pseudo-inverse
            and pseudo-determinant to calculate the multivariate normal PDF. This method is used if this parameters is
            set to "pseudo_inv", and is numerically stable but slow. We thus also provide the option to use the regular
            inverse by setting this parameter to "regular_inv", which works for full rank matrices and sometimes for
            lower rank matrices (see increase_cov_diagonal).
        threshold_amplitude : float, optional
            Fraction of the maximum amplitude in the spectra below which the frequency amplitudes are considered zero.
            Also used as the threshold for the eigenvalues when calculating the pseudo-inverse and pseudo-determinant.
        increase_cov_diagonal : float, optional
            Only used if matrix_inversion_method is "regular_inv". Calculating the inverse of a (covariance) matrix is
            numerically unstable, and it does not exist if the matrix is not full rank. By adding a small component to the
            diagonal when the inverse increases the rank and can improve stability for lower rank matrices. The parameter is
            what fraction of the variance should be added to the diagonal of the covariance matrix only when calculating
            the inverse.
        ignore_llh_normalization : bool, optional
            We are generally only interrested in likelihood ratios or delta log likelihood. In this case the normalization of
            the distribution cancels out and can just be set to 1. This speeds up initializing the class/covariance matrices.
    """

    def __init__(self, n_antennas, n_samples, sampling_rate, matrix_inversion_method="pseudo_inv", threshold_amplitude=1e-2, increase_cov_diagonal=0, ignore_llh_normalization=True):
        self.n_antennas = n_antennas
        self.n_samples = n_samples
        self.sampling_rate = sampling_rate
        self.matrix_inversion_method = matrix_inversion_method
        self.threshold_amplitude = threshold_amplitude
        self.increase_cov_diagonal = increase_cov_diagonal
        self.ignore_llh_normalization = ignore_llh_normalization
        self.frequencies = np.fft.rfftfreq(n_samples, 1.0/sampling_rate)
        self.n_frequencies = len(self.frequencies)
        self.spectra = None
        self.Vrms = None
        self.cov = None
        self.cov_inv = None
        self.cov_log_det = None
        self.t_array = np.arange(n_samples) * 1.0 / sampling_rate
        self.data_saved = None

        logger.info("NoiseModel initialized with {} antennas, {} samples, and {} Hz sampling rate".format(n_antennas, n_samples, sampling_rate))
        logger.info("To set the covariance matrices/spectra for the likelihood calculation run either initialize_with_spectra() or initialize_with_data()")

    def initialize_with_spectra(self, spectra, Vrms=None):
        """
        Initialize the noise model using spectra

        Parameters
        ----------
            spectra : numpy.ndarray
                Array containing spectra with dimensions [n_antennas,n_frequencies] or [n_frequencies]. For the latter case, all antennas
                are assumed to have the same spectrum. The spectra are defined as the mean of the Fourier transforms of the noise traces.
            Vrms : numpy.ndarray, optional
                List of Vrms values for each antenna. If provided, the spectra will be normalized to these values. Otherwise
                the normalization of the spectra is used. Default value is None. If only one value is given, all antennas
                are assumed to have the same Vrms.
        """
        assert spectra.dtype != complex, "Provided spectra are complex. Please provide the magnitude of the spectra/filter(s) instead."
        assert np.atleast_2d(spectra).shape[1] == self.n_frequencies, "The shape of the provided spectra does not match the number of samples per trace"
        if len(spectra.shape) == 1:
            spectra = np.tile(spectra,[self.n_antennas, 1])
        if Vrms is not None and len(np.atleast_1d(Vrms)) == 1:
            Vrms = np.tile(Vrms, self.n_antennas)
        self.Vrms = Vrms

        # Scale spectra to Vrms if provided:
        if Vrms is not None:
            for i_ant in range(self.n_antennas):
                spectrum_power = np.sum(spectra[i_ant]**2) * (self.frequencies[1] - self.frequencies[0])
                time_domain_power = Vrms[i_ant]**2 * self.n_samples * 1/self.sampling_rate
                spectra[i_ant, :] = spectra[i_ant, :] / np.sqrt(spectrum_power) * np.sqrt(time_domain_power)

        covariance_matrices, covariance_matrices_inverse = self._calculate_covariance_matrices_from_spectra(abs(spectra))
        self._set_covariance_matrices(covariance_matrices, spectra, cov_inv=covariance_matrices_inverse)

    def initialize_with_data(self, data, method="using_spectra"):
        """
        Initialize the noise model using traces containing noise

        Parameters
        ----------
            data : numpy.ndarray
                Array containing traces with noise with dimensions [n_datasets,n_antennas,n_samples] or [n_datasets,n_samples] for one antenna
            method : str, optional
                Method to calculate the covariance matrices. If set to "using_spectra", the spectra are calculated
                from the data and the covariance matrices are calculated from the spectra. If set to "autocorrelation",
                the covariance matrices are calculated directly from the data using numpy.cov()
        """
        if self.n_antennas == 1 and len(data.shape) == 2:
            data = data[:,np.newaxis,:]

        if method == "using_spectra":
            spectra = self._calculate_spectra_from_data(data)
            covariance_matrices, covariance_matrices_inverse = self._calculate_covariance_matrices_from_spectra(spectra)
            self._set_covariance_matrices(covariance_matrices, spectra, cov_inv=covariance_matrices_inverse)
        elif method == "autocorrelation":
            spectra = self._calculate_spectra_from_data(data)
            covariance_matrices = self._calculate_covariance_matrices_from_data(data)
            self._set_covariance_matrices(covariance_matrices, spectra)
        self.Vrms = np.std(data, axis=(0,2))

    def _set_covariance_matrices(self, cov, spectra, cov_inv=None):
        """
        Sets the covariance matrices, their (pseudo-)inverses, and log-determinants. Additionally, the
        noise power spectral density is saved, for calculations in the frequency domain.

        Parameters
        ----------
            cov : numpy.ndarray
                Covariance matrices for all antennas. Has dimensions [n_antennas,n_samples,n_samples]
            spectra : numpy.ndarray
                Array containing spectra with dimensions [n_antennas,n_frequencies] or [n_frequencies]. For the latter case, all antennas
                are assumed to have the same spectrum.
        """

        self.cov = cov

        # Set inverse or calculate it if not provided:
        if cov_inv is not None:
            self.cov_inv = cov_inv
        elif cov_inv is None:
            self.cov_inv = np.zeros([self.n_antennas, self.n_samples, self.n_samples])
            for i_ant in range(self.n_antennas):
                if self.matrix_inversion_method == "pseudo_inv":
                    estimated_eigenvalues = spectra[i_ant, :]**2 / (2 * self.n_samples * (1/self.sampling_rate)**2)
                    self.cov_inv[i_ant, :, :] = scp.linalg.pinvh(cov[i_ant, :, :], atol = np.max(estimated_eigenvalues) * self.threshold_amplitude**2)
                elif self.matrix_inversion_method == "regular_inv":
                    self.cov_inv[i_ant, :, :] = np.linalg.inv(cov[i_ant, :, :] + np.diag(np.ones(self.n_samples) * cov[i_ant, 0, 0] * self.increase_cov_diagonal))
                else:
                    raise Exception("""matrix_inversion_method not recognized. Choose "pseudo_inv" or "regular_inv" """)

        # Calculate log-determinant of covariance matrix:
        if not self.ignore_llh_normalization:
            self.cov_log_det = np.zeros(self.n_antennas)
            for i_ant in range(self.n_antennas):
                if self.matrix_inversion_method == "pseudo_inv":
                    eigen_values = np.linalg.eigvalsh(cov[i_ant, :, :])
                    self.cov_log_det[i_ant] = np.sum(np.log(eigen_values[eigen_values > np.max(eigen_values) * self.threshold_amplitude**2]))
                elif self.matrix_inversion_method == "regular_inv":
                    self.cov_log_det[i_ant] = np.linalg.det(cov[i_ant, :, :])
        else:
            self.cov_log_det = np.ones(self.n_antennas)

        # Calculate noise power spectral density from the spectra:
        self.noise_psd = np.zeros([self.n_antennas, self.n_frequencies])
        for i in range(self.n_antennas):
            self.noise_psd[i,:] = 2 * spectra[i,:]**2 * (self.frequencies[1] - self.frequencies[0])

        # Save spectra:
        self.spectra = spectra


    def _calculate_spectra_from_data(self, data):
        """
        Calculates the spectra for all antennas

        Parameters
        ----------
            data : numpy.ndarray
                Array containing data with dimensions [n_datasets,n_antennas,n_samples]

        Returns
        -------
            numpy.ndarray
                spectra (average of Fourier transforms of events) for each antenna with dimensions [n_antennas,n_frequencies]
        """
        spectra = np.zeros([self.n_antennas, self.n_frequencies])
        for i_ant in range(self.n_antennas):
            fourier_transforms = fft.time2freq(data[:, i_ant, :], self.sampling_rate)
            fourier_transforms_mean = np.sqrt(np.mean(fourier_transforms.real**2 + fourier_transforms.imag**2, axis=0))
            spectra[i_ant, :] = fourier_transforms_mean

        return spectra

    def _calculate_covariance_matrices_from_spectra(self, spectra):
        """
        Calculates covariance matrices from spectra

        Parameters
        ----------
            spectra : numpy.ndarray
                Array containing spectra with dimensions [n_antennas,n_frequencies] or [n_frequencies]. For the latter case, all antennas
                are assumed to have the same spectrum.

        Returns
        -------
            covariance_matrices : numpy.ndarray
                Covariance matrices for each antenna with dimensions [n_frequenciess,n_frequenciess]
            covariance_matrices_inverse : numpy.ndarray
                Inverse of covariance matrices for each antenna with dimensions [n_frequenciess,n_frequenciess]
        """
        # The normalization convention of the Fourier transform in NuRadioReco has to be taken into account:
        amplitudes = spectra * self.sampling_rate / np.sqrt(2) / np.sqrt(self.n_samples)

        covariance_matrices = np.zeros([self.n_antennas, self.n_samples, self.n_samples])
        covariance_matrices_inverse = np.zeros([self.n_antennas, self.n_samples, self.n_samples])

        for i_ant in range(self.n_antennas):
            active_amplitudes = amplitudes[i_ant, amplitudes[i_ant, :] > np.max(amplitudes[i_ant, :]) * self.threshold_amplitude]
            active_frequencies = self.frequencies[amplitudes[i_ant, :]  > np.max(amplitudes[i_ant, :] * self.threshold_amplitude)]

            # Calculate first row of covariance matrix:
            covariance_one_row = np.zeros(self.n_samples)
            covariance_inverse_one_row = np.zeros(self.n_samples)
            for i_freq in np.arange(0, len(active_frequencies)):
                covariance_one_row +=  2 * active_amplitudes[i_freq]**2 * np.cos(2*np.pi * active_frequencies[i_freq] * self.t_array ) / self.n_samples
                covariance_inverse_one_row += 2 * (1/active_amplitudes[i_freq]**2) * np.cos(2*np.pi * active_frequencies[i_freq] * self.t_array) / self.n_samples

            # Construct covariances matrix and its inverse assuming they are circulant:
            for i_bin in range(self.n_samples):
                covariance_matrices[i_ant, :, i_bin] = np.roll(covariance_one_row, i_bin)
                covariance_matrices_inverse[i_ant, :, i_bin] = np.roll(covariance_inverse_one_row, i_bin)

        return covariance_matrices, covariance_matrices_inverse

    def _calculate_covariance_matrices_from_data(self, data):
        """
        Calculates the covariance matrix for each antenna using numpy.cov and averages along the diagonals
        assuming the covariance matrix is circulant

        Parameters
        ----------
            data : numpy.ndarray
                Array containing data with dimensions [n_datasets,n_antennas,n_samples]

        Returns
        -------
            numpy.ndarray
                Covariance matrices for each antenna with dimensions [n_frequenciess,n_frequenciess]
        """

        covariance_matrices = np.zeros([self.n_antennas, self.n_samples, self.n_samples])

        for i_ant in range(self.n_antennas):
            covariance_matrix = np.cov(data[:, i_ant, :].T)

            # Calculate averages along diagonals:
            shifted_cov_matrix = np.zeros([self.n_samples, self.n_samples])
            for i_bin in range(self.n_samples):
                shifted_cov_matrix[:, i_bin] = np.roll(covariance_matrix[:, i_bin], -i_bin)
            covariances_avg = np.mean(shifted_cov_matrix, axis=1)

            # Make sure covariance matrix is symmetric (no numerical errors):
            for i_bin in range(int(self.n_samples / 2)):
                mean = 0.5 * (covariances_avg[i_bin] + covariances_avg[-i_bin])
                covariances_avg[i_bin] = mean
                covariances_avg[-i_bin] = mean

            # Construct covariances matrix:
            averaged_covariance_matrix = np.zeros([self.n_samples, self.n_samples])
            for i_bin in range(self.n_samples):
                averaged_covariance_matrix[:, i_bin] = np.roll(covariances_avg, i_bin)

            covariance_matrices[i_ant, :, :] = averaged_covariance_matrix

        return covariance_matrices

    def _log_multivariate_normal(self, x, mu, cov_inv, cov_log_det):
        """
        Calculates the multivariate normal probability (PDF) of vector x, given means mu and covariance matrix
        inverse and determinant.

        Parameters
        ----------
            x : numpy.ndarray
                Vector containing noise with dimensions [n_samples]
            mu : numpy.ndarray
                Means of the distribution with dimensions [n_samples]
            cov_inv : numpy.ndarray
                Inverse covariance matrix of the distribution with dimensions [n_samples,n_samples]
            cov_log_det : float
                Determinant of covariance matrix of the distribution

        Returns
        -------
            float
                Natural logarithm to the PDF value of the vector x given the multivariate normal distribution described by mu, cov_inv, and cov_log_det
        """
        n = len(x)
        term_1 = -0.5 * n * np.log(2*np.pi)
        term_2 = -0.5 * cov_log_det
        # np.matmul sometimes returns a np.matrix instead of np.array, which needs at least 2 dimensions. So we need to cast and flatten it:
        term_3_temp = np.array(np.matmul(cov_inv, x-mu)).flatten()
        term_3 = -0.5 * np.matmul(x-mu, term_3_temp)
        return term_1 + term_2 + term_3

    def _log_multivariate_normal_freq(self, x_minus_mu_fft, noise_psd):
        """
        Calculates the multivariate normal probability (PDF) of vector x, given the fourier transformed trace
        minus signal and the noise power spectral density. This calculation is more effecient than the time domain
        calculation.

        Parameters
        ----------
            x : numpy.ndarray
                Vector containing noise with dimensions [n_samples]
            mu : numpy.ndarray
                Means of the distribution with dimensions [n_samples]
            cov_inv : numpy.ndarray
                Inverse covariance matrix of the distribution with dimensions [n_samples,n_samples]
            cov_log_det : float
                Determinant of covariance matrix of the distribution

        Returns
        -------
            float
                Natural logarithm to the PDF value of the vector x given the multivariate normal distribution described by mu, cov_inv, and cov_log_det
        """
        n = self.n_samples
        term_1 = -0.5 * n * np.log(2*np.pi)
        term_2 = -0.5 * np.sum(np.log(noise_psd[noise_psd > np.max(noise_psd) * self.threshold_amplitude**2])) #-0.5 * self.cov_log_det[0]
        integrand = abs(x_minus_mu_fft*x_minus_mu_fft.conj())/noise_psd
        term_3 = -0.5 * 4*np.sum(integrand[noise_psd > np.max(noise_psd) * self.threshold_amplitude**2]) * (self.frequencies[1]-self.frequencies[0])
        return term_1 + term_2 + term_3

    def calculate_delta_llh(self, data, signal=None, frequency_domain=False):
        """
        Calculates delta log likelihood for the datasets relative to the most probable trace

        Parameters
        ----------
            data : numpy.ndarray
                Array containing data with dimensions [n_datasets,n_antennas,n_samples] or [n_antennas,n_samples]. For one antenna,
                the shapes [n_datasets,n_samples] or [n_samples] are also allowed.
            signal : numpy.ndarray, optional
                Array containing neutrino signal signal of dimensions [n_antennas,n_samples].
                If no signal is provided, it will be set to zeros.
            frequency_domain : bool, optional
                If True, calculate the delta log likelihood in the frequency domain, which is faster.

        Returns
        -------
            numpy.ndarray
                The delta log likelihood for the data relative to the most probable noise
        """
        # Handle different shapes of data and signal:
        if self.n_antennas == 1 and len(data.shape) == 2:
            data = data[:, np.newaxis, :]
        if self.n_antennas > 1 and len(data.shape) == 2:
            data = data[np.newaxis, :, :]
        if self.n_antennas == 1 and len(data.shape) == 1:
            data = data[np.newaxis, np.newaxis, :]

        if signal is None:
            means = np.zeros([self.n_antennas, self.n_samples])
        elif len(signal.shape) == 2:
            means = signal
        elif self.n_antennas == 1 and len(signal.shape) == 1:
            means = signal[np.newaxis,:]

        n_datasets = len(data)

        LLH_best = 0
        for i_ant in range(self.n_antennas):
            if not frequency_domain:
               LLH_best += self._log_multivariate_normal(x=means[i_ant, :], mu=means[i_ant, :], cov_inv=self.cov_inv[i_ant, :, :], cov_log_det=self.cov_log_det[i_ant])
            elif frequency_domain:
                LLH_best += self._log_multivariate_normal_freq(x_minus_mu_fft=np.zeros(self.n_frequencies), noise_psd=self.noise_psd[i_ant])

        LLH_array = np.zeros(n_datasets)

        for i_data in range(n_datasets):
            # Sum over likelihood for all antennas:
            LLH = 0
            for i_antenna in range(self.n_antennas):
                if not frequency_domain:
                    LLH += self._log_multivariate_normal(x=data[i_data, i_antenna, :], mu=means[i_antenna, :], cov_inv=self.cov_inv[i_antenna, :, :], cov_log_det=self.cov_log_det[i_antenna])
                if frequency_domain:
                    x_minus_mu_fft = fft.time2freq(data[i_data, i_antenna, :] - means[i_antenna, :], self.sampling_rate)
                    LLH += self._log_multivariate_normal_freq(x_minus_mu_fft=x_minus_mu_fft, noise_psd=self.noise_psd[i_antenna])
            LLH_array[i_data] = LLH

        # Print warning if the data contains frequencies not present in the assumed noise spectrum:
        if not (data == self.data_saved).all():
            self.data_saved = np.copy(data)
            for i_data in range(n_datasets):
                for i_ant in range(self.n_antennas):
                    if any(np.abs(fft.time2freq(data[i_data, i_ant, :], self.sampling_rate))[self.spectra[i_ant, :] > np.max(self.spectra[i_ant, :]) * self.threshold_amplitude] / self.spectra[i_ant, :][self.spectra[i_ant, :] > np.max(self.spectra[i_ant, :]) * self.threshold_amplitude] > 100):
                        msg = (f"Warning: The ratio of the Fourier transform of the data and the"
                               f"spectra of the noise model is larger than 100. This indicates that"
                               f"the noise model is initialized with a wrong spectrum. "
                               f"Max ratio: {np.max(np.abs(fft.time2freq(data[i_data, i_ant,:], self.sampling_rate))[self.spectra[i_ant, :] > np.max(self.spectra[i_ant,:]) * self.threshold_amplitude] / self.spectra[i_ant, :][self.spectra[i_ant, :] > np.max(self.spectra[i_ant, :]) * self.threshold_amplitude])}")
                        logger.warning(msg)

        return np.squeeze(LLH_array - LLH_best)

    def calculate_minus_two_delta_llh(self, data, signal=None, frequency_domain=False):
        """
        Calculates the minus two delta log likelihood for the datasets relative to the most probable noise

        Parameters
        ----------
            data : numpy.ndarray
                Array containing data with dimensions [n_datasets,n_antennas,n_samples] or [n_antennas,n_samples]
            signal : numpy.ndarray, optional
                Array containing neutrino signal signal of dimensions [n_antennas,n_samples].
                If no signal is provided, it will be set to zeros.
            frequency_domain : bool, optional
                If True, calculate the delta log likelihood in the frequency domain, which is faster.

        Returns
        -------
            np.array
                Minus two delta log likelihood for the data given the noise model
        """
        return -2*self.calculate_delta_llh(data, signal=signal, frequency_domain=frequency_domain)

    def calculate_minus_two_delta_llh_station(self, station, sim_station, time_grid=None, use_channels=None, frequency_domain=False, plot=True, return_traces=False):
        """
        Calculates the minus two delta log likelihood with a NuRadioReco station containing the data channels, i.e., with noise, and a sim station that contains the noiseless traces.
        This function should correctly loop over the antennas and add together different ray-tracing solutions in the readout window of the data with the desired time offset between the
        sim noiseless traces and data. By providing a time_grid, the likelihood can be calculated for different time offsets between the data and the signal.

        Parameters
        ----------
            station : NuRadioReco.framework.base_station.Station
                Station containing the data channels
            sim_station : NuRadioReco.framework.base_station.Station
                sim station containing channels for the signal for all antennas and all ray-tracing solutions.
            use_channels : list
                List of channel ids of station to use in the calculation
            time_grid : numpy.ndarray
                Array (or single number) containing time offsets between the data and the signal to calculate the likelihood for. The time offset is
                the time between the start of the data channel of the first antenna and the start of the signal/sim channel (first solution) of the
                first antenna. If the time_grid is not provided, the function will calculate the likelihood for a coarse time grid and then refine the
                time grid around the best time offset.
            frequency_domain : bool, optional
                If True, calculate the delta log likelihood in the frequency domain, which is faster.
            plot : bool, optional
                If True, plot the data and signal for each time offset in the time_grid.
            return_traces : bool, optional
                If True, return the data traces and the signal traces for the best time offset.

        Returns
        -------
            float
                Best minus two delta log likelihood
            float
                Best time offset
            numpy.ndarray
                Array containing the minus two delta log likelihood for each time offset in the time_grid
            numpy.ndarray, optional
                Array containing the data traces. Only returned if return_traces is True.
            numpy.ndarray, optional
                Array containing the signal traces for the best time offset. Only returned if return_traces is True.
        """

        if use_channels is None:
            use_channels = station.get_channel_ids()
        assert len(use_channels) == self.n_antennas, f"Number of channels to use ({len(use_channels)}) does not match the number of antennas ({self.n_antennas}) in the noise model"

        if time_grid is None:
            data_times = list(station.iter_channels())[0].get_times()
            data_duration = data_times[-1] - data_times[0]
            simulation_times = list(sim_station.iter_channels())[0].get_times()
            duration_simulation = simulation_times[-1] - simulation_times[0]
            time_grid_coarse = np.arange(-duration_simulation, data_duration, 0.5 * 1/self.sampling_rate)
            llh_best, t_best, LLH_array = self.calculate_minus_two_delta_llh_station(station, sim_station, time_grid=time_grid_coarse, use_channels = use_channels, frequency_domain=frequency_domain, plot=plot, return_traces=False)
            time_grid = np.linspace(t_best - 2 * 1/self.sampling_rate, t_best + 2 * 1/self.sampling_rate, 100)

        data_array = np.zeros([self.n_antennas, self.n_samples])

        # We use the time difference between the start of the data and first solution in the first antenna as a reference
        t_0_data = station.get_channel(use_channels[0]).get_trace_start_time()
        t_0_sim = list(sim_station.iter_channels())[0].get_trace_start_time() # There is probably a better way to do this

        referece_time_offset = t_0_sim - t_0_data

        trace_start_times = np.zeros(self.n_antennas)
        n_skipped = 0
        for i_ant, channel in enumerate(station.iter_channels()):
            if not channel.get_id() in use_channels:
                n_skipped += 1
                continue
            assert channel.get_number_of_samples() == self.n_samples, f"Number of samples in data channel {i_ant} ({channel.get_number_of_samples()}) does not match the number of samples in the noise model ({self.n_samples})"
            data_array[i_ant-n_skipped, :] = channel.get_trace()
            trace_start_times[i_ant-n_skipped]  = channel.get_trace_start_time()

        # Loop over the times in the time_grid:
        LLH_array = np.zeros(len(np.atleast_1d(time_grid)))
        signal_arrays = np.zeros([len(time_grid), self.n_antennas, self.n_samples])
        for i_time, time_offset in enumerate(np.atleast_1d(time_grid)):

            for i_ant, channel_id in enumerate(use_channels):

                # Make empty channel:
                signal_readout_channel = NuRadioReco.framework.channel.Channel(1)
                signal_readout_channel.set_trace(np.zeros(self.n_samples), self.sampling_rate)

                # Set trace start time of the readout window so it keeps track of the relative readout times
                # of the data windows, but moved to where the signal is located:
                signal_readout_channel.set_trace_start_time(trace_start_times[i_ant] + referece_time_offset - time_offset) # a positive time_offset moves the signal to the right relative to the data

                # Now add the simulation to the readout window:
                for i_channel, sim_channel in enumerate(sim_station.get_channels_by_channel_id(channel_id)):
                    signal_readout_channel.add_to_trace(sim_channel)

                signal_arrays[i_time, i_ant, :] = signal_readout_channel.get_trace()

            LLH_array[i_time] = self.calculate_minus_two_delta_llh(data_array, signal_arrays[i_time, :, :], frequency_domain=frequency_domain)

        if plot:

            plt.figure(figsize=(20,5))
            plt.plot(time_grid, LLH_array, ".",ls="-")
            plt.xlabel("Time [ns]")
            plt.ylabel(r"$-2\Delta \ln\mathcal{L}$")
            plt.title(f"Likelihood for different time offsets")
            plt.yscale("log")
            plt.tight_layout()
            plt.show()
            plt.close()

            fig, ax = plt.subplots(self.n_antennas, figsize=[10, 2*self.n_antennas])
            for i_ant in range(self.n_antennas):
                ax[i_ant].plot(list(station.iter_channels())[0].get_times(), data_array[i_ant,:], "k-", label="Data")
                ax[i_ant].plot(list(station.iter_channels())[0].get_times(), signal_arrays[np.argmin(LLH_array),i_ant,:], "b--", label="Signal (best time offset)")
                ax[i_ant].set_xlabel("Time [ns]")
                ax[i_ant].set_ylabel("Voltage [V]")
                ax[i_ant].legend()
                if i_ant == 0:
                    ax[i_ant].set_title(f"Best time offset: {time_grid[np.argmin(LLH_array)]} ns")
            fig.tight_layout()
            plt.show()
            plt.close()

            fig, ax = plt.subplots(self.n_antennas, figsize=[10, 2*self.n_antennas])
            for i_ant in range(self.n_antennas):
                data_fft = fft.time2freq(data_array[i_ant,:], self.sampling_rate)
                signal_fft = fft.time2freq(signal_arrays[np.argmin(LLH_array),i_ant,:], self.sampling_rate)
                ax[i_ant].plot(self.frequencies, abs(data_fft[:]), "k-", label="Data FFT")
                ax[i_ant].plot(self.frequencies, abs(signal_fft[:]), "b-", label="Signal FFT")
                ax[i_ant].plot(self.frequencies, abs(self.spectra[i_ant,:]), "r-", label="Assumed noise spectrum")
                if i_ant == self.n_antennas - 1: ax[i_ant].set_xlabel("Frequency [Hz]")
                ax[i_ant].set_ylabel("Amplitude [V/GHz]")
                if i_ant == 0:
                    ax[i_ant].legend()
                ax[i_ant].set_title(f"Antenna {i_ant}")
            fig.tight_layout()
            plt.savefig("debug_llh_spectra.png")
            plt.show()
            plt.close()

        llh_best = np.min(LLH_array)
        t_best = time_grid[np.argmin(LLH_array)]

        if return_traces:
            return llh_best, t_best, LLH_array, data_array, signal_arrays[np.argmin(LLH_array),:,:]
        else:
            return llh_best, t_best, LLH_array


    def get_minus_two_delta_llh_function(self, data_to_fit, signal_function, frequency_domain=False):
        """
        Convenience function. Returns a function that calculates the minus two delta log likelihood for signal given a dataset with noise.
        The returned function can be used directly in optimizers like scipy.optimize.minimize.

        Parameters
        ----------
            data_to_fit : numpy.ndarray
                Array containing one dataset with dimensions [n_antennas,n_samples]
            signal_function : function
                Function which takes a vector containing parameters as input and returns a signal array with dimensions [n_antennas,n_samples].
            frequency_domain : bool, optional
                If True, calculate the delta log likelihood in the frequency domain, which is faster.

        Returns
        -------
            minus_two_delta_llh_func : function
                Function that returns a minus two delta log likelihood for a set of parameters given a data realization and the noise model
        """

        def minus_two_delta_llh_func(params):
            signal = signal_function(params)
            LLH_signal = self.calculate_delta_llh(data_to_fit, signal, frequency_domain=frequency_domain)
            minus_two_delta_llh_signal = -2 * LLH_signal
            return minus_two_delta_llh_signal

        return minus_two_delta_llh_func

    def save_covariance_matrix(self, antenna, filename):
        """
        Save compressed version (one row) of covariance matrix for one antenna along with the sample rate in GHz,
        number of samples, and the spectra.

        Parameters
        ----------
            antenna : int
                Which antenna to save the covariance matrix for
            filename : str
                Name of file. Should follow the naming scheme:
                noise_covariance_matrix_<experiment>_<station_id>_<antenna_id>_<date_from>_to_<date_to>
                e.g.: noise_covariance_matrix_RNOG_station21_antenna6_2023-03-01_to_2023-10-31
        """
        cov_one_row = self.cov[antenna,:,0]
        object_to_save = np.array([self.sampling_rate, self.n_samples, cov_one_row, self.spectra[antenna,:]],dtype=object)
        np.save(filename, object_to_save)

    def load_covariance_matrices(self, filenames):
        """
        Load (one row of) covariance matrices for a all antennas (self.n_antennas) along with the sample rate in GHz. The full covariance matrices are then constructed and assigned to self.cov[:,:,:].

        Parameters
        ----------
            filenames : list[str]
                List of filenames which contain the covariance matrices. Should be of length self.n_antennas.
        """
        assert len(filenames) == self.n_antennas, f"Number of filenames ({len(filenames)}) does not match the number of antennas ({self.n_antennas})"

        covariance_matrices = np.zeros([self.n_antennas, self.n_samples, self.n_samples])
        spectra = np.zeros([self.n_antennas, self.n_samples])

        for i_ant, file in enumerate(filenames):
            cov_sample_rate, cov_n_samples, cov_one_row, spectrum = np.load(file, allow_pickle=True)

            # Until resampling is implementet, only matching sampling rates and number of samples are allowed:
            assert cov_sample_rate == self.sampling_rate, f"Sampling rate ({self.sampling_rate}) does not match covariance matrix sampling rate ({cov_sample_rate})"
            assert cov_n_samples == self.n_samples, f"Number of samples ({self.n_samples}) does not match covariance matrix number of samples ({cov_n_samples})"

            # Construct covariances matrix assuming it is circulant:
            constructed_covariance_matrix = np.zeros([self.n_samples, self.n_samples])
            for i_bin in range(self.n_samples):
                constructed_covariance_matrix[:, i_bin] = np.roll(cov_one_row, i_bin)

            covariance_matrices[i_ant,:,:] = constructed_covariance_matrix
            spectra[i_ant] = spectrum

        self._set_covariance_matrices(covariance_matrices, spectra=spectra)

    def resample_covariance_matrices(self, new_sampling_rate):
        """
        Resample the covariance matrices to new sampling rate and the same trace duration.

        Parameters
        ----------
        new_sample_rate : float
            New sampling rate in GHz
        """

        # Get new number of samples:
        n_samples_new = int(self.n_samples * new_sampling_rate/self.sampling_rate)
        self.n_samples = n_samples_new
        self.sampling_rate = new_sampling_rate
        self.frequencies = np.fft.rfftfreq(n_samples_new, 1.0/new_sampling_rate)
        n_frequencies_new = len(self.frequencies)
        self.n_frequencies = n_frequencies_new

        covariance_matrices = np.zeros([self.n_antennas, n_samples_new, n_samples_new])
        spectra = np.zeros([self.n_antennas, n_samples_new])

        for i_ant in range(self.n_antennas):
            cov_one_row = self.cov[i_ant, 0, :]

            # Resample covariance matrix and spectra:
            cov_one_row, t_array = scp.signal.resample(cov_one_row, n_samples_new, t=self.t_array)
            spectra[i_ant] = scp.signal.resample(self.spectra[i_ant], n_frequencies_new)

            # Construct covariances matrix:
            covariance_matrix = np.zeros([n_samples_new, n_samples_new])
            for i_bin in range(n_samples_new):
                covariance_matrix[:, i_bin] = np.roll(cov_one_row, i_bin)

            covariance_matrices[i_ant, :, :] = covariance_matrix

        self._set_covariance_matrices(covariance_matrices, spectra)

    def calculate_fisher_information_matrix(self, signal_function, paramters_x0, dx, frequency_domain=False, ignore_parameters=[]):
        """
        Calculate Fisher information matrix for a set of parameter values (paramters_x0) which generates a signal using the covariance matrices of the noise.

        Parameters
        ----------
            signal_function : function
                Function that takes a list of parameter values as input and returns a neutrino signal with dimensions [n_antennas,n_samples]
            paramters_x0 : list
                Parameter values for which to calculate the Fisher information matrix
            dx : list
                Finite differences to use when calculating derivatives of signal_function with respect to the parameters
            frequency_domain : bool, optional
                If True, calculate the Fisher information matrix in the frequency domain.
            ignore_parameters : list, optional
                List of parameters indicies of signal_function to ignore when calculating the Fisher information matrix, e.g. [2, 5]. The method then
                returns a lower dimensional (len(paramters_x0) - len(ignore_parameters)) Fisher information matrix.

        Returns
        -------
            fisher_information_matrix : numpy.array
                Fisher information matrix for the parameters given the noise model

        """
        n_parameters = len(paramters_x0) - len(ignore_parameters)

        # Calculate derivatives:
        derivatives = np.zeros([n_parameters, self.n_antennas, self.n_samples])
        derivatives_fft = np.zeros([n_parameters, self.n_antennas, self.n_frequencies])
        signal_0 = signal_function(paramters_x0)
        i_skipped = 0
        for i_param in range(n_parameters):
            if i_param in ignore_parameters:
                i_skipped += 1
            paramters_x1 = np.copy(paramters_x0)
            paramters_x1[i_param + i_skipped] = paramters_x0[i_param + i_skipped] + dx[i_param + i_skipped]
            derivatives[i_param, :, :] = (signal_function(paramters_x1) - signal_0) / dx[i_param + i_skipped]
            if frequency_domain:
                derivatives_fft[i_param, :, :] = fft.time2freq(derivatives[i_param, :, :], self.sampling_rate)

        # Calculate Fisher information matrix
        fisher_information_matrix = np.zeros([n_parameters, n_parameters])
        for i_param in range(n_parameters):
            for j_param in range(n_parameters):
                for i_ant in range(self.n_antennas):
                    if not frequency_domain:
                        fisher_information_matrix[i_param, j_param] += np.matmul(derivatives[i_param, i_ant, :], np.matmul(self.cov_inv[i_ant, :, :], derivatives[j_param, i_ant, :]))
                    elif frequency_domain:
                        integrand = np.real(derivatives_fft[i_param, i_ant, :] * derivatives_fft[j_param, i_ant, :].conj()) / self.noise_psd[i_ant, :]
                        fisher_information_matrix[i_param, j_param] += 4 * np.sum(integrand[self.noise_psd[i_ant, :] > np.max(self.noise_psd[i_ant, :] * self.threshold_amplitude ** 2)]) * (self.frequencies[1] - self.frequencies[0])

        return fisher_information_matrix

    def get_dof(self):
        """
        Get number of degrees of freedom based on the noise spectra and threshold amplitude.
        """
        dof = 0
        for i_ant in range(self.n_antennas):
            dof += 2 * sum(self.spectra[i_ant] > np.max(self.spectra[i_ant, :]) * self.threshold_amplitude)
        return dof

    ### Plotting: ###

    def plot_data(self, data, plot_range=None, linestyle_and_color = "auto", make_new_figure=True):
        """
        Plots a dataset containing noise

        Parameters
        ----------
            data : numpy.ndarray
                Array containing data with dimensions [n_samples]
            plot_range : float
                Range along x-axis (self.t_array) to plot in units of nanoseconds
            linestyle_and_color : str, optional
                String specifying linestyle and color, e.f. "k-" for black solid line, or "b--" for blue
                dashed line. If set to "auto", matplotlib will set the color and style.
            make_new_figure : bool
                If True create a new figure
        """
        if make_new_figure:
            plt.figure(figsize=[4.2,3])

        if linestyle_and_color == "auto":
            plt.plot(self.t_array, data)
        else:
            plt.plot(self.t_array, data, linestyle_and_color)

        axis = plt.axis()
        if plot_range is None:
            plt.axis([0, max(self.t_array), axis[2], axis[3]])
        else:
            plt.axis([plot_range[0], plot_range[1], axis[2], axis[3]])
        plt.xlabel("Time [ns]")
        plt.ylabel("Voltage [V]")
        plt.tight_layout()

    def plot_covariance_matrix(self, cov, plot_range=None, make_new_figure=True):
        """
        Plots a covariance matrix

        Parameters
        ----------
            cov : numpy.ndarray
                Covariance matrix of dimensions [n_samples,n_samples]
            plot_range : float
                Range along x- and y-axes (self.t_array) to plot in units of nanoseconds
            make_new_figure : bool
                If True create a new figure
        """
        if make_new_figure:
            plt.figure(figsize=[4.2,3])

        # imshow looks better than pcolormesh:
        plt.imshow(cov, vmax=np.max(cov), vmin=-np.max(cov), cmap="seismic")
        plt.colorbar(label=f"Cov$(t_i,t_j)$")
        plt.xlabel(f"$t_i$ [ns]")
        plt.ylabel(f"$t_j$ [ns]")

        # Do some weird zoom in/out to get the ticks right:
        if plot_range is not None:
            plt.axis([plot_range[0], plot_range[1], plot_range[0], plot_range[1]])
            xticks = plt.xticks()
            yticks = plt.yticks()
            plt.xticks(xticks[0]*self.sampling_rate, xticks[1])
            plt.yticks(yticks[0]*self.sampling_rate, yticks[1])
            plt.axis([plot_range[0]*self.sampling_rate, plot_range[1]*self.sampling_rate, plot_range[0]*self.sampling_rate, plot_range[1]*self.sampling_rate])
            plt.gca().invert_yaxis()
        else:
            axis = plt.axis()
            plt.axis([0, axis[1]/self.sampling_rate, axis[2]/self.sampling_rate, 0])
            xticks = plt.xticks()
            yticks = plt.yticks()
            plt.xticks(xticks[0]*self.sampling_rate, xticks[1])
            plt.yticks(yticks[0]*self.sampling_rate, yticks[1])
            plt.axis(axis)
        plt.tight_layout()

    def plot_covariance_matrix_first_row(self, cov, plot_range=None, linestyle_and_color = "auto", make_new_figure=True):
        """
        Plots one row of a covariance matrix

        Parameters
        ----------
            cov : numpy.ndarray
                Covariance matrix of dimensions [n_samples,n_samples]
            plot_range : float
                Range along x-axis (self.t_array) to plot in units of nanoseconds
            linestyle_and_color : str, optional
                String specifying linestyle and color, e.f. "k-" for black solid line, or "b--" for blue
                dashed line. If set to "auto", matplotlib will set the color and style.
            make_new_figure : bool
                If True create a new figure
        """
        if make_new_figure:
            plt.figure(figsize=[4.2,3])


        if linestyle_and_color == "auto":
            plt.plot(self.t_array, cov[0,:])
        else:
            plt.plot(self.t_array, cov[0,:], linestyle_and_color)

        axis = plt.axis()
        if plot_range is None:
            plt.axis([0, max(self.t_array), axis[2], axis[3]])
        else:
            plt.axis([plot_range[0], plot_range[1], axis[2], axis[3]])
        plt.xlabel(f"$\Delta t$ [ns]")
        plt.ylabel(f"Cov$(\Delta t)$")
        plt.tight_layout()

    def plot_llh_distribution(self, data, n_dof=None, signal=None, frequency_domain=False, make_new_figure=True):
        """
        Calculate the llh values for many datasets and plot the distribution alongside a chi2
        distribution with dof equal to the number of samples

        Parameters
        ----------
            data : numpy.ndarray
                Array containing data with dimensions [n_samples]
            n_dof : int, optional
                Number of degrees of freedom for the chi2 distribution. If not provided, it is set equal to
                the number of samples
            frequency_domain : bool, optional
                If True, calculate the delta log likelihood in the frequency domain, which is faster.
            make_new_figure : bool
                If True create a new figure
        """
        LLH_array = self.calculate_delta_llh(data, signal=signal, frequency_domain=frequency_domain)

        n_datasets = len(data)
        if n_dof is None:
            n_dof = self.get_dof()

        if make_new_figure:
            plt.figure(figsize=[4.2,3])

        hist = plt.hist(-2*LLH_array, bins=np.arange(0, max(-2*LLH_array)+1, 15), histtype="step", color="b", label=str(n_datasets) + " datasets", density=False)

        axis = plt.axis()

        chi2 = scp.stats.chi2
        x = np.linspace(n_dof * 0.1, n_dof * 10, 10000)
        plt.plot(x, chi2.pdf(x, n_dof) * len(LLH_array) * (hist[1][1] - hist[1][0]), "r-", alpha=0.6, label="chi2, dof = "+str(n_dof))

        plt.axis([np.min([axis[0], n_dof * 0.9]), np.max([axis[1], n_dof * 1.2]), axis[2], axis[3]])

        plt.xlabel(f"$-2\Delta LLH$")
        plt.ylabel("Counts")
        plt.legend(loc=1)
        plt.tight_layout()
