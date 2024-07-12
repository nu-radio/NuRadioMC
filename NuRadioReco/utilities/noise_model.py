import numpy as np
from NuRadioReco.utilities import fft
import scipy as scp
import matplotlib.pyplot as plt


class NoiseModel:
    """
    Probabilistic description of thermal noise in radio detectors. The noise is assumed to be a multivariate gaussion of
    dimension n_samples. The covariance matrix of the distirbution can be calculated from the spectrum of the noise or 
    directly from datasets containing only noise.

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
            a distribution in a sub-space whih has lower number of dimensions than n_samples, i.e. some dimensions are
            degenerate.  The correct mathematical way to describe the degenerate case is by using the pseudo-inverse
            and pseudo-determinant to calculate the multivariate normal PDF. This method is used if this parameters is 
            set to "pseudo_inv", and is numerically stable but slow. We thus also provide the option to use the regular 
            inverse by setting this parameter to "regular_inv", which works for full rank matrices and sometimes for
            lower rank matrices (see increase_cov_diagonal).
        threshold_amplitude : float, optional
            Below this threshold the frequency amplitudes of the spectra is considered zero. Also used as the threshold 
            for the eigenvalues when calculating the pseudo-inverse and pseudo-determinant.
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

    def __init__(self, n_antennas, n_samples, sampling_rate, matrix_inversion_method="pseudo_inv", threshold_amplitude=1e-12, increase_cov_diagonal=0, ignore_llh_normalization=True):
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
        #assert spectra.shape[1] == self.n_frequencies, "The dimensionality of the provided spectra does not match the number of samples per trace"
        if len(spectra.shape) == 1:
            spectra = np.tile(spectra,[self.n_antennas,1])
        if Vrms is not None and len(np.atleast_2d(Vrms)) == 1:
            Vrms = np.tile(Vrms, self.n_antennas)
        self.Vrms = Vrms

        # Scale spectra to Vrms if provided:
        if Vrms is not None:
            for i in range(self.n_antennas):
                spectrum_power = np.sum(spectra[i]**2) * (self.frequencies[1] - self.frequencies[0])
                time_domain_power = Vrms[i]**2 * self.n_samples * 1/self.sampling_rate
                spectra[i,:] = spectra[i,:] / np.sqrt(spectrum_power) * np.sqrt(time_domain_power)
        
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
            for i in range(self.n_antennas):
                if self.matrix_inversion_method == "pseudo_inv":
                    self.cov_inv[i,:,:] = scp.linalg.pinvh(cov[i,:,:], atol=self.threshold_amplitude)
                elif self.matrix_inversion_method == "regular_inv":
                    self.cov_inv[i,:,:] = np.linalg.inv(cov[i,:,:] + np.diag(np.ones(self.n_samples) * cov[i,0,0] * self.increase_cov_diagonal))
                else:
                    raise Exception("""matrix_inversion_method not recognized. Choose "pseudo_inv" or "regular_inv" """)
        
        # Calculate log-determinant of covariance matrix:
        if not self.ignore_llh_normalization:
            self.cov_log_det = np.zeros(self.n_antennas)
            for i in range(self.n_antennas):
                if self.matrix_inversion_method == "pseudo_inv":
                    eigen_values = np.linalg.eigvalsh(cov[i,:,:])
                    self.cov_log_det[i] = np.sum(np.log(eigen_values[eigen_values > self.threshold_amplitude]))
                elif self.matrix_inversion_method == "regular_inv":
                    self.cov_log_det[i] = np.linalg.det(cov[i,:,:])
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
                spectra (average of Fourier transforms of events) for each antenna with dimensions [n_antennas,n_frequenciess]
        """
        spectra = np.zeros([self.n_antennas,self.n_frequencies])
        for i in range(self.n_antennas):
            fourier_transforms = fft.time2freq(data[:,i,:], self.sampling_rate)
            fourier_transforms_mean = np.sqrt(np.mean(fourier_transforms.real**2 + fourier_transforms.imag**2, axis=0))
            spectra[i,:] = fourier_transforms_mean
        
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

        for i in range(self.n_antennas):
            active_amplitudes = amplitudes[i, amplitudes[i,:] > self.threshold_amplitude]
            active_frequencies = self.frequencies[amplitudes[i,:]  > self.threshold_amplitude]

            # Calculate first row of covariance matrix:
            covariance_one_row = np.zeros(self.n_samples)
            covariance_inverse_one_row = np.zeros(self.n_samples)
            for j in np.arange(0, len(active_frequencies)):
                covariance_one_row +=  2 * active_amplitudes[j]**2 * np.cos(2*np.pi * active_frequencies[j] * self.t_array ) / self.n_samples
                covariance_inverse_one_row += 2 * (1/active_amplitudes[j]**2) * np.cos(2*np.pi * active_frequencies[j] * self.t_array) / self.n_samples

            # Construct covariances matrix and its incerse assuming they are circulant:
            for j in range(self.n_samples):
                covariance_matrices[i,:,j] = np.roll(covariance_one_row, j)
                covariance_matrices_inverse[i,:,j] = np.roll(covariance_inverse_one_row, j)

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

        for i in range(self.n_antennas):
            covariance_matrix = np.cov(data[:,i,:].T)

            # Calculate averages along diagonals:
            shifted_cov_matrix = np.zeros([self.n_samples, self.n_samples])
            for j in range(self.n_samples):
                shifted_cov_matrix[:,j] = np.roll(covariance_matrix[:, j], -j)
            covariances_avg = np.mean(shifted_cov_matrix, axis=1)

            # Make sure covariance matrix is symmetric (no numerical errors):
            for k in range(int(self.n_samples / 2)):
                covariances_avg[-k] = covariances_avg[k]

            # Construct covariances matrix:
            averaged_covariance_matrix = np.zeros([self.n_samples, self.n_samples])
            for j in range(self.n_samples):
                averaged_covariance_matrix[:,j] = np.roll(covariances_avg, j)

            covariance_matrices[i,:,:] = averaged_covariance_matrix

        return covariance_matrices
        self._set_covariance_matrices(covariance_matrices)

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
        term_2 = -0.5 * np.sum(np.log(noise_psd[noise_psd>self.threshold_amplitude])) #-0.5 * self.cov_log_det[0]
        integrand = abs(x_minus_mu_fft*x_minus_mu_fft.conj())/noise_psd
        term_3 = -0.5 * 4*np.sum(integrand[noise_psd>self.threshold_amplitude]) * (self.frequencies[1]-self.frequencies[0])
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
            data = data[:,np.newaxis,:]
        if self.n_antennas != 1 and len(data.shape) == 2:
            data = data[np.newaxis,:,:]
        if self.n_antennas == 1 and len(data.shape) == 1:
            data = data[np.newaxis,np.newaxis,:]

        if signal is None:
            means = np.zeros([self.n_antennas, self.n_samples])
        elif len(signal.shape) == 2:
            means = signal
        elif self.n_antennas == 1 and len(signal.shape) == 1:
            means = signal[np.newaxis,:]

        n_datasets = len(data)

        LLH_best = 0
        for i in range(self.n_antennas):
            if not frequency_domain:
               LLH_best += self._log_multivariate_normal(x=means[i,:], mu=means[i,:], cov_inv=self.cov_inv[i,:,:], cov_log_det=self.cov_log_det[i])
            elif frequency_domain:
                LLH_best += self._log_multivariate_normal_freq(x_minus_mu_fft=np.zeros(self.n_frequencies), noise_psd=self.noise_psd[i])

        LLH_array = np.zeros(n_datasets)

        for i in range(n_datasets):
            # Sum over likelihood for all antennas:
            LLH = 0
            for j in range(self.n_antennas):
                if not frequency_domain:
                    LLH += self._log_multivariate_normal(x=data[i,j,:], mu=means[j,:], cov_inv=self.cov_inv[j,:,:], cov_log_det=self.cov_log_det[j])
                if frequency_domain:
                    x_minus_mu_fft = fft.time2freq(data[i,j,:]-means[j,:], self.sampling_rate)
                    LLH += self._log_multivariate_normal_freq(x_minus_mu_fft=x_minus_mu_fft, noise_psd=self.noise_psd[j])
            
            LLH_array[i] = LLH

        return LLH_array - LLH_best
    
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
            LLH_signal = self.calculate_delta_llh(signal, data_to_fit, frequency_domain=frequency_domain)
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

        for i, file in enumerate(filenames):
            cov_sample_rate, cov_n_samples, cov_one_row, spectrum = np.load(file, allow_pickle=True)

            # Until resampling is implementet, only matching sampling rates and number of samples are allowed:
            assert cov_sample_rate == self.sampling_rate, f"Sampling rate ({self.sampling_rate}) does not match covariance matrix sampling rate ({cov_sample_rate})"
            assert cov_n_samples == self.n_samples, f"Number of samples ({self.n_samples}) does not match covariance matrix number of samples ({cov_n_samples})"
            
            # Construct covariances matrix assuming it is circulant:
            constructed_covariance_matrix = np.zeros([self.n_samples, self.n_samples])
            for j in range(self.n_samples):
                constructed_covariance_matrix[:,j] = np.roll(cov_one_row, j)

            covariance_matrices[i,:,:] = constructed_covariance_matrix
            spectra[i] = spectrum

        self._set_covariance_matrices(covariance_matrices, spectra=spectra)
    
    def resample_covariance_matrices(self, new_sampling_rate):
        """
        Resample the covariance matrices to new sampling rate

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

        for i in range(self.n_antennas):
            cov_one_row = self.cov[i,0,:]
            
            # Resample covariance matrix and spectra:
            cov_one_row, t_array = scp.signal.resample(cov_one_row, n_samples_new, t=self.t_array)
            spectra[i] = scp.signal.resample(self.spectra[i], n_frequencies_new)

            # Construct covariances matrix:
            covariance_matrix = np.zeros([n_samples_new, n_samples_new])
            for j in range(n_samples_new):
                covariance_matrix[:,j] = np.roll(cov_one_row, j)

            covariance_matrices[i,:,:] = covariance_matrix

        self._set_covariance_matrices(covariance_matrices, spectra)

    def calculate_fisher_information_matrix(self, signal_function, paramters_x0, dx, frequency_domain=False):
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

        Returns
        -------
            fisher_information_matrix : numpy.array
                Fisher information matrix for the parameters given the noise model

        """
        n_parameters = len(paramters_x0)

        # Calculate derivatives:
        derivatives = np.zeros([n_parameters, self.n_antennas, self.n_samples])
        derivatives_fft = np.zeros([n_parameters, self.n_antennas, self.n_frequencies])
        signal_0 = signal_function(paramters_x0)
        for i in range(n_parameters):
            paramters_x1 = np.copy(paramters_x0)
            paramters_x1[i] = paramters_x0[i] + dx[i]
            derivatives[i,:,:] = (signal_function(paramters_x1) - signal_0)/dx[i]
            if frequency_domain:
                derivatives_fft[i,:,:] = fft.time2freq(derivatives[i,:,:], self.sampling_rate)
        
        # Calculate Fisher information matrix
        fisher_information_matrix = np.zeros([n_parameters,n_parameters])
        for i in range(n_parameters):
            for j in range(n_parameters):
                for k in range(self.n_antennas):
                    if not frequency_domain:
                        fisher_information_matrix[i,j] += np.matmul(derivatives[i,k,:], np.matmul(self.cov_inv[k,:,:], derivatives[j,k,:]))
                    elif frequency_domain:
                        integrand = abs(derivatives_fft[i,k,:]*derivatives_fft[j,k,:].conj())/self.noise_psd
                        fisher_information_matrix[i,j] += 4*np.sum(integrand[self.noise_psd>self.threshold_amplitude]) * (self.frequencies[1]-self.frequencies[0])
        
        return fisher_information_matrix

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

    def plot_llh_distribution(self, data, n_dof=None, frequency_domain=False, make_new_figure = True):
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
        LLH_array = self.calculate_delta_llh(data, frequency_domain=frequency_domain)

        n_datasets = len(data)
        if n_dof is None:
            n_dof = int(self.n_antennas * self.n_samples)

        if make_new_figure:
            plt.figure(figsize=[4.2,3])

        plt.hist(-2*LLH_array, bins=int(n_datasets/20), histtype="step", color="b", label=str(n_datasets) + " datasets", density=True)

        axis = plt.axis()

        chi2 = scp.stats.chi2
        x = np.linspace(n_dof * 0.1, n_dof * 10, 10000)
        plt.plot(x, chi2.pdf(x, n_dof), "r-", alpha=0.6, label="chi2, dof = "+str(n_dof))

        plt.axis([np.min([axis[0], n_dof * 0.9]), np.max([axis[1], n_dof * 1.2]), axis[2], axis[3]])

        plt.xlabel(f"$-2\Delta LLH$")
        plt.ylabel("Density")
        plt.legend(loc=1)
        plt.tight_layout()
