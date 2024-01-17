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
        spectra : numpy.ndarray, optional
            Spectra of the noise in the antennas. Has dimensions [n_antennas,n_samples/2+1]. Default value is None, 
            and can instead be calculated using NoiseModel.calculate_spectra_from_data.
        add_white_noise : float, optional
            Calculating the inverse of a (covariance) matrix is numerically unstable and is impossible if the matrix 
            is not full rank. By adding a small component to the diagonal of the covariance matrix (corresponding to 
            white noise) it is guaranteed to be full rank, and the inverse can be calculated. This parameter should 
            neither be too large or too small.
    """

    def __init__(self, n_antennas, n_samples, sampling_rate, spectra=None, add_white_noise=1e-12):
        self.n_antennas = n_antennas
        self.n_samples = n_samples
        self.sampling_rate = sampling_rate
        self.spectra = spectra
        self.frequencies = np.fft.rfftfreq(n_samples, 1.0/sampling_rate)
        self.n_frequencies = len(self.frequencies)
        self.cov = None
        self.cov_inv = None
        self.cov_det = None
        self.add_white_noise = add_white_noise
        self.t_array = np.arange(n_samples) * 1.0 / sampling_rate

        if spectra is not None:
            assert np.shape(spectra)[1] == self.n_frequencies, "The dimensionality of the provided spectrum does not match the number of samples per trace"

    def _set_covariance_matrices(self, cov):
        """
        Sets the covariance matrices, their inverses, and log-determinants (for optimimization reasons)

        Parameters
        ----------
            cov : numpy.ndarray
                Covariance matrices for all antennas. Has dimensions [n_antennas,n_samples,n_samples]
        """
        self.cov = cov
        self.cov_inv = np.zeros([self.n_antennas, self.n_samples, self.n_samples])
        self.cov_det = np.zeros(self.n_antennas)
        for i in range(self.n_antennas):
            self.cov_inv[i,:,:] = np.linalg.inv(cov[i,:,:] + np.diag(np.ones(self.n_samples) * self.add_white_noise))
            self.cov_det[i] = np.linalg.slogdet(cov[i,:,:])[0]

    def calculate_spectra_from_data(self, data):
        """
        Calculates the spectra for all antennas

        Parameters
        ----------
            data : numpy.ndarray
                Array containing data with dimensions [n_datasets,n_antennas,n_samples]
        """
        spectra = np.zeros([self.n_antennas,self.n_frequencies])
        for i in range(self.n_antennas):
            fourier_transforms = fft.time2freq(data[:,i,:], self.sampling_rate)
            fourier_transforms_mean = np.sqrt(np.mean(fourier_transforms.real**2 + fourier_transforms.imag**2, axis=0))
            spectra[i,:] = fourier_transforms_mean
        
        self.spectra = spectra

    def calculate_covariance_matrices_from_spectra(self):
        """
        Calculates covariance matrices from self.spectra

        """

        assert self.spectra is not None, "Spectra are not set"

        # The normalization convention of the Fourier transform in NuRadioReco has to be taken into account:
        amplitudes = self.spectra * self.sampling_rate * np.sqrt(2) / self.n_samples

        covariance_matrices = np.zeros([self.n_antennas, self.n_samples, self.n_samples])

        for i in range(self.n_antennas):
            # Calculate first row of covariance matrix:
            covariance_total = np.zeros(self.n_samples)
            for j in np.arange(0, self.n_frequencies):
                covariance = 0.5 * amplitudes[i,j]**2 * np.cos(self.frequencies[j]*(2*np.pi)*self.t_array)
                covariance_total += covariance

            # Make sure covariance matrix is symmetric:
            # for i in range(int(n_samples/2)): cov_total[-i] = cov_total[i]

            # Construct covariances matrix assuming it is circulant:
            constructed_covariance_matrix = np.zeros([self.n_samples, self.n_samples])
            for j in range(self.n_samples):
                constructed_covariance_matrix[:,j] = np.roll(covariance_total, j)

            covariance_matrices[i,:,:] = constructed_covariance_matrix

        self._set_covariance_matrices(covariance_matrices)

    def calculate_covariance_matrices_from_data(self, data):
        """
        Calculates the covariance matrix for each antenna using numpy.cov and averages along the diagonals 
        assuming the covariance matrix is circulant

        Parameters
        ----------
            data : numpy.ndarray
                Array containing data with dimensions [n_datasets,n_antennas,n_samples]
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

        self._set_covariance_matrices(covariance_matrices)

    def _log_multivariate_normal(self, x, mu, cov_inv, cov_det):
        """
        Calculates the multivariate normal probability of vector x, given means mu and covariance matrix 
        inverse and determinant.
        """
        n = len(x)
        term_1 = -0.5 * n * np.log(2*np.pi)
        term_2 = -0.5 * cov_det
        # np.matmul sometimes returns a np.matrix instead of np.array, which
        # needs at least 2 dimensions. So we need to cast and flatten it:
        term_3_temp = np.array(np.matmul(cov_inv, x-mu)).flatten()
        term_3 = -0.5 * np.matmul(x-mu, term_3_temp)
        return term_1 + term_2 + term_3

    def calculate_llh(self, data, signal=None):
        """
        Calculates delta log likelihood for the datasets relative to the most probable noise

        Parameters
        ----------
            data : numpy.ndarray
                Array containing data with dimensions [n_datasets,n_antennas,n_samples] or [n_antennas,n_samples]
            signal : numpy.ndarray, optional
                Array containing neutrino signal signal of dimensions [n_antennas,n_samples].
                If no signal is provided, it will be set to zeros.
        """
        if signal is None:
            means = np.zeros([self.n_antennas, self.n_samples])
        else:
            means = signal

        # If only one dataset is given, add dimension along axis=0 ([1,n_antennas,n_samples]):
        if len(np.shape(data)) == 2:
            data = np.array([data])

        n_datasets = len(data)

        LLH_best = 0
        for i in range(self.n_antennas):
            LLH_best += self._log_multivariate_normal(x=means[i,:], mu=means[i,:], cov_inv=self.cov_inv[i,:,:], cov_det=self.cov_det[i])

        LLH_array = np.zeros(n_datasets)

        for i in range(n_datasets):
            # Sum over likelihood for all antennas:
            LLH = 0
            for j in range(self.n_antennas):
                LLH += self._log_multivariate_normal(x=data[i,j,:], mu=means[j,:], cov_inv=self.cov_inv[j,:,:], cov_det=self.cov_det[j])
            LLH_array[i] = LLH

        return LLH_array - LLH_best

    def get_minus_two_delta_llh_function(self, data_to_fit, signal_function):
        """
        Convenience function. Returns a function that calculates the minus two delta log likelihood for signal given a dataset with noise.
        The returned function can be used directly in optimizers like scipy.optimize.minimize.

        Parameters
        ----------
            data_to_fit : numpy.ndarray
                Array containing one dataset with dimensions [n_antennas,n_samples]
            signal_function : function
                Function which takes a vector containing parameters as input and returns a signal array with dimensions [n_antennas,n_samples].
        """

        LLH_best = self.calculate_llh(data_to_fit, data_to_fit)

        def llh_func(params):
            signal = signal_function(*params)
            LLH_signal = self.calculate_llh(signal, data_to_fit)
            two_delta_llh_signal = -2 * (LLH_signal - LLH_best)
            return two_delta_llh_signal

        return llh_func
    
    def save_covariance_matrix(self, antenna, filename):
        """
        Save compressed version (half of one row) of covariance matrix for one antenna along with the sample rate in GHz and number of samples

        Parameters
        ----------
            antenna : int
                Which antenna to save the covariance matrix for
            filename : str
                Name of file. Should follow the naming scheme:
                noise_covariance_matrix_<experiment>_<station_id>_<antenna_id>_<date_from>_to_<date_to>
                e.g.: noise_covariance_matrix_RNOG_station21_antenna6_2023-03-01_to_2023-10-31
        """

        if self.n_samples%2 == 0:
            cov_compressed = self.cov[antenna,:int(self.n_samples/2),0]
        if self.n_samples%2 == 1:
            cov_compressed = self.cov[antenna,:int(self.n_samples/2)+1,0]
        object_to_save = np.array([self.sampling_rate, self.n_samples, cov_compressed],dtype=object)
        np.save(filename, object_to_save)

    def load_covariance_matrices(self, filenames):
        """
        Load compressed version (half of one row) of covariance matrices for a all antennas (self.n_antennas) along with the sample rate in GHz. The full covariance matrices are then constructed and assigned to self.cov[:,:,:].

        Parameters
        ----------
            filenames : list[str]
                List of filenames which contain the covariance matrices. Should be of length self.n_antennas.
        """
        assert len(filenames) == self.n_antennas, f"Number of filenames ({len(filenames)}) does not match the number of antennas ({self.n_antennas})"

        covariance_matrices = np.zeros([self.n_antennas, self.n_samples, self.n_samples])

        for i, file in enumerate(filenames):
            cov_sample_rate, cov_n_samples, cov_compressed = np.load(file, allow_pickle=True)

            # Until resampling is implementet, only matching sampling rates and number of samples are allowed:
            assert cov_sample_rate == self.sampling_rate, f"Sampling rate ({self.sampling_rate}) does not match covariance matrix sampling rate ({cov_sample_rate})"
            assert cov_n_samples == self.n_samples, f"Number of samples ({self.n_samples}) does not match covariance matrix number of samples ({cov_n_samples})"

            if self.n_samples%2 == 0:
                cov_one_row = np.zeros(self.n_samples)
                cov_one_row[:int(self.n_samples/2)] = cov_compressed
                cov_one_row[int(self.n_samples/2):] = np.flip(cov_compressed)
            if self.n_samples%2 == 1:
                cov_one_row = np.zeros(self.n_samples)
                cov_one_row[:int(self.n_samples/2)+1] = cov_compressed
                cov_one_row[int(self.n_samples/2)+1:] = np.flip(cov_compressed[:-1])
            
            # Construct covariances matrix assuming it is circulant:
            constructed_covariance_matrix = np.zeros([self.n_samples, self.n_samples])
            for j in range(self.n_samples):
                constructed_covariance_matrix[:,j] = np.roll(cov_one_row, j)

            covariance_matrices[i,:,:] = constructed_covariance_matrix

        self._set_covariance_matrices(covariance_matrices)

    ### Plotting: ###

    def plot_data(self, data, plot_range=None, linestyle_and_color = "auto", make_new_figure=True):
        """
        Plots a dataset containing noise

        Parameters
        ----------
            data : numpy.array
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
            cov : numpy.array
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
            cov : numpy.array
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

    def plot_llh_distribution(self, data, make_new_figure = True):
        """
        Calculate the llh values for many datasets and plot the distribution alongside a chi2 
        distribution with dof equal to the number of samples

        Parameters
        ----------
            data : numpy.array
                Array containing data with dimensions [n_samples]
            make_new_figure : bool
                If True create a new figure
        """
        LLH_array = self.calculate_llh(data)

        n_datasets = len(data)
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
