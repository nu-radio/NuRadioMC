import numpy as np
import matplotlib.pyplot as plt
from NuRadioReco.utilities import fft

import logging
logger = logging.getLogger("NuRadioReco.utilities.matched_filter")

class MatchedFilter:
    """
    Matched filter class for multiple antennas with equal sampling rate and trace length. Calculates
    the best matching time, matched filter SNR, and likelihood for a template [n_antennas, n_samples]
    to a data trace [n_antennas, n_samples] using the noise power spectral density.

    The class currently performs the matched filter search for one data event and one signal template
    at a time. To repeat the search for many datasets and/or templates, the user has to set up a loop
    and run set_data() and/or set_template() multiple times.

    See Appendix B in https://arxiv.org/abs/2510.21925 for more details.

    Parameters
    ----------
        n_samples: int
            Number of samples in the traces

        sampling_rate: float
            Sampling rate of the traces

        n_antennas: int
            Number of antennas (channels)

        data_traces: np.ndarray, optional
            Traces containing the data with shape [n_antennas, n_samples], or
            optionally [n_samples] for one antenna

        template_traces: np.ndarray, optional
            Traces containing the template with shape [n_antennas, n_samples], or
            optionally [n_samples] for one antenna

        noise_power_spectral_density: np.ndarray, optional
            Traces containing the noise power spectral density with shape [n_antennas, n_frequencies], or
            optionally [n_frequencies] for one antenna

        spectra_threshold_fraction: float, optional
            The fraction of the maximum of the noise spectra to be used as threshold. Frequencies
            with noise spectra below this threshold are ignored in the matched filter calculation.
    """
    def __init__(self, n_samples, sampling_rate, n_antennas, data_traces=None, template_traces=None, noise_power_spectral_density=None, spectra_threshold_fraction=0.01, debug=False):
        self.n_samples = n_samples
        self.sampling_rate = sampling_rate
        self.n_antennas = n_antennas
        self.df = 1. / n_samples * sampling_rate
        self.dt = 1. / sampling_rate

        frequencies = np.fft.rfftfreq(n_samples, self.dt)
        self.n_frequencies = len(frequencies)
        self.frequencies_flattened = np.tile(frequencies, n_antennas)

        self.spectra_threshold_fraction = spectra_threshold_fraction

        self._results_valid = False
        self.debug = debug

        logger.info("Matched Filter initialized with {} antennas, {} samples, and {} Hz sampling rate".format(n_antennas, n_samples, sampling_rate))

        if noise_power_spectral_density is not None:
            self.set_noise_psd(noise_power_spectral_density)
        else:
            logger.info("No noise power spectral density set. Run set_noise_psd(), set_noise_psd_from_data(), or set_noise_psd_from_spectra()")

        if data_traces is not None:
            self.set_data(data_traces)
        else:
            logger.info("No data traces set. Run set_data() to set data traces")

        if template_traces is not None:
            self.set_template(template_traces)
        else:
            logger.info("No template traces set. Run set_template() to set template traces")


    def set_data(self, data_traces):
        """
        Set the data traces (one event) and calculate the data normalization factor

        Parameters
        ----------
            data_traces: np.ndarray
                Traces containing the data with shape [n_antennas, n_samples], or
                optionally [n_samples] for one antenna
        """
        data_traces = np.atleast_2d(data_traces)
        assert data_traces.shape[0] == self.n_antennas, f"Data trace shape {data_traces.shape} does not match the number of antennas, {self.n_antennas}"
        assert data_traces.shape[1] == self.n_samples, f"Data trace shape {data_traces.shape} does not match the number of samples, {self.n_samples}"

        #self.data = data_traces
        self.data_fft = fft.time2freq(data_traces, self.sampling_rate).flatten()

        integrand_data = abs(self.data_fft * self.data_fft.conj()) / self.noise_psd
        self.data_factor = 4 * np.sum(integrand_data[self.noise_psd > self.noise_psd_threshold]) * self.df

        self._results_valid = False

    def set_template(self, template_traces):
        """
        Set the template traces (one event) and calculate the data normalization factor

        Parameters
        ----------
            template_traces: np.ndarray
                Traces containing the template with shape [n_antennas, n_samples], or
                optionally [n_samples] for one antenna
        """
        template_traces = np.atleast_2d(template_traces)
        assert template_traces.shape[0] == self.n_antennas, f"Template trace shape {template_traces.shape} does not match the number of antennas, {self.n_antennas}"
        assert template_traces.shape[1] == self.n_samples, f"Template trace shape {template_traces.shape} does not match the number of samples, {self.n_samples}"

        #self.template = template_traces
        self.template_fft = fft.time2freq(template_traces, self.sampling_rate).flatten()

        integrand_template = abs(self.template_fft * self.template_fft.conj()) / self.noise_psd
        self.template_factor = 4 * np.sum(integrand_template[self.noise_psd > self.noise_psd_threshold] * self.df)

        self._results_valid = False

    def set_noise_psd(self, noise_power_spectral_density):
        """
        Set the noise power spectral density, which is here defined as 2*mean(abs(fft.time2freq(noise))^2)*df
        and has units of V^2/GHz.

        See self.set_noise_psd_from_data and self.set_noise_psd_from_spectra for alternative ways to
        set the noise PSD.

        Parameters
        ----------
            noise_power_spectral_density: np.ndarray
                Traces containing the noise power spectral density with shape [n_antennas, n_frequencies], or
                optionally [n_frequencies] for one antenna
        """
        if noise_power_spectral_density.ndim == 1:
            self.noise_psd = np.tile(noise_power_spectral_density, self.n_antennas)
        elif noise_power_spectral_density.shape[0] == self.n_antennas:
            self.noise_psd = noise_power_spectral_density.flatten()
        else:
            raise ValueError("Noise power spectral density has wrong shape")

        self.noise_psd_threshold = np.max(self.noise_psd) * self.spectra_threshold_fraction**2

    def set_noise_psd_from_data(self, noise_traces):
        """
        Set the noise power spectral density using data containing purely noise.

        Parameters
        ----------
            noise_traces: np.ndarray
                Traces containing the noise with shape [n_events, n_antennas, n_samples]

        """
        assert noise_traces.shape[0] > 1, f"The noise PSD should be calculated using more than one noise trace"
        assert noise_traces.shape[1] == self.n_antennas, f"Noise trace shape {noise_traces.shape} does not match the number of antennas, {self.n_antennas}"
        assert noise_traces.shape[2] == self.n_samples, f"Noise trace shape {noise_traces.shape} does not match the number of samples, {self.n_samples}"

        noise_fft = fft.time2freq(noise_traces[:,:], self.sampling_rate)
        noise_psd = 2 * np.mean(abs( noise_fft * noise_fft.conj()), axis=0) * self.df

        self.set_noise_psd(noise_psd)

    def set_noise_psd_from_spectra(self, spectra, Vrms = None):
        """
        Calculate the noise power spectral density using the spectra of the noise defined
        as sqrt(mean(abs(fft.time2freq(noise))^2)). By specifying Vrms, the spectra normalizations
        of the spectra are ignored and rescale to the given Vrms values, and spectra can then be
        the normalized noise filters.

        Parameters
        ----------
            spectra: np.ndarray
                Spectra or filters of the noise with shape [n_antennas, n_frequencies], or
                optionally [n_frequencies] for one antenna
            Vrms: float or np.ndarray, optional
                The Vrms value(s) to which the spectra should be rescaled. If a float is given,
                all antennas are rescaled to the same Vrms. If an array is given it should have
                shape [n_antennas]. If None, the spectra normalizations are not changed.
        """
        noise_psd = np.zeros([self.n_antennas, self.n_frequencies])
        for i in range(self.n_antennas):
            if spectra.ndim == 1:
                spectra = np.tile(spectra, (self.n_antennas,1))
            noise_psd[i,:] = 2 * spectra[i,:]**2 * self.df

        # Scale to Vrms:
        if Vrms is not None:

            if Vrms.ndim == 0:
                Vrms = np.tile(Vrms, self.n_antennas)

            for i in range(self.n_antennas):
                freq_domain_power = np.sum(0.5 * noise_psd[i,:])
                time_domain_power = Vrms[i]**2 * self.n_samples * self.dt
                noise_psd[i,:] = noise_psd[i,:] / freq_domain_power * time_domain_power

        self.set_noise_psd(noise_psd)


    def matched_filter_search(self, time_shift_array, plot=False):
        """
        Perform the matched filter search for the template and the data to find the best matching time
        and matched filter output.

        The method stores the matched filter output for later calculations of the matched filter SNR and
        likelihood.

        Parameters
        ----------
            time_shift_array: np.ndarray
                Array of time shifts of the template to be tested. Note that often a grid with finer
                spacing than the sampling rate is needed.
            plot: bool (default: False)
                Whether to plot the matched filter output as a function of time shift

        Returns
        -------
            t_best: float
                The best matching time shift
            matched_filter_output: float
                The matched filter output at the best matching time shift
        """
        integrand = (self.data_fft * self.template_fft.conj() / self.noise_psd)[None, :] * np.exp(1j * 2*np.pi * self.frequencies_flattened[None, :] * time_shift_array[:, None])
        output = 4 * np.real( np.sum(integrand[:, self.noise_psd > self.noise_psd_threshold], axis=-1) ) * self.df

        self.matched_filter_output = np.max(output)

        self._results_valid = True

        if plot:
            plt.figure(figsize=[20,3])
            plt.plot(time_shift_array, output)
            plt.xlabel("Time [ns]")
            plt.ylabel("Matched filter output")
            plt.tight_layout()
            plt.show()

        if self.debug:
            plt.figure(figsize=[20,2])
            plt.plot(self.frequencies_flattened, (self.noise_psd/2/self.df)**(0.5) )
            plt.plot(self.frequencies_flattened, np.abs(self.data_fft), alpha = 0.5)
            plt.plot(self.frequencies_flattened, np.abs(self.template_fft))
            plt.hlines((self.noise_psd_threshold/2/self.df)**0.5, 0, np.max(self.frequencies_flattened), colors='gray', linestyles='dashed')
            plt.xlabel("Frequency [GHz]")
            plt.ylabel("Amplitude [V/√GHz]")
            plt.legend(["Noise PSD", "Data spectrum", "Template spectrum"])
            plt.tight_layout()
            plt.show()

        return time_shift_array[np.argmax(output)], self.matched_filter_output

    def calculate_matched_filter_SNR(self):
        """
        Calculate the matched filter SNR (or matched filter score) using the matched filter output and
        the template normalization factor. If SNR >> 1, the data is likely to contain the signal template.
        The matched filter SNR can be seen as the matched filter score and is the relevant quantity to compare
        between different templates, to find the best matching template.

        Returns
        -------
            SNR: float
                The matched filter SNR, i.e. the matched filter score
        """
        assert self._results_valid, "Calculated matched_filter_output is not valid, since either the template and data were re-defined after the matched_filter_search method was called."

        SNR = self.matched_filter_output / np.sqrt(self.template_factor)

        return SNR

    def calculate_matched_filter_delta_log_likelihood(self, relative_to = None):
        """
        Calculate the matched filter delta log likelihood  of the tempalte given the data using the matched
        filter output, the template normalization factor, and the data normalization factor. This is the 
        likelihood marginalized over the signal amplitude and time. 

        If relative_to is None, the the "delta" refers to that the constants in the log likelihood are
        omitted, and it can be seen as the log likelihood relative to the most probable template (the data
        itself). In this case, if the template describes the signal in the data and the noise PSD describes
        the noise, the delta log likelihood should follow a chi2 distribution with degrees of freedom equal
        to two times the number of noise_psd amplitudes above the threshold (see self.get_degrees_of_freedom()).

        If relative_to is "zeros", the delta log likelihood is calculated relative to a template with zeros, i.e.
        no signal. In this case, the delta log likelihood can be used to test the significance of the detected signal.

        Returns
        -------
            delta_log_likelihood: float
                The matched filter delta log likelihood (profiled over amplitude and time)
        """
        assert self._results_valid, "Calculated matched_filter_output is not valid, since either the template and data were re-defined after the matched_filter_search method was called."

        if relative_to is None:
            return  -1/2 * self.data_factor + 1/2 * self.matched_filter_output**2 / self.template_factor
        elif relative_to == "zeros":
            return  1/2 * self.matched_filter_output**2 / self.template_factor

    def calculate_matched_filter_amplitude_estimate(self):
        """
        Calculate the matched filter estimate of the amplitude, i.e., the factor the template
        has to be multiplied with to best match the data.

        Returns
        -------
            amplitude_estimate: float
                The matched filter amplitude estimate
        """
        assert self._results_valid, "Calculated matched_filter_output is not valid, since either the template and data were re-defined after the matched_filter_search method was called."

        return self.matched_filter_output / self.template_factor

    def get_degrees_of_freedom(self):
        """
        Get the degrees of freedom of the matched filter delta log likelihood, which is equal to
        two times the number of noise_psd amplitudes above the threshold.

        Returns
        -------
            dof: int
                The degrees of freedom of the matched filter delta log likelihood
        """
        return 2 * np.sum(self.noise_psd > self.noise_psd_threshold)
