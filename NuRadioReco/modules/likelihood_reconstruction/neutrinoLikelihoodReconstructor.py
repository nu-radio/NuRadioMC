from NuRadioReco.modules.base.module import register_run

import os
import numpy as np
import matplotlib.pyplot as plt
import copy

from NuRadioReco.utilities.analytic_pulse import get_analytic_pulse_freq
from NuRadioReco.utilities import units, fft, minimization, matched_filter, trace_utilities
from NuRadioReco.framework.electric_field import ElectricField
from NuRadioReco.framework.sim_station import SimStation
from NuRadioReco.framework.event import Event
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import stationParameters as stnp
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.electricFieldBandPassFilter
import NuRadioReco.modules.channelLengthAdjuster
from NuRadioReco.modules.likelihood_reconstruction import likelihood_calculator
from NuRadioReco.modules.likelihood_reconstruction.shower_simulator import ShowerSimulator
from radiotools import helper as hp
from radiotools import coordinatesystems

efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin(debug=False, pre_pulse_time=200*units.ns, post_pulse_time=200*units.ns, caching=False)
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelBandPassFilter.begin()
electricFieldBandPassFilter = NuRadioReco.modules.electricFieldBandPassFilter.electricFieldBandPassFilter()
channelLengthAdjuster = NuRadioReco.modules.channelLengthAdjuster.channelLengthAdjuster()

import logging
logger = logging.getLogger('NuRadioReco.neutrinoLikelihoodReconstructor')


class neutrinoLikelihoodReconstructor:
    """
    Class for reconstructing a neutrino shower in a station. This class forward folds a simulated hadronic
    shower E-field calculated using the Alvares2009 parameterization through the detector response and compares
    it to a measured set of data traces in a likelihood objective function. The -2DeltaLLH is minimized in two
    stages, first using a matched filter to fit the shape of the signal and second a -2DeltaLLH minimization 
    to fine-tune the reconstructed parameters. The likelihood is calculated using the spectrum of the noise, which
    enables correct error estimates of reconstructed parameters.

    This class is similar to voltageToAnalyticEfieldConverter, but uses a likelihood based on the
    noise spectrum instead of a chi-square and has an improved minimization strategy.

    The class assumes that the hardware response is subtracted from the data, e.g.,
    hardwareResponseIncorporator.run(event, station, det, sim_to_data=False, mingainlin=0.001) has been run.

    For a full description of the method, see https://arxiv.org/abs/2510.21925.
    """

    def __init__(self):
        pass

    def begin(self, n_channels, n_samples, sampling_rate, noise_spectra, Vrms, config_file=None, detector_simulation_filter_amp=None, use_chi2=False, debug=False):
        """

        Parameters
        ----------
            n_channels: int
                Number of channels to be used in the reconstruction

            n_samples: int
                Number of samples in the traces

            sampling_rate: float
                Sampling rate of the traces

            noise_spectra: np.ndarray
                Noise spectrum for each channel to be used for the likelihood calculation, i.e., sqrt(mean(abs(rfft(noise_traces))^2)).
                The overall normalizations of the spectra are ignored and set through the parameter Vrms.

            Vrms: float
                RMS of the noise in each channel. Used for the likelihood calculation.

            detector_simulation_filter_amp: function, optional
                Function to apply filter amplification in the detector simulation.

            use_chi2: bool, optional
                Whether to use chi2 minimization instead of likelihood. Mostly used for debugging and method comparison.

            debug: bool, optional
                Extra plots and printouts for debugging
        """

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.sampling_rate = sampling_rate
        self.Vrms = Vrms
        self.use_chi2 = use_chi2
        self.debug = debug
        self.config_file = os.path.join(os.path.dirname(__file__), 'signal_model_config.yaml') if config_file is None else config_file

        self.delta_t = 1/self.sampling_rate
        self.t_array_matched_filter = np.arange(0, self.n_samples) * self.delta_t - self.n_samples * self.delta_t/ 2
        self.i_shift_cc = np.arange(0, self.n_samples)
        self.frequencies = np.fft.rfftfreq(self.n_samples, 1. / self.sampling_rate)

        # initialize likelihood calculator:
        self.likelihood_calculator = likelihood_calculator.LikelihoodCalculator(
            n_antennas = self.n_channels,
            n_samples = self.n_samples,
            sampling_rate = self.sampling_rate,
            matrix_inversion_method = "pseudo_inv",
            threshold_amplitude = 0.1
        )
        self.likelihood_calculator.initialize_with_spectra(noise_spectra, self.Vrms)
        self.noise_psd = self.likelihood_calculator.noise_psd

        # initialize matched filter:
        self.matched_filter = matched_filter.MatchedFilter(
            n_samples = self.n_samples,
            sampling_rate = self.sampling_rate,
            n_antennas = self.n_channels,
            noise_power_spectral_density = self.noise_psd,
            spectra_threshold_fraction = 0.1
        )

        self.detector_simulation_filter_amp = detector_simulation_filter_amp

    @register_run()
    def run(self, evt, station, det, parameters_initial, charge_excess_profile_id=0, use_channels=None, reference_channel=0, full_output=True):
        """
        Run the likelihood reconstruction of electric field.

        Parameters
        ----------
        evt: NuRadioReco.framework.event.Event
            The event to run the module on.

        station: NuRadioReco.framework.station.Station
            The station object containing the channels with the data traces.

        det: NuRadioReco.framework.detector.Detector
            The detector description.
        
        parameters_initial: np.ndarray
            Initial parameters for the reconstruction. Should be an array of
            length 7 containing the following parameters in this order:
            [energy, zenith, azimuth, vertex_r, vertex_theta, vertex_phi, vertex_time]
            in standard NuRadioMC units.

        charge_excess_profile_id: int, optional
            ID of the charge excess profile to use in the shower simulation. Not used for Alvarez2009. Default: 0

        use_channels: list, optional
            List of channel IDs to be used for the reconstruction. If None, all channels are used.

        full_output: bool, optional
            If True, return the reconstructed signal, the signal parameters and the minus two
            log-likelihood of the reconstructed signal. Default: False

        Returns
        -------
        fitted_signal: np.ndarray
            The reconstructed signal in the readout traces.
            Only returned if `full_output` enabled

        fitted_params_best: np.ndarray
            The best fit parameters of the signal model.
            Only returned if `full_output` enabled

        minus_two_llh_best: float
            The minus two log-likelihood value of the reconstructed signal.
            Only returned if `full_output` enabled
        
        uncertainties_fit: np.ndarray
            Estimated marginalized uncertainties on the 7 reconstructed parameters using the Fisher
            information matrix
        """

        if use_channels is None:
            use_channels = station.get_channel_ids()

        traces = []
        trace_start_times = []
        for channel in station.iter_channels():
            if channel.get_id() not in use_channels:
                continue
            traces.append(channel.get_trace())
            trace_start_times.append(channel.get_trace_start_time())
        traces = np.array(traces)

        assert len(use_channels) == self.n_channels, "Number of channels in use_channels does not match n_channels in begin()"
        assert traces.shape[-1] == self.n_samples, "Number of samples in traces does not match n_samples in begin()"
        assert channel.get_sampling_rate() == self.sampling_rate, "Sampling rate of channel does not match sampling rate in begin()"

        # Define signal function:
        signal_model = ShowerSimulator(
            config_file = self.config_file,
            det = det,
            station_id = station.get_id(),
            reference_channel = reference_channel,
            evt_time = station.get_station_time(),
            use_channels = use_channels,
            detector_simulation_filter_amp = self.detector_simulation_filter_amp,
            pre_pulse_time = 100 * units.ns # not used
        )
        # convert vertex time to pulse time (time at which the pulse reaches the reference channel):
        # start_point = hp.spherical_to_cartesian(parameters_initial[4], parameters_initial[5]) * parameters_initial[3]
        # end_point = det.get_relative_position(station.get_id(), reference_channel)
        # signal_model.propagator.set_start_and_end_point(start_point, end_point)
        # signal_model.propagator.find_solutions()
        # reference_travel_time = signal_model.propagator.get_travel_time(0)
        # parameters_initial[6] = parameters_initial[6] + reference_travel_time + det.get_cable_delay(station.get_id(), reference_channel) - trace_start_times[reference_channel]

        def signal_model_wrapper(parameters):
            signal = signal_model.simulate_single_shower_forward_folding(
                energy = parameters[0],
                zenith = parameters[1],
                azimuth = parameters[2],
                vertex_r = parameters[3],
                vertex_theta = parameters[4],
                vertex_phi = parameters[5],
                pulse_time = parameters[6],
                type = "HAD",
                trace_start_times = trace_start_times,
                charge_excess_profile_id = charge_excess_profile_id,
            )[1]
            return signal

        #self.matched_filter.set_data(traces)

        if self.debug:
            # plot initial signal for debugging:
            signal_initial = signal_model_wrapper(parameters_initial)
            t_array = trace_start_times[0] + np.arange(0, self.n_samples) * self.delta_t
            fig, ax = plt.subplots(self.n_channels, 1, figsize=(10, self.n_channels*3))
            for i_ch in range(self.n_channels):
                ax[i_ch].plot(t_array, traces[i_ch], label="data")
                ax[i_ch].plot(t_array, signal_initial[i_ch], ls="--", label="initial")
                ax[i_ch].set_ylabel("Voltage [V]")
            ax[0].legend()
            ax[-1].set_xlabel("Time [s]")
            plt.tight_layout()
            plt.savefig("debug_StationElectricFieldReconstructor_initial.png")
            plt.show()
            plt.close()

        initial_likelihood, minus_two_llh, fitted_parameters, uncertainties_fit = self._reconstruct_signal(traces, signal_model_wrapper, parameters_initial)

        fitted_signal = signal_model_wrapper(fitted_parameters)

        # save results to station object:
        # if self.travel_time_shifts is None:
        #     efield_time = fitted_params_best[4] - det.get_cable_delay(station.get_id(), use_channels[0])
        # elif self.travel_time_shifts is not None:
        #     efield_time = fitted_params_best[4] - det.get_cable_delay(station.get_id(), use_channels[0]) - self.travel_time_shifts[0]
        # efield_parameters = np.array([fitted_params_best[0], fitted_params_best[1], fitted_params_best[2], fitted_params_best[3], efield_time, fitted_params_best[5]])
        # electric_field = self._get_efield(efield_parameters, fitted_params_best[6], fitted_params_best[7], use_channels, apply_filter=save_filtered_efield)
        # electric_field.set_parameter(efp.signal_energy_fluence, fluence_reco_best)
        # electric_field.set_parameter_error(efp.signal_energy_fluence, fluence_uncertainty_best)
        # electric_field.set_parameter(efp.polarization_angle, polarization_reco_best)
        # electric_field.set_parameter_error(efp.polarization_angle, polarization_uncertainty_best)
        # electric_field.set_parameter(efp.cr_spectrum_slope, fitted_params_best[2])
        # electric_field.set_parameter(efp.signal_time, trace_start_times[0] + fitted_params_best[3])
        # electric_field.set_parameter(efp.cr_spectrum_quadratic_term, fitted_params_best[5])
        # electric_field.set_parameter(efp.zenith, fitted_params_best[6])
        # electric_field.set_parameter(efp.azimuth, fitted_params_best[7])

        # station.add_electric_field(electric_field)


        # Convert fitted parameters to vertex time instead of pulse time for output:
        # fitted_parameters[6] = fitted_parameters[6] + trace_start_times[reference_channel] - det.get_cable_delay(station.get_id(), reference_channel) - reference_travel_time
        # parameters_initial[6] = parameters_initial[6] + trace_start_times[reference_channel] - det.get_cable_delay(station.get_id(), reference_channel) - reference_travel_time

        if full_output:
            return initial_likelihood, fitted_signal, fitted_parameters, minus_two_llh, uncertainties_fit

    def _function_to_minimize_mf(self, data, signal):
        """
        Calculate the objective function for the first minimization.
        """

        if not self.use_chi2:
            self.matched_filter.set_template(signal)
            t_best, x_best = self.matched_filter.matched_filter_search(time_shift_array=self.t_array_matched_filter)
            llh_mf = self.matched_filter.calculate_matched_filter_delta_log_likelihood()
            return -2 * llh_mf

        elif self.use_chi2:
            i_max, cross = self._cross_correlation(data, signal, shift_array=self.i_shift_cc)
            return -cross

    def _function_to_minimize_llh(self, data, signal):
        """
        Calculate the log-likelihood objective function of the 2nd minimization
        """
        if not self.use_chi2:
            minus_two_llh = self.likelihood_calculator.calculate_minus_two_delta_llh(data, signal)
            #print("Minus two delta LLH:", minus_two_llh)
            return minus_two_llh

        elif self.use_chi2:
            return self._chi2(data, signal)

    def _cross_correlation(self, data, signal, shift_array):
        """
        Calculate the cross-correlation between the data and the signal.

        Parameters
        ----------
        data: np.ndarray
            Data from the two antennas

        signal: np.ndarray
            Signal from the two antennas

        shift_array: np.ndarray
            Array of shift indicies to calculate the cross-correlation for

        Returns
        -------
        float
            Normalized cross-correlation between the data and the signal
        """

        cross_correlation_array = np.zeros(len(shift_array))
        for i, shift in enumerate(shift_array):
            cross_correlation_array[i] = np.sum(data[0,:] * np.roll(signal[0,:], shift)) + np.sum(data[1,:] * np.roll(signal[1,:], shift)) / np.sqrt(np.sum(data[0,:]**2) * np.sum(signal[0,:]**2) + np.sum(data[1,:]**2) * np.sum(signal[1,:]**2))

        cross = np.max(cross_correlation_array)
        i_max = shift_array[np.argmax(cross_correlation_array)]

        return i_max, cross

    def _chi2(self, data, signal):
        """
        Calculate the chi2 value between the data and the signal.
        """
        if isinstance(self.Vrms, np.ndarray) or isinstance(self.Vrms, list):
            sigma = self.Vrms[:,None]
        else:
            sigma = self.Vrms
        chi2 = np.sum((data - signal)**2 / sigma**2)
        return chi2


    def _reconstruct_signal(self, data, signal_function, parameters_initial):
        """
        Reconstruct the signal from the given data.

        Parameters
        ----------
        data: np.ndarray
            Data traces for the channels to be used in the reconstruction

        signal_function: callable
            Function to model the signal

        parameters_initial: np.ndarray
            Initial parameters for the reconstruction

        Returns
        -------
        minus_two_llh_fit: float
            The negative log-likelihood value for the reconstructed signal

        fitted_params: np.ndarray
            The fitted parameters for the reconstructed signal
        """

        initial_likelihood = self._function_to_minimize_llh(data, signal_function(parameters_initial))

        bounds = np.array([(1 * units.PeV, 100 * units.EeV),
                            (0 * units.deg, 180 * units.deg),
                            (-360 * units.deg, 360 * units.deg),
                            (20 *units.m, 5 * units.km),
                            (90 * units.deg, 180 * units.deg),
                            (-360 * units.deg, 360 * units.deg),
                            (parameters_initial[6] - 10 * units.ns, parameters_initial[6] + 10 * units.ns)])
        scaling = np.array([units.EeV, units.rad, units.rad, units.km, units.deg, units.deg, units.ns])

        minimizer_llh = minimization.Minimizer(
            signal_function = signal_function,
            objective_function = self._function_to_minimize_llh,
            parameters_initial = parameters_initial,
            parameters_bounds = bounds,
            debug=self.debug
        )
        minimizer_llh.set_scaling(scaling)

        m = minimizer_llh.run_minimization(data=data, method="minuit")

        fitted_params = minimizer_llh.parameters
        minus_two_llh_fit = minimizer_llh.result

        # Estimate 1st order uncertainties using the Fisher information matrix:
        dx = np.array([1e-6, 1e-6, 1e-6, 1e-4, 1e-6, 1e-6, 1e-4])
        def signal_function_scaled(params_scaled):
            params = params_scaled * scaling
            return signal_function(params)
        fisher_information_matrix_fit = self.likelihood_calculator.calculate_fisher_information_matrix(signal_function_scaled, fitted_params / scaling, dx)
        f_i_fit = np.linalg.pinv(fisher_information_matrix_fit)
        uncertainties_fit = np.sqrt(np.diag(f_i_fit)) * scaling

        # if self.debug:
        #     # plot results for debugging:
        #     signal_initial = signal_function(parameters_initial)
        #     signal_initial_2 = signal_function(parameters_initial_2)
        #     signal_fit_2 = signal_function(fitted_params_2)

        #     fig, ax = plt.subplots(self.n_channels, 1, figsize=(10, self.n_channels*3))
        #     for i_ch in range(self.n_channels):
        #         t_array = trace_start_times[i_ch] + np.arange(0, self.n_samples) * self.delta_t
        #         ax[i_ch].plot(t_array, data[i_ch], label="data")
        #         ax[i_ch].plot(t_array, signal_initial[i_ch], ls="--", label="initial")
        #         ax[i_ch].plot(t_array, signal_fit[i_ch], label="fit")
        #         ax[i_ch].plot(t_array, signal_fit_adjusted[i_ch], "--", label="fit adjusted")
        #         ax[i_ch].plot(t_array, signal_initial_2[i_ch], "y:", label="initial 2")
        #         ax[i_ch].plot(t_array, signal_fit_2[i_ch], "k:", label="fit 2")

        #         # Plot bounds (matched filter):
        #         t_max = t_array[np.argmax(signal_fit[i_ch])]
        #         ax[i_ch].vlines([t_max+self.t_array_matched_filter[0], t_max+self.t_array_matched_filter[-1]], np.min(data[i_ch]*2), np.max(data[i_ch]*2), color="r", ls="--", label="Bounds (matched filter)")

        #         # Plot bounds (LLH reconstruction):
        #         s0 = signal_function(np.array([fitted_params_2[i_ch], fitted_params_2[1], fitted_params_2[2], fitted_params_2[3], bounds[4][0], fitted_params_2[5], fitted_params_2[6], fitted_params_2[7]]))
        #         t_max_bound_0 = t_array[np.argmax(s0[i_ch])]
        #         s1 = signal_function(np.array([fitted_params_2[i_ch], fitted_params_2[1], fitted_params_2[2], fitted_params_2[3], bounds[4][1], fitted_params_2[5], fitted_params_2[6], fitted_params_2[7]]))
        #         t_max_bound_1 = t_array[np.argmax(s1[i_ch])]
        #         ax[i_ch].vlines([t_max_bound_0, t_max_bound_1], np.min(data[i_ch]*2), np.max(data[i_ch]*2), color="b", ls="--", label="Bounds (LLH fit)")

        #         ax[i_ch].set_ylabel("Voltage [V]")

        #     ax[0].legend()
        #     if not self.use_chi2:
        #         ax[0].set_title(f"$-2\Delta$LLH: {minus_two_llh_fit_2} \n parameters: {fitted_params_2}")
        #     else:
        #         ax[0].set_title(f"$\chi^2$: {minus_two_llh_fit_2} \n parameters: {fitted_params_2}")
        #     ax[-1].set_xlabel("Time [s]")
        #     plt.tight_layout()
        #     plt.savefig("debug_StationElectricFieldReconstructor.png")
        #     plt.show()
        #     plt.close()

        #     # Plot spectra of (assumed) noise and data:
        #     fig, ax = plt.subplots(self.n_channels, 1, figsize=(10, self.n_channels*3))
        #     for i_ch in range(self.n_channels):
        #         ax[i_ch].plot(self.frequencies, self.likelihood_calculator.spectra[i_ch], "k-", label="Likelihood noise spectrum")
        #         ax[i_ch].plot(self.frequencies, np.abs(fft.time2freq(data[i_ch], sampling_rate=self.sampling_rate)), "b-", label="data")
        #         ax[i_ch].plot(self.frequencies, np.abs(fft.time2freq(signal_initial[i_ch], sampling_rate=self.sampling_rate)), "r-", label="initial")
        #         ax[i_ch].plot(self.frequencies, np.abs(fft.time2freq(signal_fit_2[i_ch], sampling_rate=self.sampling_rate)), "g-", label="fit")
        #         ax[i_ch].hlines( np.max(self.likelihood_calculator.spectra[i_ch])/100, 0, max(self.frequencies), "m", "--", label="threshold")
        #         ax[i_ch].set_ylabel("Amplitude [V/GHz]")
        #         #ax[i].set_yscale("log")
        #     ax[0].legend()
        #     ax[-1].set_xlabel("Frequency [GHz]")
        #     fig.tight_layout()
        #     plt.savefig("debug_StationElectricFieldReconstructor_spectra.png")
        #     plt.show()
        #     plt.close()

        return initial_likelihood, minus_two_llh_fit, fitted_params, uncertainties_fit
