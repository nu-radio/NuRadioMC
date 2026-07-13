from NuRadioReco.modules.base.module import register_run

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp

from NuRadioReco.utilities import units, fft, minimization, matched_filter
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import showerParameters as shp
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.electricFieldBandPassFilter
import NuRadioReco.modules.channelLengthAdjuster
from NuRadioReco.modules.likelihood_reconstruction import likelihood_calculator
from NuRadioReco.modules.likelihood_reconstruction.shower_simulator import ShowerSimulator
from radiotools import helper as hp

efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin(pre_pulse_time=200*units.ns, post_pulse_time=200*units.ns, caching=False)
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelBandPassFilter.begin()
electricFieldBandPassFilter = NuRadioReco.modules.electricFieldBandPassFilter.electricFieldBandPassFilter()
channelLengthAdjuster = NuRadioReco.modules.channelLengthAdjuster.channelLengthAdjuster()

import logging
logger = logging.getLogger('NuRadioReco.neutrinoLikelihoodReconstructor')


class neutrinoLikelihoodReconstructor:
    """
    Class for reconstructing a neutrino shower in a station. This class forward-folds a simulated hadronic
    shower E-field calculated using the Alvarez2009 parameterization through the detector response and compares
    it to a data traces in the provided station object using a likelihood objective function. The -2DeltaLLH is
    minimized in two stages, first using a matched filter (profiling over amplitude and time) to fit the shape
    of the signal and second a -2DeltaLLH minimization to fine-tune the reconstructed parameters. The likelihood
    is calculated using the spectrum of the noise, which enables correct error estimates of reconstructed parameters.

    In the current implementation of this class, the initial guess of the signal parameters needs to be very close
    to the true parameter values. A good guess for the vertex position can be found with interferometric vertex
    reconstruction modules. The time can be estimated from the peak of the trace of the reference antenna, and it is
    anyway profiled over in the first fitting stage. The energy probably doesn't need a good guess, since it is also
    profiled over analytically in the first fitting stage. Finally, educated guesses of the neutrino arrival direction
    can be obtained by assuming that we observe the signal close to the Cherenkov cone (i.e., signal launch angle + ~56 deg)

    For a full description of the method, see https://arxiv.org/abs/2510.21925.
    """

    def __init__(self):
        pass

    def begin(
            self,
            n_channels,
            n_samples,
            sampling_rate,
            noise_spectra,
            Vrms,
            config_file,
            detector_simulation_filter_amp = None,
            signal_search_width = 30 * units.ns,
            n_grid_matched_filter = 200,
            overwrite_speedup_options = True,
            use_chi2 = False,
            debug = False
            ):
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
                RMS of the noise in each channel. Used for the likelihood calculation

            config_file: string
                path to the simulation config file to use for the reconstruction signal model

            detector_simulation_filter_amp: function, optional
                Function to use as the _detector_simulation_filter_amp in the simulation class, e.g,
                hardware response and/or bandpass filter of the detector.

            signal_search_width: float, optional
                The width of the window to search for the signal in relative to the initial guess pulse_time. This is used as
                the width of the matched filter time grid, which will be profiled over in the first stage of the fit. And it
                is used as the bound in the final -2LLH minimization. If the peak of the signal in the trace is within the
                initial guess of the pulse_time +/- signal_search_width, the fit is likely to converge to a good minimum.
                Default: 30 * units.ns

            n_grid_matched_filter: int, optional
                Number of grid points to divide the matched filter time grid into. It is recommended that the matched filter time
                steps are at least a factor of 2 smaller than the detector sampling rate. Default: 200

            overwrite_speedup_options: bool, optional
                Some speedup options cause discrete jumps in the signal traces and hence the likelihood. This can cause the
                minimizer to get stuck in local minima and fail. The speedup options overwritten in the config file are:
                - delta_C_cut: set to 180 deg to not cut out the signal for any viewing angle
                - min_efield_amplitude: set to 0 to run the simulation for any efield amplitude
                - distance_cut: set to False to run simulation for any distance

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
        self.config_file = config_file
        self.overwrite_speedup_options = overwrite_speedup_options

        self.signal_search_width = signal_search_width
        self.delta_t = 1 / self.sampling_rate
        self.delta_t_array_matched_filter = np.linspace(-signal_search_width, signal_search_width, n_grid_matched_filter)
        self.i_shift_cc = (np.arange(-int(signal_search_width / self.delta_t), int(signal_search_width / self.delta_t) + 1)).astype(int)
        self.frequencies = np.fft.rfftfreq(self.n_samples, 1. / self.sampling_rate)

        # Initialize likelihood calculator:
        self.likelihood_calculator = likelihood_calculator.LikelihoodCalculator(
            n_antennas = self.n_channels,
            n_samples = self.n_samples,
            sampling_rate = self.sampling_rate,
            matrix_inversion_method = "pseudo_inv",
            threshold_amplitude = 0.1
        )
        self.likelihood_calculator.initialize_with_spectra(noise_spectra, self.Vrms)
        self.noise_psd = self.likelihood_calculator.noise_psd

        # Initialize matched filter:
        self.matched_filter = matched_filter.MatchedFilter(
            n_samples = self.n_samples,
            sampling_rate = self.sampling_rate,
            n_antennas = self.n_channels,
            noise_power_spectral_density = self.noise_psd,
            spectra_threshold_fraction = 0.1
        )

        self.detector_simulation_filter_amp = detector_simulation_filter_amp

    @register_run()
    def run(
        self,
        evt,
        station,
        det,
        parameters_initial,
        charge_excess_profile_id = 0,
        use_channels = None,
        reference_channel = 0,
        two_step_optimization = True,
        full_output = True,
        bounds = None
        ):
        """
        Run the likelihood reconstruction to fit a hadronic shower signal model to the traces in the station object.

        The reconstructed values are saved to the station and a shower object which is added to the station.
        If ``full_output`` is ``True``, the reconstructed parameters, likelihood values, and fit p-values are returned.

        Parameters
        ----------
        evt : NuRadioReco.framework.event.Event
            The event to run the module on.
        station : NuRadioReco.framework.station.Station
            The station object containing the channels with the data traces.
        det : NuRadioReco.framework.detector.Detector
            The detector description.
        parameters_initial : np.ndarray
            Initial guess of the parameters for the reconstruction. The array should contain seven values in the
            order ``[energy, zenith, azimuth, vertex_r_rel, vertex_theta_rel, vertex_phi_rel, pulse_time]``.
            The first three describe the shower energy and direction, the next three describe the shower vertex
            relative to the reference antenna, and the last value is the approximate pulse time relative to the
            trace start.
        charge_excess_profile_id : int, optional
            ID of the charge excess profile to use in the shower simulation. Not used for Alvarez2009.
        use_channels : list, optional
            List of channel IDs to be used for the reconstruction. If ``None``, all channels are used.
        reference_channel : int, optional
            Index of the reference channel for the reconstruction.
        two_step_optimization : bool, optional
            If ``True``, the reconstruction performs two minimizations. The first uses a matched filter approach to
            profile over amplitude and time for every step of the minimization, and the fit is then fine-tuned in a
            second ``-2LLH`` fit. If ``False``, the first step is skipped.
        full_output : bool, optional
            If ``True``, return the reconstructed signal, the signal parameters and the minus two log-likelihood of
            the reconstructed signal.
        bounds : np.ndarray, optional
            Bounds for the parameters to reconstruct. If set to None, the default values are used:
            - energy: 1 PeV to 100 EeV
            - zenith: 0 to 180 deg
            - azimuth: -360 to 360 deg
            - vertex_r_rel: 20 m to 5 km
            - vertex_theta_rel: 90 to 180 deg
            - vertex_phi_rel: -360 to 360 deg
            - pulse_time: initial guess +/- self.signal_search_width

        Returns
        -------
        parameters_fit: np.ndarray
            The best fit parameters of the signal model.
            Only returned if `full_output` enabled

        uncertainties_fit: np.ndarray
            Estimated marginalized uncertainties on the 7 reconstructed parameters using the Fisher.
            information matrix
            Only returned if `full_output` enabled

        signal_fit: np.ndarray
            The reconstructed signal in the readout traces.
            Only returned if `full_output` enabled

        minus_two_llh_initial: float
            The minus two log-likelihood value of the initial guess parameters.
            Only returned if `full_output` enabled

        minus_two_llh_fit: float
            The minus two log-likelihood value of the reconstructed signal.
            Only returned if `full_output` enabled

        p_value_fit: float
            The goodness-of-fit p-value of the reconstructed signal. This is calculated from minus_two_llh_fit
            and the degrees of freedom in the likelihood_calculator.
            Only returned if `full_output` enabled
        """

        if use_channels is None:
            use_channels = station.get_channel_ids()

        reference_index = np.argmax(use_channels == reference_channel)

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
            station_id = station.get_id(),
            config_file = self.config_file,
            detector_simulation_filter_amp = self.detector_simulation_filter_amp,
            det = det,
            reference_channel = reference_channel,
            evt_time = station.get_station_time(),
            use_channels = use_channels,
            pre_pulse_time = 100 * units.ns # not used
        )

        if self.overwrite_speedup_options:
            signal_model.config['speedup']['delta_C_cut'] = 180 * units.deg
            signal_model.config['speedup']['min_efield_amplitude'] = 0
            signal_model.config['speedup']['distance_cut'] = False

        def signal_model_wrapper(parameters):
            signal_station, signal_traces = signal_model.simulate_single_shower_forward_folding(
                energy = parameters[0],
                zenith = parameters[1],
                azimuth = parameters[2],
                vertex_r_rel = parameters[3],
                vertex_theta_rel = parameters[4],
                vertex_phi_rel = parameters[5],
                pulse_time = parameters[6],
                type = "HAD",
                trace_start_times = trace_start_times,
                charge_excess_profile_id = charge_excess_profile_id,
            )
            return signal_traces

        self.matched_filter.set_data(traces)

        if self.debug:
            # plot initial signal for debugging:
            signal_initial = signal_model_wrapper(parameters_initial)
            t_array = trace_start_times[0] + np.arange(0, self.n_samples) * self.delta_t
            fig_1, ax_1 = plt.subplots(self.n_channels, 1, figsize=(10, self.n_channels*2.5))
            for i_ch in range(self.n_channels):
                ax_1[i_ch].plot(t_array, traces[i_ch], label="Data")
                ax_1[i_ch].plot(t_array, signal_initial[i_ch], ls="--", label="Initial guess signal")
                ax_1[i_ch].set_ylabel("Voltage [V]")
                ax_1[i_ch].set_xlim(t_array[0], t_array[-1])
            ax_1[0].legend()
            ax_1[-1].set_xlabel("Time [s]")
            plt.tight_layout()
            plt.savefig("debug_neutrinoLikelihoodReconstructor.png")

            # Plot the spectra of the data and initial signal and likelihood calcualtor:
            fig_2, ax_2 = plt.subplots(self.n_channels, 1, figsize=(10, self.n_channels*2.5))
            for i_ch in range(self.n_channels):
                data_spectrum = np.abs(fft.time2freq(traces[i_ch], self.sampling_rate))
                signal_spectrum = np.abs(fft.time2freq(signal_initial[i_ch], self.sampling_rate))
                ax_2[i_ch].plot(self.frequencies, data_spectrum, label="Data")
                ax_2[i_ch].plot(self.frequencies, signal_spectrum, ls="--", label="Initial guess signal")
                ax_2[i_ch].plot(self.frequencies, self.likelihood_calculator.spectra[i_ch], color="k", ls=":", label="Assumed spectrum for likelihood")
                ax_2[i_ch].set_ylabel("Amplitude [V]")
                ax_2[i_ch].set_xlim(self.frequencies[0], self.frequencies[-1])
            ax_2[0].legend()
            ax_2[-1].set_xlabel("Frequency [Hz]")
            plt.tight_layout()
            plt.savefig("debug_neutrinoLikelihoodReconstructor_spectra.png")

        minus_two_llh_initial, minus_two_llh_fit, parameters_fit, uncertainties_fit = self._reconstruct_signal(traces, signal_model_wrapper, parameters_initial, two_step_optimization=two_step_optimization, bounds=bounds)

        signal_fit = signal_model_wrapper(parameters_fit)

        if self.debug:
            for i_ch in range(self.n_channels):
                ax_1[i_ch].plot(t_array, signal_fit[i_ch], ls=":",label="Fitted signal")
                if i_ch == 0:
                    ax_1[i_ch].legend()
            fig_1.savefig("debug_neutrinoLikelihoodReconstructor.png")
            fig_1.show()

            for i_ch in range(self.n_channels):
                signal_spectrum = np.abs(fft.time2freq(signal_fit[i_ch], self.sampling_rate))
                ax_2[i_ch].plot(self.frequencies, signal_spectrum, ls=":", label="Fitted signal")
                if i_ch == 0:
                    ax_2[i_ch].legend()
            fig_2.savefig("debug_neutrinoLikelihoodReconstructor_spectra.png")
            fig_2.show()
            plt.close()

        # Convert fitted parameters to shower parameters:
        rec_energy = parameters_fit[0]
        rec_zenith = parameters_fit[1]
        rec_azimuth = parameters_fit[2]
        rec_vertex_r_rel = parameters_fit[3]
        rec_vertex_theta_rel = parameters_fit[4]
        rec_vertex_phi_rel = parameters_fit[5]
        rec_vertex_xyz_rel = hp.spherical_to_cartesian(rec_vertex_theta_rel, rec_vertex_phi_rel) * rec_vertex_r_rel
        rec_vertex_xyz = rec_vertex_xyz_rel + det.get_relative_position(station.get_id(), 0)
        signal_model.propagator.set_start_and_end_point(rec_vertex_xyz, det.get_relative_position(station.get_id(), reference_channel))
        signal_model.propagator.find_solutions()
        travel_time = signal_model.propagator.get_travel_time(0)
        rec_vertex_time = parameters_fit[6] + trace_start_times[reference_index] - det.get_cable_delay(station.get_id(), reference_index) - travel_time

        # Convert fit uncertainties to shower parameter uncertainties:
        unc_xyz = hp.spherical_to_cartesian(rec_vertex_theta_rel, rec_vertex_phi_rel) * uncertainties_fit[3] # this ignores the uncertainties on rec_vertex_theta_rel and rec_vertex_phi_rel, but rec_vertex_r_rel has the dominant uncertainty
        time_unc_from_r_unc = uncertainties_fit[3] / (scp.constants.c / units.s / signal_model.ice.get_index_of_refraction(rec_vertex_xyz)) # 1-st order estimate
        unc_vertex_time = np.sqrt(uncertainties_fit[6]**2 + time_unc_from_r_unc**2)

        # Create reconstructed shower object and save it to the event:
        shower = NuRadioReco.framework.radio_shower.RadioShower(shower_id=0, station_ids=[station.get_id()])
        shower.set_parameter(shp.energy, rec_energy)
        shower.set_parameter_error(shp.energy, uncertainties_fit[0])
        shower.set_parameter(shp.zenith, rec_zenith)
        shower.set_parameter_error(shp.zenith, uncertainties_fit[1])
        shower.set_parameter(shp.azimuth, rec_azimuth)
        shower.set_parameter_error(shp.azimuth, uncertainties_fit[2])
        shower.set_parameter(shp.vertex, rec_vertex_xyz)
        shower.set_parameter_error(shp.vertex, unc_xyz)
        shower.set_parameter(shp.vertex_time, rec_vertex_time)
        shower.set_parameter_error(shp.vertex_time, unc_vertex_time)
        shower.set_parameter(shp.type, "HAD")
        shower.set_parameter(shp.charge_excess_profile_id, charge_excess_profile_id)
        evt.add_shower(shower)

        # Save results to station object:
        station.set_parameter(stnp.nu_zenith, rec_zenith)
        station.set_parameter(stnp.nu_azimuth, rec_azimuth)
        station.set_parameter(stnp.nu_energy, rec_energy)
        station.set_parameter(stnp.ccnc, "nc")

        # Calculate goodness of fit:
        n_dof_fit = self.likelihood_calculator.get_dof() - 7
        p_value_fit = 1 - scp.stats.chi2.cdf(minus_two_llh_fit, n_dof_fit)

        if full_output:
            return parameters_fit, uncertainties_fit, signal_fit, minus_two_llh_initial, minus_two_llh_fit, p_value_fit

    def _function_to_minimize_mf(self, data, signal):
        """
        Calculate the objective function for the first minimization.
        """

        if not self.use_chi2:
            self.matched_filter.set_template(signal)
            t_best, x_best = self.matched_filter.matched_filter_search(time_shift_array=self.delta_t_array_matched_filter)
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
        Calculate the cross-correlation between the data and the signal. The shift_array
        is the shift indicies to calculate the cross-correlation for.
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


    def _reconstruct_signal(self, data, signal_function, parameters_initial, two_step_optimization=True, bounds=None):
        """
        Reconstruct the signal from the given data and the provided signal model using the initial guess of the signal parameters.
        """

        normalization = np.array([units.PeV, units.deg, units.deg, units.m, units.deg, units.deg, units.ns])

        if bounds is None:
            bounds = np.array([(1 * units.PeV, 100 * units.EeV),
                                (0 * units.deg, 180 * units.deg),
                                (-360 * units.deg, 360 * units.deg),
                                (20 * units.m, 5 * units.km),
                                (0 * units.deg, 180 * units.deg),
                                (-360 * units.deg, 360 * units.deg),
                                (parameters_initial[6] - self.signal_search_width * units.ns, parameters_initial[6] + self.signal_search_width)])

        minus_two_llh_initial = self._function_to_minimize_llh(data, signal_function(parameters_initial))

        # Matched filter optimization to get close to the global minimum and good initial parameters for the likelihood fit:
        if two_step_optimization:
            minimizer_mf = minimization.Minimizer(
                signal_function = signal_function,
                objective_function = self._function_to_minimize_mf,
                parameters_initial = parameters_initial,
                parameters_bounds = bounds,
                normalization = normalization,
                fixed = [True, False, False, False, False, False, True],
                debug = self.debug
            )
            m_mf = minimizer_mf.run_minimization(data=data, method="minuit")

            params_fit_mf = minimizer_mf.parameters
            signal_fit_mf = signal_function(params_fit_mf)

            # Get best matching time shift:
            self.matched_filter.set_template(signal_fit_mf)
            t_shift, x = self.matched_filter.matched_filter_search(time_shift_array=self.delta_t_array_matched_filter)
            t_guess = parameters_initial[6] + t_shift

            # Fit energy:
            parameters_initial_amplitude = np.array([params_fit_mf[0], params_fit_mf[1], params_fit_mf[2], params_fit_mf[3], params_fit_mf[4], params_fit_mf[5], t_guess])
            minimizer_amplitude = minimization.Minimizer(
                signal_function = signal_function,
                objective_function = self._function_to_minimize_llh,
                parameters_initial = parameters_initial_amplitude,
                parameters_bounds = bounds,
                normalization = normalization,
                fixed = [False, True, True, True, True, True, True],
                debug=self.debug
            )
            m_amplitude = minimizer_amplitude.run_minimization(data=data, method="minuit")
            energy_guess = minimizer_amplitude.parameters[0]

            parameters_initial_llh = np.array([energy_guess, params_fit_mf[1], params_fit_mf[2], params_fit_mf[3], params_fit_mf[4], params_fit_mf[5], t_guess])

        elif not two_step_optimization:
            parameters_initial_llh = np.copy(parameters_initial)


        minimizer_llh = minimization.Minimizer(
            signal_function = signal_function,
            objective_function = self._function_to_minimize_llh,
            parameters_initial = parameters_initial_llh,
            parameters_bounds = bounds,
            normalization = normalization,
            fixed = [False, False, False, False, False, False, False],
            debug = self.debug
        )
        m = minimizer_llh.run_minimization(data=data, method="minuit")

        params_fit = minimizer_llh.parameters
        minus_two_llh_fit = minimizer_llh.result

        # Estimate 1st-order uncertainties using the Fisher information matrix:
        dx = np.array([1e-6, 1e-6, 1e-6, 1e-4, 1e-6, 1e-6, 1e-4])
        def signal_function_scaled(params_scaled):
            params = params_scaled * normalization
            return signal_function(params)
        fisher_information_matrix_fit = self.likelihood_calculator.calculate_fisher_information_matrix(signal_function_scaled, params_fit / normalization, dx)
        f_i_fit = np.linalg.pinv(fisher_information_matrix_fit)
        uncertainties_fit = np.sqrt(np.diag(f_i_fit)) * normalization

        return minus_two_llh_initial, minus_two_llh_fit, params_fit, uncertainties_fit
