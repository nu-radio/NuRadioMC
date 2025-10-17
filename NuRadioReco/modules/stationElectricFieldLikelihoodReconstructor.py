from NuRadioReco.modules.base.module import register_run

import numpy as np
import matplotlib.pyplot as plt
import copy

from NuRadioReco.utilities.analytic_pulse import get_analytic_pulse_freq
from NuRadioReco.utilities import units, noise_model, fft, trace_minimizer, matched_filter, trace_utilities
from NuRadioReco.framework.electric_field import ElectricField
from NuRadioReco.framework.sim_station import SimStation
from NuRadioReco.framework.event import Event
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import stationParameters as stnp
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.electricFieldBandPassFilter
import NuRadioReco.modules.channelLengthAdjuster
from radiotools import helper as hp
from radiotools import coordinatesystems

efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin(debug=False, pre_pulse_time=0, post_pulse_time=0, caching=False)
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelBandPassFilter.begin()
electricFieldBandPassFilter = NuRadioReco.modules.electricFieldBandPassFilter.electricFieldBandPassFilter()
channelLengthAdjuster = NuRadioReco.modules.channelLengthAdjuster.channelLengthAdjuster()

import logging
logger = logging.getLogger('NuRadioReco.StationElectricLikelhihoodFieldReconstructor')


class StationElectricLikelhihoodFieldReconstructor:
    """
    Class for reconstructing electric fields in a station, e.g., a dual polarized antenna or the
    upwardfacing LPDAs in an RNO-G shallow station. This class forward fold an analytical electric
    field, assumed to be the same in all channels, and compares it to a measured set of data traces
    in a likelihood objective function. The -2DeltaLLH is minimized in two stages, first using a
    matched filter to fit the shape of the signal and second a -2DeltaLLH minimization to fine-tune
    the reconstructed parameters. The likelihood is calculated using the spectrum of the noise, which
    enables correct error estimates of reconstructed parameters.

    This method is similar to voltageToAnalyticEfieldConverter, but uses a likelihood based on the
    noise spectrum instead of a chi-square and has an improved minimization strategy.

    This class assumes that the hardware response is subtracted from the data, e.g.,
    hardwareResponseIncorporator.run(event, station, det, sim_to_data=False, mingainlin=0.001) has been run.
    """

    def __init__(self):
        pass

    def begin(self, n_channels, n_samples, sampling_rate, noise_spectra, Vrms, filter_settings_low, filter_settings_high, use_chi2=False, zenith_azimuth_free=False, debug=False):
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

            filter_settings_low: dict, optional
                Low-pass filter settings to be applied to the electric field signal. The same filter must have been applied to
                the data and noise before this module is run.

            filter_settings_high: dict, optional
                High-pass filter settings to be applied to the electric field signal. The same filter must have been applied to
                the data and noise before this module is run.

            use_chi2: bool, optional
                Whether to use chi2 minimization instead of likelihood. Mostly used for debugging and method comparison.

            zenith_azimuth_free: bool, optional
                Whether to reconstruct the zenith and azimuth arrival direction of the electric field in the minimization, or keep them fixed.
                The initial (or fixed) value used is the reconstructed values present in the station object or the MC values in the sim_station
                object.

            debug: bool, optional
                Extra plots and printouts for debugging
        """

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.sampling_rate = sampling_rate
        self.Vrms = Vrms
        self.filter_settings_low = filter_settings_low
        self.filter_settings_high = filter_settings_high
        self.use_chi2 = use_chi2
        self.zenith_azimuth_free = zenith_azimuth_free
        self.debug = debug

        self.delta_t = 1/self.sampling_rate
        self.t_array_matched_filter = np.arange(0, self.n_samples) * self.delta_t - self.n_samples * self.delta_t/ 2
        self.i_shift_cc = np.arange(0, self.n_samples)
        self.frequencies = np.fft.rfftfreq(self.n_samples, 1. / self.sampling_rate)

        # initialize noise model:
        self.noise_model = noise_model.NoiseModel(
            n_antennas = self.n_channels,
            n_samples = self.n_samples,
            sampling_rate = self.sampling_rate,
            matrix_inversion_method = "pseudo_inv",
            threshold_amplitude = 0.1
        )
        self.noise_model.initialize_with_spectra(noise_spectra, self.Vrms)
        self.noise_psd = self.noise_model.noise_psd

        # initialize matched filter:
        self.matched_filter = matched_filter.MatchedFilter(
            n_samples = self.n_samples,
            sampling_rate = self.sampling_rate,
            n_antennas = self.n_channels,
            noise_power_spectral_density = self.noise_psd,
            spectra_threshold_fraction = 0.1
        )

    @register_run()
    def run(self, evt, station, det, use_channels=None, signal_search_window=None, use_MC_direction=False):
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

            use_channels: list, optional
                List of channel IDs to be used for the reconstruction. If None, all channels are used.

            signal_search_window: tuple, optional
                Time window (start, end) to search for the signal in the traces.

            use_MC_direction: bool, optional
                Whether to use the Monte Carlo true arrival direction for the reconstruction if it is
                present in the sim_station object.
        """

        if use_channels is None:
            use_channels = station.get_channel_ids()

        if use_MC_direction and (station.get_sim_station() is not None):
            zenith = station.get_sim_station()[stnp.zenith]
            azimuth = station.get_sim_station()[stnp.azimuth]
        else:
            logger.warning("Using reconstructed angles as no simulation present")
            zenith = station[stnp.zenith]
            azimuth = station[stnp.azimuth]

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

        def signal_function(parameters):
            return self._get_signal(parameters, det, station.get_id(), use_channels, trace_start_times)

        self.matched_filter.set_data(traces)

        minus_two_llh_best = None
        f_theta_f_phi_initial = [[0.5, 1], [0.5, -1], [-0.5, 1], [-0.5, -1]]
        for i_fit in range(4):

            parameters_initial = np.array([f_theta_f_phi_initial[i_fit][0], f_theta_f_phi_initial[i_fit][1], -1, np.pi/2, 300, -10, zenith, azimuth])

            minus_two_llh, polarization_reco, polarization_uncertainty, fluence_reco, fluence_uncertainty, fitted_params, fitted_params_uncertainties = self._reconstruct_signal(
                traces, signal_function, parameters_initial, trace_start_times, second_order=True, signal_search_window=signal_search_window)

            if minus_two_llh_best is None or minus_two_llh < minus_two_llh_best:
                minus_two_llh_best = np.copy(minus_two_llh)
                polarization_reco_best = np.copy(polarization_reco)
                fluence_reco_best = np.copy(fluence_reco)
                polarization_uncertainty_best = np.copy(polarization_uncertainty)
                fluence_uncertainty_best = np.copy(fluence_uncertainty)
                fitted_params_best = np.copy(fitted_params)
                fitted_params_uncertainties_best = np.copy(fitted_params_uncertainties)

        initial_signal = signal_function(parameters_initial)
        fitted_signal = signal_function(fitted_params_best)

        # save results to station object:
        electric_field = self._get_efield(fitted_params_best[:6], fitted_params_best[6], fitted_params_best[7], use_channels, apply_filter=True)
        electric_field.set_parameter(efp.signal_energy_fluence, fluence_reco_best)
        electric_field.set_parameter_error(efp.signal_energy_fluence, fluence_uncertainty_best)
        electric_field.set_parameter(efp.polarization_angle, polarization_reco_best)
        electric_field.set_parameter_error(efp.polarization_angle, polarization_uncertainty_best)
        electric_field.set_parameter(efp.cr_spectrum_slope, fitted_params_best[2])
        electric_field.set_parameter(efp.signal_time, trace_start_times[0] + fitted_params_best[3])
        electric_field.set_parameter(efp.cr_spectrum_quadratic_term, fitted_params_best[5])
        electric_field.set_parameter(efp.zenith, fitted_params_best[6])
        electric_field.set_parameter(efp.azimuth, fitted_params_best[7])

        # compute expected polarization
        site = det.get_site(station.get_id())
        exp_efield = hp.get_lorentzforce_vector(fitted_params_best[6], fitted_params_best[7], hp.get_magnetic_field_vector(site))
        cs = coordinatesystems.cstrafo(fitted_params_best[6], fitted_params_best[7], site=site)
        exp_efield_onsky = cs.transform_from_ground_to_onsky(exp_efield)
        exp_pol_angle = np.arctan2(exp_efield_onsky[2], exp_efield_onsky[1])
        electric_field.set_parameter(efp.polarization_angle_expectation, exp_pol_angle)

        electric_field.set_trace_start_time(trace_start_times[0])

        station.add_electric_field(electric_field)

        # Delete this code below for NuRadioReco PR:
        results_method = dict(
            method_name="LLH",
            polarization_reco=polarization_reco_best / units.deg,
            fluence_reco=fluence_reco_best,
            f_theta_reco=np.abs(fitted_params_best[0]),
            f_phi_reco=np.abs(fitted_params_best[1]),
            llh=minus_two_llh_best,
            error_fit=fitted_params_uncertainties_best,
            error_polarization=polarization_uncertainty_best / units.deg,
            error_fluence=fluence_uncertainty_best,
            params=fitted_params_best,
            signal_search_window=signal_search_window,
            n_dof = np.sum(np.linalg.matrix_rank(self.noise_model.cov_inv)),
            zenith_reco=fitted_params_best[6] if self.zenith_azimuth_free else None,
            azimuth_reco=fitted_params_best[7] if self.zenith_azimuth_free else None,
            A_theta=np.sign(fitted_params_best[0]) * np.abs(fitted_params_best[0])**0.5,
            A_phi=np.sign(fitted_params_best[1]) * np.abs(fitted_params_best[1])**0.5,
        )

        return results_method, fitted_signal, initial_signal

    def _function_to_minimize_1(self, data, signal):
        """
        Calculate the objective function for the first minimization.
        """

        if not self.use_chi2:
            # Matched filter:
            self.matched_filter.set_template(signal)
            t_best, x_best = self.matched_filter.matched_filter_search(time_shift_array=self.t_array_matched_filter)
            llh_mf = self.matched_filter.calculate_matched_filter_delta_log_likelihood()

            return -2 * llh_mf

        elif self.use_chi2:
            i_max, cross = self._cross_correlation(data, signal, shift_array=self.i_shift_cc)
            return -cross

    def _function_to_minimize_2(self, data, signal):
        """
        Calculate the log-likelihood objective function of the 2nd minimization
        """
        if not self.use_chi2:
            minus_two_llh =  self.noise_model.calculate_minus_two_delta_llh(data, signal)
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

        Returns
        -------
        float
            Cross-correlation between the data and the signal
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

    def _get_efield(self, parameters, zenith_arrival, azimuth_arrival, use_channels, apply_filter=False):
        """
        Get the electric field in the two antennas for the given parameters.

        Parameters
        ----------
        parameters: np.ndarray
            Parameters of the signal model
            0: fluence theta
            1: fluence phi
            2: slope of the pulse
            3: phase of the pulse
            4: time of the pulse (maximum of absolute hilbert envelope) at coordinates (0,0,0) relative to start of the 0th trace in ns
            5: quadratic term

        zenith_arrival: float
            Zenith angle of the incoming efield

        azimuth_arrival: float
            Azimuth angle of the incoming efield

        use_channels: list
            List of channels that the electric field applies to

        apply_filter: bool
            Whether to apply the filter to the electric field

        Returns
        -------
        np.ndarray
            Electric field for given parameters
        """

        n_samples_time = self.n_samples
        sampling_rate = self.sampling_rate

        amp_p0 = 1
        amp_p1 = parameters[2]
        phase_p0 = parameters[3]
        phase_p1 = -parameters[4] * 2*np.pi
        quadratic_term = parameters[5]
        quadratic_term_offset = self.filter_settings_high["passband"][0]

        # Calculate the electric field:
        efield_norm = get_analytic_pulse_freq(amp_p0, amp_p1, phase_p0, n_samples_time, sampling_rate, phase_p1=phase_p1, bandpass=None, quadratic_term=quadratic_term, quadratic_term_offset=quadratic_term_offset)

        # Set the electric field:
        electric_field = ElectricField(use_channels, position=None, shower_id=None, ray_tracing_id=None)
        electric_field.set_frequency_spectrum(np.array([np.zeros_like(efield_norm), efield_norm, efield_norm]), self.sampling_rate)
        electric_field.set_trace_start_time(0)
        electric_field[efp.zenith] = zenith_arrival
        electric_field[efp.azimuth] = azimuth_arrival
        electric_field[efp.ray_path_type] = "direct"

        # Make a copy of the electric field and apply filter:
        efield_filtered = copy.copy(electric_field)
        sim_station = SimStation(0)
        sim_station.add_electric_field(efield_filtered)
        evt = Event(1, 1)
        electricFieldBandPassFilter.run(evt, sim_station, det=None, **self.filter_settings_low)
        electricFieldBandPassFilter.run(evt, sim_station, det=None, **self.filter_settings_high)

        # Normalize the electric field to specified fluence (filtered):
        f_R, f_theta, f_phi = trace_utilities.get_electric_field_energy_fluence(efield_filtered.get_trace(), efield_filtered.get_times())
        if apply_filter:
            trace = efield_filtered.get_frequency_spectrum()
        else:
            trace = electric_field.get_frequency_spectrum()
        trace[1] *= np.sign(parameters[0]) * np.sqrt(np.abs(parameters[0]) / f_theta)
        trace[2] *= np.sign(parameters[1]) * np.sqrt(np.abs(parameters[1]) / f_phi)

        electric_field.set_frequency_spectrum(trace, self.sampling_rate)

        return electric_field

    def _get_signal(self, parameters, det, station_id, use_channels, trace_start_times, filter_before_det_resp=True):
        """
        Get the signal in the two antennas for the given parameters.

        Parameters
        ----------
        parameters: np.ndarray
            Parameters of the signal model
            0: fluence theta
            1: fluence phi
            2: slope of the pulse
            3: phase of the pulse
            4: time of the pulse (maximum of absolute hilbert envelope) at coordinates (0,0,0) relative to start of the 0th trace in ns
            5: quadratic term
            6: zenith_arrival
            7: azimuth_arrival

        det: NuRadioReco.framework.detector.Detector
            The detector description to use for the simulation

        station_id: int
            The ID of the station to use for the simulation

        use_channels: list
            List of channels to calculate the signal for

        trace_start_times: list
            List of start times for data traces. The 4th parameter is relative to the first time in this list.

        filter_before_det_resp: bool
            Whether to apply the filter to the efield (before detector response) or to the channel traces (after detector response)

        Returns
        -------
        np.ndarray
            Signal for given parameters
        """

        zenith_arrival = parameters[6]
        azimuth_arrival = parameters[7]

        electric_field = self._get_efield(parameters, zenith_arrival, azimuth_arrival, use_channels, apply_filter=filter_before_det_resp)
        electric_field.set_trace_start_time(trace_start_times[0])

        sim_station = SimStation(station_id)
        sim_station.add_electric_field(electric_field)
        sim_station.set_is_cosmic_ray()
        sim_station[stnp.zenith] = zenith_arrival
        sim_station[stnp.azimuth] = azimuth_arrival

        evt = Event(1, 1)
        station = NuRadioReco.framework.station.Station(station_id)
        station.add_sim_station(sim_station)
        station[stnp.zenith] = zenith_arrival
        station[stnp.azimuth] = azimuth_arrival
        efieldToVoltageConverter.run(evt, station, det, channel_ids=use_channels)

        for i_ch, channel_id in enumerate(use_channels):
            channel = station.get_channel(channel_id)

            # Make new channel which are the signal in the readout windows of the data trace:
            signal_channel = NuRadioReco.framework.channel.Channel(channel_id)
            signal_channel.set_trace(np.zeros(self.n_samples), self.sampling_rate)
            signal_channel.set_trace_start_time(trace_start_times[i_ch])
            signal_channel.add_to_trace(channel, raise_error=False)
            station.add_channel(signal_channel, overwrite=True)

        # Apply bandpass filter. It is safest to apply rectangular filters again, in case of FFT or trace cutting artefacts:
        if not filter_before_det_resp or (self.filter_settings_high["filter_type"]=="rectangular" and self.filter_settings_low["filter_type"]=="rectangular"):
            channelBandPassFilter.run(evt, station, det, **self.filter_settings_low)
            channelBandPassFilter.run(evt, station, det, **self.filter_settings_high)

        traces = []
        for i_ch in range(self.n_channels):
            traces.append(station.get_channel(use_channels[i_ch]).get_trace())

        return np.array(traces)

    def _reconstruct_signal(self, data, signal_function, parameters_initial, trace_start_times, second_order=True, signal_search_window=None):
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

        trace_start_times: np.ndarray
            Start times of the channels to be used in the reconstruction. Used to bound the time parameter of the signal in the reconstruction.

        second_order: bool, default: True
            If True, the second order correction to the analytical efield spectrum is fitted

        signal_search_window: tuple, optional
            The time window to search for the signal in the data. The window is the global
            time of the electric field pulse at the (0,0,0) coordinate. If None, the entire
            time range is used.

        Returns
        -------
        minus_two_llh_fit_2: float
            The negative log-likelihood value for the reconstructed signal

        polarization: float
            The polarization angle of the reconstructed signal

        fluence: float
            The fluence of the reconstructed signal

        fitted_params_2: np.ndarray
            The fitted parameters for the reconstructed signal

        uncertainties_fit: np.ndarray
            The uncertainties on the fitted parameters

        error_polarization: float
            The error on the polarization angle

        error_fluence: float
            The error on the fluence
        """

        dx_array = np.array([1e-3, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
        fisher_information_matrix = self.noise_model.calculate_fisher_information_matrix(signal_function, parameters_initial, dx_array, ignore_parameters = [6,7] if not self.zenith_azimuth_free else [])
        f_i = np.linalg.pinv(fisher_information_matrix)
        uncertainties_1 = np.sqrt(np.diag(f_i))
        scaling = np.append(uncertainties_1, [1, 1]) if not self.zenith_azimuth_free else uncertainties_1


        bounds = np.array([
            (-10000, 10000),
            (-10000, 10000),
            (-100, -0.0001),
            (-3*np.pi, 3*np.pi),
            (np.min(trace_start_times) - trace_start_times[0], np.max(trace_start_times) + (self.n_samples-1)*self.delta_t - trace_start_times[0]),
            (-500, 0),
            (0, np.pi/2),
            (-2*np.pi, 2*np.pi)
            ])

        if signal_search_window is not None:
            search_window_length = signal_search_window[1] - signal_search_window[0]
            parameters_initial[4] = signal_search_window[0] + search_window_length / 2 - trace_start_times[0]
            self.t_array_matched_filter = np.arange(-search_window_length/2, search_window_length/2, self.delta_t/2)
            self.i_shift_cc = (self.t_array_matched_filter / self.sampling_rate).astype(int)

        reconstructor_1 = trace_minimizer.TraceMinimizer(
            signal_function = signal_function,
            objective_function = self._function_to_minimize_1,
            parameters_initial = parameters_initial,
            parameters_bounds = bounds,
        )
        if self.zenith_azimuth_free:
            reconstructor_1.fix_parameters([True, False, False, False, True, not(second_order), False, False])
        else:
            reconstructor_1.fix_parameters([True, False, False, False, True, not(second_order), True, True])
        reconstructor_1.set_scaling(scaling)

        m = reconstructor_1.run_minimization(data=data, method="minuit")

        fitted_params_1 = reconstructor_1.parameters

        signal_fit = signal_function(fitted_params_1)

        # get time:
        if not self.use_chi2:
            self.matched_filter.set_template(signal_fit)
            t_offset, x = self.matched_filter.matched_filter_search(time_shift_array=self.t_array_matched_filter)
        else:
            i_max, cross = self.cross_correlation(data, signal_fit, shift_array=self.i_shift_cc)
            t_offset = i_max / self.sampling_rate
        parameters_adjusted = np.array([fitted_params_1[0], fitted_params_1[1], fitted_params_1[2], fitted_params_1[3], (fitted_params_1[4]+t_offset), parameters_initial[5], fitted_params_1[6], fitted_params_1[7]])
        signal_fit_adjusted = signal_function(parameters_adjusted)
        signal_fit_adjusted = signal_fit_adjusted / np.max(signal_fit_adjusted) * np.max(data)


        # Get best matching time and amplitude of the fit:
        amplitude_correction = (np.max(data) / np.max(signal_fit))**2
        parameters_initial_2 = np.array([fitted_params_1[0]*amplitude_correction, fitted_params_1[1]*amplitude_correction, fitted_params_1[2], fitted_params_1[3], (parameters_initial[4]+t_offset), parameters_initial[5], parameters_initial[6], parameters_initial[7]])

        if signal_search_window is not None:
            bounds[4] = (signal_search_window[0], signal_search_window[1]) - trace_start_times[0]

            # A wider window often results in more stable minimization:
            bounds[4][0] = bounds[4][0] - (bounds[4][1] - bounds[4][0]) / 2
            bounds[4][1] = bounds[4][1] + (bounds[4][1] - bounds[4][0]) / 2

        reconstructor_2 = trace_minimizer.TraceMinimizer(
            signal_function = signal_function,
            objective_function = self._function_to_minimize_2,
            parameters_initial = parameters_initial_2,
            parameters_bounds = bounds,
        )

        if self.zenith_azimuth_free:
            reconstructor_2.fix_parameters([False, False, False, False, False, not(second_order), False, False])
        else:
            reconstructor_2.fix_parameters([False, False, False, False, False, not(second_order), True, True])

        fisher_information_matrix2 = self.noise_model.calculate_fisher_information_matrix(signal_function, fitted_params_1, dx_array, ignore_parameters = [6,7] if not self.zenith_azimuth_free else [])
        f_i_2 = np.linalg.pinv(fisher_information_matrix2)
        errors_2 = np.sqrt(np.diag(f_i_2))
        scaling_2 = np.append(errors_2, [1, 1]) if not self.zenith_azimuth_free else errors_2
        reconstructor_2.set_scaling(scaling_2)

        m = reconstructor_2.run_minimization(data=data, method="minuit")

        fitted_params_2 = reconstructor_2.parameters
        minus_two_llh_fit_2 = reconstructor_2.result

        f_theta = np.abs(fitted_params_2[0])
        f_phi = np.abs(fitted_params_2[1])
        fluence = f_theta + f_phi
        A_theta = np.sign(fitted_params_2[0]) * f_theta**0.5
        A_phi = np.sign(fitted_params_2[1]) * f_phi**0.5
        polarization = np.arctan2(A_phi, A_theta)

        fisher_information_matrix_fit = self.noise_model.calculate_fisher_information_matrix(signal_function, fitted_params_2, dx_array, ignore_parameters = [6,7] if not self.zenith_azimuth_free else [])
        f_i_fit = np.linalg.pinv(fisher_information_matrix_fit)
        uncertainties_fit = np.sqrt(np.diag(f_i_fit))
        f_theta_uncertainty = uncertainties_fit[0]
        f_phi_uncertainty = uncertainties_fit[1]
        uncertainty_fluence = np.sqrt(f_theta_uncertainty**2 + f_phi_uncertainty**2)
        uncertainty_polarization = np.sqrt( (np.sqrt(f_theta) / (2 * np.sqrt(f_phi) * (f_phi+f_theta)) )**2 * f_phi_uncertainty**2 + ( -np.sqrt(f_phi) / (2 * np.sqrt(f_theta) * (f_phi+f_theta)) )**2 * f_theta_uncertainty**2)

        if self.debug:
            # plot results for debugging:
            signal_initial = signal_function(parameters_initial)
            signal_initial_2 = signal_function(parameters_initial_2)
            signal_fit_2 = signal_function(fitted_params_2)

            fig, ax = plt.subplots(self.n_channels, 1, figsize=(10, self.n_channels*3))
            for i_ch in range(self.n_channels):
                t_array = trace_start_times[i_ch] + np.arange(0, self.n_samples) * self.delta_t
                ax[i_ch].plot(t_array, data[i_ch], label="data")
                ax[i_ch].plot(t_array, signal_initial[i_ch], ls="--", label="initial")
                ax[i_ch].plot(t_array, signal_fit[i_ch], label="fit")
                ax[i_ch].plot(t_array, signal_fit_adjusted[i_ch], "--", label="fit adjusted")
                ax[i_ch].plot(t_array, signal_initial_2[i_ch], "y:", label="initial 2")
                ax[i_ch].plot(t_array, signal_fit_2[i_ch], "k:", label="fit 2")

                # Plot signal search window:
                if signal_search_window is not None:
                    ax[i_ch].vlines([signal_search_window[0], signal_search_window[1]], np.min(data[i_ch]*2), np.max(data[i_ch]*2), color="g", ls=":", label="search window (efield)")

                # Plot bounds (matched filter):
                t_max = t_array[np.argmax(signal_fit[i_ch])]
                ax[i_ch].vlines([t_max+self.t_array_matched_filter[0], t_max+self.t_array_matched_filter[-1]], np.min(data[i_ch]*2), np.max(data[i_ch]*2), color="r", ls="--", label="matched filter")

                # Plot bounds (LLH reconstruction):
                s0 = signal_function(np.array([fitted_params_2[i_ch], fitted_params_2[1], fitted_params_2[2], fitted_params_2[3], bounds[4][0], fitted_params_2[5], fitted_params_2[6], fitted_params_2[7]]))
                t_max_bound_0 = t_array[np.argmax(s0[i_ch])]
                s1 = signal_function(np.array([fitted_params_2[i_ch], fitted_params_2[1], fitted_params_2[2], fitted_params_2[3], bounds[4][1], fitted_params_2[5], fitted_params_2[6], fitted_params_2[7]]))
                t_max_bound_1 = t_array[np.argmax(s1[i_ch])]
                ax[i_ch].vlines([t_max_bound_0, t_max_bound_1], np.min(data[i_ch]*2), np.max(data[i_ch]*2), color="b", ls="--", label="bounds")

                ax[i_ch].set_ylabel("Voltage [V]")

            ax[0].legend()
            if not self.use_chi2:
                ax[0].set_title(f"$-2\Delta$LLH: {minus_two_llh_fit_2} \n parameters: {fitted_params_2}")
            else:
                ax[0].set_title(f"$\chi^2$: {minus_two_llh_fit_2} \n parameters: {fitted_params_2}")
            ax[-1].set_xlabel("Time [s]")
            plt.tight_layout()
            plt.savefig("debug_StationElectricFieldReconstructor.png")
            plt.show()
            plt.close()

            # Plot spectra of (assumed) noise and data:
            fig, ax = plt.subplots(self.n_channels, 1, figsize=(10, self.n_channels*3))
            for i_ch in range(self.n_channels):
                ax[i_ch].plot(self.frequencies, self.noise_model.spectra[i_ch], "k-", label="noise model")
                ax[i_ch].plot(self.frequencies, np.abs(fft.time2freq(data[i_ch], sampling_rate=self.sampling_rate)), "b-", label="data")
                ax[i_ch].plot(self.frequencies, np.abs(fft.time2freq(signal_initial[i_ch], sampling_rate=self.sampling_rate)), "r-", label="initial")
                ax[i_ch].plot(self.frequencies, np.abs(fft.time2freq(signal_fit_2[i_ch], sampling_rate=self.sampling_rate)), "g-", label="fit")
                ax[i_ch].hlines( np.max(self.noise_model.spectra[i_ch])/100, 0, max(self.frequencies), "m", "--", label="threshold")
                ax[i_ch].set_ylabel("Amplitude [V/GHz]")
                #ax[i].set_yscale("log")
            ax[0].legend()
            ax[-1].set_xlabel("Frequency [GHz]")
            fig.tight_layout()
            plt.savefig("debug_StationElectricFieldReconstructor_spectra.png")
            plt.show()
            plt.close()

        return minus_two_llh_fit_2, polarization, uncertainty_polarization, fluence, uncertainty_fluence, fitted_params_2, uncertainties_fit
