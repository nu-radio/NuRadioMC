import numpy as np
from NuRadioReco.utilities import units, fft, trace_utilities, bandpass_filter
import NuRadioReco.utilities.trace_utilities
import NuRadioReco.detector.antennapattern
import NuRadioReco.detector.RNO_G.analog_components
import NuRadioReco.detector.ARIANNA.analog_components
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import channelParameters as chp
import NuRadioReco.modules.iftElectricFieldReconstructor.operators
import NuRadioReco.framework.base_trace
import NuRadioReco.framework.electric_field
import scipy
import nifty5 as ift
import matplotlib.pyplot as plt
import scipy.signal
import radiotools.helper


class IftElectricFieldReconstructor:
    """
    Module that uses Information Field Theory to reconstruct the electric field.
    A description how this method works can be found at https://arxiv.org/abs/2102.00258
    """
    def __init__(self):
        self.__antenna_pattern_provider = NuRadioReco.detector.antennapattern.AntennaPatternProvider()
        self.__passband = None
        self.__filter_type = None
        self.__debug = False
        self.__efield_scaling = None
        self.__amp_dct = None
        self.__phase_dct = None
        self.__used_channel_ids = None
        self.__trace_samples = None
        self.__fft_operator = None
        self.__n_shifts = None
        self.__trace_start_times = None
        self.__n_iterations = None
        self.__n_samples = None
        self.__polarization = None
        self.__electric_field_template = None
        self.__convergence_level = None
        self.__relative_tolerance = None
        self.__use_sim = False
        self.__pulse_time_prior = None
        self.__pulse_time_uncertainty = None
        self.__phase_slope = None
        self.__slope_passbands = None
        self.__energy_fluence_passbands = None
        return

    def begin(
        self,
        electric_field_template,
        passband=None,
        filter_type='butter',
        amp_dct=None,
        pulse_time_prior=20. * units.ns,
        pulse_time_uncertainty=5. * units.ns,
        n_iterations=5,
        n_samples=20,
        polarization='pol',
        relative_tolerance=1.e-7,
        convergence_level=3,
        energy_fluence_passbands=None,
        slope_passbands=None,
        phase_slope='both',
        debug=False
    ):
        """
        Define settings for the reconstruction.

        Parameters
        -------------
        electric_field_template: NuRadioReco.framework.base_trace.BaseTrace object
            BaseTrace (or child class) object containing an electric field template
            that is used to determine the position of the radio pulse in the channel
            waveforms.
        passband: list of floats or None
            Lower and upper bound of the filter that should be applied to the channel
            waveforms and the IFT model. If None is passed, no filter is applied
        filter_type: string
            Name of the filter type to be used. Has to be implemented in the NuRadioReco.utilities.
            bandpass_filter.get_filter_response function. Only used if passband is not None
        amp_dct: dictionary
            Dictionary containing the prior settings for the electric field spectrum
        pulse_time_prior: float
            Expected pulse time relative to the trace start time. Note that this is the time of the
            electric field pulse, not the voltage pulse
        pulse_time_uncertainty: float
            Uncertainty on the pulse time
        n_iterations: integer
            Number of times the IFT minimizer iterates. More iterations lead to better results, but
            increase run time.
        n_samples: integer
            Number of prior samples the IFT minimizer uses to find the maximum prior. Also the number of
            samples used to estimate uncertainties
        polarization: string
            Polarization of the reconstructed radio signal. If set to "theta" or "phi", only that
            component of the electric field is reconstructed. If set to "pol", both components
            are reconstructed.
        relative_tolerance: float
            Relative improvement for the minimizer in a cycle for the optimization to finish.
        convergence_level: integer
            Number of cycles the relative improvement of the minimizer has to be below relative_tolerance
            for the optimization to finish.
        energy_fluence_passbands: list of floats
            List of passbands for which the energy fluence is calculated
        slope_passbands: list of floats
            List of passbands to calculate the ratio of the energy fluences in different passbands.
        phase_slope: string
            Specifies the sign of the slope of the linear function describing the phase of the electric field.
            Options are "negative", "positive" and "both". If "both" is selected, positive and negative slopes
            are used and the best fit is selected.
        debug: bool
            If true, debug plots are drawn.

        """
        self.__passband = passband
        self.__filter_type = filter_type
        self.__debug = debug
        self.__n_iterations = n_iterations
        self.__n_samples = n_samples
        self.__trace_samples = len(electric_field_template.get_times())
        self.__polarization = polarization
        self.__electric_field_template = electric_field_template
        self.__convergence_level = convergence_level
        self.__relative_tolerance = relative_tolerance
        self.__pulse_time_prior = pulse_time_prior
        self.__pulse_time_uncertainty = pulse_time_uncertainty
        if phase_slope not in ['both', 'negative', 'positive']:
            raise ValueError('Phase slope has to be either both, negative of positive.')
        self.__phase_slope = phase_slope
        if slope_passbands is None:
            self.__slope_passbands = [
                [
                    (130. * units.MHz, 200 * units.MHz),
                    (200. * units.MHz, 350. * units.MHz)
                ]
            ]
        if energy_fluence_passbands is None:
            self.__energy_fluence_passbands = [
                (130. * units.MHz, 500. * units.MHz)
            ]
        else:
            self.__slope_passbands = slope_passbands
        if amp_dct is None:
            self.__amp_dct = {
                'n_pix': 64,  # spectral bins
                # Spectral smoothness (affects Gaussian process part)
                'a': .01,
                'k0': 2.,
                # Power-law part of spectrum:
                'sm': -4.9,
                'sv': .5,
                'im': 2.,
                'iv': .5
            }
        else:
            self.__amp_dct = amp_dct
        return

    def make_priors_plot(self, event, station, detector, channel_ids):
        """
        Plots samples from the prior distribution of the electric field.

        Parameters
        --------------
        event: NuRadioReco.framework.event.Event object
        station: NuRadioReco.framework.station.Station object
        detector: NuRadioReco.detector.detector.Detector object or child object
        channel_ids: list of floats
            IDs of the channels to use for the electric field reconstruction
        """
        self.__used_channel_ids = []
        self.__efield_scaling = False
        self.__used_channel_ids = channel_ids
        self.__prepare_traces(event, station, detector)
        ref_channel = station.get_channel(self.__used_channel_ids[0])
        sampling_rate = ref_channel.get_sampling_rate()
        time_domain = ift.RGSpace(self.__trace_samples)
        frequency_domain = time_domain.get_default_codomain()
        self.__fft_operator = ift.FFTOperator(frequency_domain.get_default_codomain())
        amp_operators, filter_operator = self.__get_detector_operators(
            station,
            detector,
            frequency_domain,
            sampling_rate,
        )
        self.__draw_priors(event, station, frequency_domain)

    def run(self, event, station, detector, channel_ids, efield_scaling, use_sim=False):
        """
        Run the electric field reconstruction

        Parameters
        ----------------
        event: NuRadioReco.framework.event.Event object
        station: NuRadioReco.framework.station.Station object
        detector: NuRadioReco.detector.detector.Detector object or child object
        channel_ids: list of integers
            IDs of the channels to be used for the electric field reconstruction
        efield_scaling: boolean
            If true, a small variation in the amplitude between channels is included
            in the IFT model.
        use_sim: boolean
            If true, the simChannels are used to identify the position of the radio pulse.

        """
        self.__used_channel_ids = []    # only use channels with associated E-field and zenith
        self.__efield_scaling = efield_scaling
        self.__used_channel_ids = channel_ids
        self.__use_sim = use_sim
        self.__prepare_traces(event, station, detector)
        ref_channel = station.get_channel(self.__used_channel_ids[0])
        sampling_rate = ref_channel.get_sampling_rate()
        time_domain = ift.RGSpace(self.__trace_samples)
        frequency_domain = time_domain.get_default_codomain()
        large_frequency_domain = ift.RGSpace(self.__trace_samples * 2, harmonic=True)
        self.__fft_operator = ift.FFTOperator(frequency_domain.get_default_codomain())
        amp_operators, filter_operator = self.__get_detector_operators(
            station,
            detector,
            frequency_domain,
            sampling_rate,
        )
        final_KL = None
        positive_reco_KL = None
        negative_reco_KL = None
        # Run Positive Phase Slope #
        if self.__phase_slope == 'both' or self.__phase_slope == 'positive':
            phase_slope = 2. * np.pi * self.__pulse_time_prior * self.__electric_field_template.get_sampling_rate() / self.__trace_samples
            phase_uncertainty = 2. * np.pi * self.__pulse_time_uncertainty * self.__electric_field_template.get_sampling_rate() / self.__trace_samples
            self.__phase_dct = {
                'sm': phase_slope,
                'sv': phase_uncertainty,
                'im': 0.,
                'iv': 10.
            }
            likelihood = self.__get_likelihood_operator(
                frequency_domain,
                large_frequency_domain,
                amp_operators,
                filter_operator
            )
            self.__draw_priors(event, station, frequency_domain)
            ic_sampling = ift.GradientNormController(1E-8, iteration_limit=min(1000, likelihood.domain.size))
            H = ift.StandardHamiltonian(likelihood, ic_sampling)

            ic_newton = ift.DeltaEnergyController(name='newton',
                                                  iteration_limit=200,
                                                  tol_rel_deltaE=self.__relative_tolerance,
                                                  convergence_level=self.__convergence_level)
            minimizer = ift.NewtonCG(ic_newton)
            median = ift.MultiField.full(H.domain, 0.)
            min_energy = None
            best_reco_KL = None
            for k in range(self.__n_iterations):
                print('----------->>>   {}   <<<-----------'.format(k))
                KL = ift.MetricGaussianKL(median, H, self.__n_samples, mirror_samples=True)
                KL, convergence = minimizer(KL)
                median = KL.position
                if min_energy is None or KL.value < min_energy:
                    min_energy = KL.value
                    print('New min Energy', KL.value)
                    best_reco_KL = KL
                    if self.__phase_slope == 'both':
                        suffix = '_positive_phase'
                    else:
                        suffix = ''
                    if self.__debug:
                        self.__draw_reconstruction(
                            event,
                            station,
                            KL,
                            suffix
                        )
            positive_reco_KL = best_reco_KL
            final_KL = best_reco_KL
        # Run Negative Phase Slope ###
        if self.__phase_slope == 'both' or self.__phase_slope == 'negative':
            phase_slope = 2. * np.pi * (self.__pulse_time_prior * self.__electric_field_template.get_sampling_rate() - self.__trace_samples) / self.__trace_samples
            phase_uncertainty = 2. * np.pi * self.__pulse_time_uncertainty * self.__electric_field_template.get_sampling_rate() / self.__trace_samples
            self.__phase_dct = {
                'sm': phase_slope,
                'sv': phase_uncertainty,
                'im': 0.,
                'iv': 10.
            }
            likelihood = self.__get_likelihood_operator(
                frequency_domain,
                large_frequency_domain,
                amp_operators,
                filter_operator
            )
            # self.__draw_priors(event, station, frequency_domain)
            ic_sampling = ift.GradientNormController(1E-8, iteration_limit=min(1000, likelihood.domain.size))
            H = ift.StandardHamiltonian(likelihood, ic_sampling)

            ic_newton = ift.DeltaEnergyController(name='newton',
                                                  iteration_limit=200,
                                                  tol_rel_deltaE=self.__relative_tolerance,
                                                  convergence_level=self.__convergence_level)
            minimizer = ift.NewtonCG(ic_newton)
            median = ift.MultiField.full(H.domain, 0.)
            min_energy = None
            best_reco_KL = None
            for k in range(self.__n_iterations):
                print('----------->>>   {}   <<<-----------'.format(k))
                KL = ift.MetricGaussianKL(median, H, self.__n_samples, mirror_samples=True)
                KL, convergence = minimizer(KL)
                median = KL.position
                if min_energy is None or KL.value < min_energy:
                    min_energy = KL.value
                    print('New min Energy', KL.value)
                    best_reco_KL = KL
                    if self.__phase_slope == 'both':
                        suffix = '_negative_phase'
                    else:
                        suffix = ''
                    if self.__debug:
                        self.__draw_reconstruction(
                            event,
                            station,
                            KL,
                            suffix
                        )
            negative_reco_KL = best_reco_KL
            final_KL = best_reco_KL
        if self.__phase_slope == 'both':
            if negative_reco_KL.value < positive_reco_KL.value:
                final_KL = negative_reco_KL
            else:
                final_KL = positive_reco_KL
        self.__store_reconstructed_efields(
            event, station, final_KL
        )
        if self.__debug:
            self.__draw_reconstruction(
                event,
                station,
                final_KL,
                ''
            )
        return True

    def __prepare_traces(
        self,
        event,
        station,
        det
    ):
        """
        Prepares the channel waveforms for the reconstruction by correcting
        for time differences between channels, cutting them to the
        right size and locating the radio pulse.
        """
        if self.__debug:
            plt.close('all')
            fig1 = plt.figure(figsize=(18, 12))
            ax1_1 = fig1.add_subplot(len(self.__used_channel_ids), 2, (1, 2 * len(self.__used_channel_ids) - 1))
            fig2 = plt.figure(figsize=(18, 12))

        self.__noise_levels = np.zeros(len(self.__used_channel_ids))
        self.__n_shifts = np.zeros_like(self.__used_channel_ids)
        self.__trace_start_times = np.zeros(len(self.__used_channel_ids))
        self.__data_traces = np.zeros((len(self.__used_channel_ids), self.__trace_samples))
        max_channel_length = 0
        passband = [100. * units.MHz, 200 * units.MHz]
        sim_channel_traces = []
        for channel_id in self.__used_channel_ids:
            channel = station.get_channel(channel_id)
            if self.__use_sim:
                sim_channel_sum = NuRadioReco.framework.base_trace.BaseTrace()
                sim_channel_sum.set_trace(np.zeros_like(channel.get_trace()), channel.get_sampling_rate())
                sim_channel_sum.set_trace_start_time(channel.get_trace_start_time())
                for sim_channel in station.get_sim_station().get_channels_by_channel_id(channel_id):
                    sim_channel_sum += sim_channel
                if sim_channel_sum.get_number_of_samples() > max_channel_length:
                    max_channel_length = sim_channel_sum.get_number_of_samples()
                sim_channel_traces.append(sim_channel_sum)
            else:
                if channel.get_number_of_samples() > max_channel_length:
                    max_channel_length = channel.get_number_of_samples()
        correlation_sum = np.zeros(self.__electric_field_template.get_number_of_samples() + max_channel_length)
        if self.__debug:
            plt.close('all')
            fig1 = plt.figure(figsize=(16, 8))
            ax1_1 = fig1.add_subplot(121)
            ax1_1.grid()
            ax1_2 = fig1.add_subplot(122)
            ax1_2.grid()
            fig2 = plt.figure(figsize=(12, 12))
        channel_trace_templates = np.zeros((len(self.__used_channel_ids), len(self.__electric_field_template.get_trace())))
        for i_channel, channel_id in enumerate(self.__used_channel_ids):
            channel = station.get_channel(channel_id)
            amp_response = det.get_amplifier_response(station.get_id(), channel_id, self.__electric_field_template.get_frequencies())
            antenna_orientation = det.get_antenna_orientation(station.get_id(), channel_id)
            antenna_pattern = self.__antenna_pattern_provider.load_antenna_pattern(det.get_antenna_model(station.get_id(), channel_id))
            antenna_response = antenna_pattern.get_antenna_response_vectorized(
                self.__electric_field_template.get_frequencies(),
                channel.get_parameter(chp.signal_receiving_zenith),
                0.,
                antenna_orientation[0],
                antenna_orientation[1],
                antenna_orientation[2],
                antenna_orientation[3]
            )
            channel_spectrum_template = fft.time2freq(
                self.__electric_field_template.get_filtered_trace(passband, filter_type='butterabs'),
                self.__electric_field_template.get_sampling_rate()
            ) * amp_response * (antenna_response['theta'] + antenna_response['phi'])
            channel_trace_template = fft.freq2time(channel_spectrum_template, self.__electric_field_template.get_sampling_rate())
            channel_trace_templates[i_channel] = channel_trace_template
            channel.apply_time_shift(-channel.get_parameter(chp.signal_time_offset), True)
            if self.__use_sim:
                sim_channel_traces[i_channel].apply_time_shift(-channel.get_parameter(chp.signal_time_offset), True)
                channel_trace = sim_channel_traces[i_channel].get_filtered_trace(passband, filter_type='butterabs')
            else:
                channel_trace = channel.get_filtered_trace(passband, filter_type='butterabs')
            if self.__use_sim:
                correlation = radiotools.helper.get_normalized_xcorr(np.abs(scipy.signal.hilbert(channel_trace_template)), np.abs(scipy.signal.hilbert(channel_trace)))
            else:
                correlation = radiotools.helper.get_normalized_xcorr(channel_trace_template, channel_trace)
            correlation = np.abs(correlation)
            correlation_sum[:len(correlation)] += correlation
            toffset = -(np.arange(0, correlation.shape[0]) - len(channel_trace)) / channel.get_sampling_rate()  # - propagation_times[i_channel, i_solution] - channel.get_trace_start_time()
            if self.__use_sim:
                sim_channel_traces[i_channel].apply_time_shift(channel.get_parameter(chp.signal_time_offset), True)
            # else:
            #     channel.apply_time_shift(channel.get_parameter(chp.signal_time_offset), True)
            if self.__debug:
                ax1_1.plot(toffset, correlation)

        for i_channel, channel_id in enumerate(self.__used_channel_ids):
            channel = station.get_channel(channel_id)
            time_offset = channel.get_parameter(chp.signal_time_offset)
            channel_trace = channel.get_filtered_trace(passband, filter_type='butterabs')
            toffset = -(np.arange(0, correlation_sum.shape[0]) - len(channel_trace)) / channel.get_sampling_rate()
            if self.__debug:
                ax2_1 = fig2.add_subplot(len(self.__used_channel_ids), 2, 2 * i_channel + 1)
                ax2_1.grid()
                ax2_1.plot(channel.get_times(), channel_trace / units.mV, c='C0', alpha=1.)
                ax2_1.set_title('Channel {}'.format(channel_id))
                ax2_1.plot(self.__electric_field_template.get_times() + channel.get_trace_start_time() + toffset[np.argmax(correlation_sum)], channel_trace_templates[i_channel] / np.max(channel_trace_templates[i_channel]) * np.max(channel_trace) / units.mV, c='C1')
                sim_channel_sum = None
                for sim_channel in station.get_sim_station().iter_channels():
                    if sim_channel.get_id() == channel_id:
                        if sim_channel_sum is None:
                            sim_channel_sum = sim_channel
                        else:
                            sim_channel_sum += sim_channel
                if sim_channel_sum is not None:
                    sim_channel_sum.apply_time_shift(-channel.get_parameter(chp.signal_time_offset), True)
                    ax2_1.plot(sim_channel_sum.get_times(), sim_channel_sum.get_filtered_trace(passband, filter_type='butterabs') / units.mV, c='k', alpha=.5)
                    ax2_1.set_xlim([sim_channel_sum.get_trace_start_time() - 50, sim_channel_sum.get_times()[-1] + 50])
                    sim_channel_sum.apply_time_shift(channel.get_parameter(chp.signal_time_offset), True)

            channel.apply_time_shift(-toffset[np.argmax(correlation_sum)])
            self.__data_traces[i_channel] = channel.get_trace()[:self.__trace_samples]
            self.__noise_levels[i_channel] = np.sqrt(np.mean(channel.get_trace()[self.__trace_samples + 1:]**2))
            self.__n_shifts[i_channel] = int((toffset[np.argmax(correlation_sum)] + time_offset) * channel.get_sampling_rate())
            self.__trace_start_times[i_channel] = channel.get_trace_start_time() + (toffset[np.argmax(correlation_sum)] + time_offset)
            if self.__debug:
                ax2_2 = fig2.add_subplot(len(self.__used_channel_ids), 2, 2 * i_channel + 2)
                ax2_2.grid()
                ax2_2.plot(np.arange(len(self.__data_traces[i_channel])) / channel.get_sampling_rate(), self.__data_traces[i_channel])
            channel.apply_time_shift(channel.get_parameter(chp.signal_time_offset) + toffset[np.argmax(correlation_sum)], True)
        self.__scaling_factor = np.max(self.__data_traces)
        self.__data_traces /= self.__scaling_factor
        self.__noise_levels /= self.__scaling_factor

        if self.__debug:
            ax1_2.plot(correlation_sum)
            fig2.tight_layout()
            fig2.savefig('{}_{}_traces.png'.format(event.get_run_number(), event.get_id()))

    def __get_detector_operators(
        self,
        station,
        detector,
        frequency_domain,
        sampling_rate
    ):
        """
        Creates the operators to simulate the detector response.
        """
        amp_operators = []
        self.__gain_scaling = []
        self.__classic_efield_recos = []
        frequencies = frequency_domain.get_k_length_array().val / self.__trace_samples * sampling_rate
        hardware_responses = np.zeros((len(self.__used_channel_ids), 2, len(frequencies)), dtype=complex)
        if self.__passband is not None:
            b, a = scipy.signal.butter(10, self.__passband, 'bandpass', analog=True)
            w, h = scipy.signal.freqs(b, a, frequencies)
            filter_field = ift.Field(ift.DomainTuple.make(frequency_domain), np.abs(h))
            filter_operator = ift.DiagonalOperator(filter_field, frequency_domain)
            if self.__filter_type == 'butter':
                filter_phase = np.unwrap(np.angle(h))
            else:
                filter_phase = 0
        else:
            filter_operator = ift.ScalingOperator(1., frequency_domain)
            filter_phase = 0
        for i_channel, channel_id in enumerate(self.__used_channel_ids):
            channel = station.get_channel(channel_id)
            receiving_zenith = channel.get_parameter(chp.signal_receiving_zenith)
            if channel.has_parameter(chp.signal_receiving_azimuth):
                receive_azimuth = channel.get_parameter(chp.signal_receiving_azimuth)
            else:
                receive_azimuth = 0.
            antenna_response = NuRadioReco.utilities.trace_utilities.get_efield_antenna_factor(station, frequencies, [channel_id], detector, receiving_zenith, receive_azimuth, self.__antenna_pattern_provider)[0]
            amp_response = detector.get_amplifier_response(station.get_id(), channel_id, frequencies)
            amp_gain = np.abs(amp_response)
            amp_phase = np.unwrap(np.angle(amp_response))
            total_gain = np.abs(amp_gain) * np.abs(antenna_response)
            total_phase = np.unwrap(np.angle(antenna_response)) + amp_phase + filter_phase
            total_phase[:, total_phase.shape[1] // 2:] *= -1
            total_phase[:, total_phase.shape[1] // 2 + 1] = 0
            total_phase *= -1
            hardware_responses[i_channel, 0] = (total_gain * np.exp(1.j * total_phase))[0]
            hardware_responses[i_channel, 1] = (total_gain * np.exp(1.j * total_phase))[1]
        max_gain = np.max(np.abs(hardware_responses))
        self.__gain_scaling = max_gain
        hardware_responses /= max_gain
        for i_channel, channel_id in enumerate(self.__used_channel_ids):
            amp_field_theta = ift.Field(ift.DomainTuple.make(frequency_domain), hardware_responses[i_channel][0])
            amp_field_phi = ift.Field(ift.DomainTuple.make(frequency_domain), hardware_responses[i_channel][1])
            amp_operators.append([ift.DiagonalOperator(amp_field_theta), ift.DiagonalOperator(amp_field_phi)])

        return amp_operators, filter_operator

    def __get_likelihood_operator(
        self,
        frequency_domain,
        large_frequency_domain,
        hardware_operators,
        filter_operator
    ):
        """
        Creates the IFT model from which the maximum posterior is calculated
        """
        power_domain = ift.RGSpace(large_frequency_domain.get_default_codomain().shape[0], harmonic=True)
        power_space = ift.PowerSpace(power_domain)
        self.__amp_dct['target'] = power_space
        A = ift.SLAmplitude(**self.__amp_dct)
        self.__power_spectrum_operator = A
        correlated_field = ift.CorrelatedField(large_frequency_domain.get_default_codomain(), A)
        realizer = ift.Realizer(self.__fft_operator.domain)
        realizer2 = ift.Realizer(self.__fft_operator.target)
        inserter = NuRadioReco.modules.iftElectricFieldReconstructor.operators.Inserter(realizer.target)
        large_sp = correlated_field.target
        small_sp = ift.RGSpace(large_sp.shape[0] // 2, large_sp[0].distances)
        zero_padder = ift.FieldZeroPadder(small_sp, large_sp.shape, central=False)
        domain_flipper = NuRadioReco.modules.iftElectricFieldReconstructor.operators.DomainFlipper(zero_padder.domain, target=ift.RGSpace(small_sp.shape, harmonic=True))
        mag_S_h = (domain_flipper @ zero_padder.adjoint @ correlated_field)
        mag_S_h = NuRadioReco.modules.iftElectricFieldReconstructor.operators.SymmetrizingOperator(mag_S_h.target) @ mag_S_h
        subtract_one = ift.Adder(ift.Field(mag_S_h.target, -6))
        mag_S_h = realizer2.adjoint @ (subtract_one @ mag_S_h).exp()
        fft_operator = ift.FFTOperator(frequency_domain.get_default_codomain())

        scaling_domain = ift.UnstructuredDomain(1)
        add_one = ift.Adder(ift.Field(inserter.domain, 1))

        polarization_domain = ift.UnstructuredDomain(1)
        likelihood = None
        self.__efield_trace_operators = []
        self.__efield_spec_operators = []
        self.__channel_trace_operators = []
        self.__channel_spec_operators = []
        polarization_inserter = NuRadioReco.modules.iftElectricFieldReconstructor.operators.Inserter(mag_S_h.target)
        polarization_field = realizer2 @ polarization_inserter @ (2. * ift.FieldAdapter(polarization_domain, 'pol'))
        for i_channel, channel_id in enumerate(self.__used_channel_ids):
            phi_S_h = NuRadioReco.modules.iftElectricFieldReconstructor.operators.SlopeSpectrumOperator(frequency_domain.get_default_codomain(), self.__phase_dct['sm'], self.__phase_dct['im'], self.__phase_dct['sv'], self.__phase_dct['iv'])
            phi_S_h = realizer2.adjoint @ phi_S_h
            scaling_field = (inserter @ add_one @ (.1 * ift.FieldAdapter(scaling_domain, 'scale{}'.format(i_channel))))
            if self.__polarization == 'theta':
                efield_spec_operator_theta = ((filter_operator @ (mag_S_h * (1.j * phi_S_h).exp())))
                efield_spec_operator_phi = None
                channel_spec_operator = (hardware_operators[i_channel][0] @ efield_spec_operator_theta)
            elif self.__polarization == 'phi':
                efield_spec_operator_theta = None
                efield_spec_operator_phi = ((filter_operator @ (mag_S_h * (1.j * phi_S_h).exp())))
                channel_spec_operator = (hardware_operators[i_channel][1] @ efield_spec_operator_phi)
            elif self.__polarization == 'pol':
                efield_spec_operator_theta = ((filter_operator @ ((mag_S_h * polarization_field.cos()) * (1.j * phi_S_h).exp())))
                efield_spec_operator_phi = ((filter_operator @ ((mag_S_h * polarization_field.sin()) * (1.j * phi_S_h).exp())))
                channel_spec_operator = (hardware_operators[i_channel][0] @ efield_spec_operator_theta) + (hardware_operators[i_channel][1] @ efield_spec_operator_phi)
            else:
                raise ValueError('Unrecognized polarization setting {}. Possible values are theta, phi and pol'.format(self.__polarization))
            efield_spec_operators = [
                efield_spec_operator_theta,
                efield_spec_operator_phi
            ]
            efield_trace_operator = []
            if self.__efield_scaling:
                for efield_spec_operator in efield_spec_operators:
                    if efield_spec_operator is not None:
                        efield_trace_operator.append(((realizer @ fft_operator.inverse @ efield_spec_operator)) * scaling_field)
                    else:
                        efield_trace_operator.append(None)
                channel_trace_operator = ((realizer @ fft_operator.inverse @ (channel_spec_operator))) * scaling_field
            else:
                for efield_spec_operator in efield_spec_operators:
                    if efield_spec_operator is not None:
                        efield_trace_operator.append(((realizer @ fft_operator.inverse @ efield_spec_operator)))
                    else:
                        efield_trace_operator.append(None)
                channel_trace_operator = ((realizer @ fft_operator.inverse @ (channel_spec_operator)))
            noise_operator = ift.ScalingOperator(self.__noise_levels[i_channel]**2, frequency_domain.get_default_codomain())
            data_field = ift.Field(ift.DomainTuple.make(frequency_domain.get_default_codomain()), self.__data_traces[i_channel])
            self.__efield_spec_operators.append(efield_spec_operators)
            self.__efield_trace_operators.append(efield_trace_operator)
            self.__channel_spec_operators.append(channel_spec_operator)
            self.__channel_trace_operators.append(channel_trace_operator)
            if likelihood is None:
                likelihood = ift.GaussianEnergy(mean=data_field, inverse_covariance=noise_operator.inverse)(self.__channel_trace_operators[i_channel])
            else:
                likelihood += ift.GaussianEnergy(mean=data_field, inverse_covariance=noise_operator.inverse)(self.__channel_trace_operators[i_channel])
        return likelihood

    def __store_reconstructed_efields(
        self,
        event,
        station,
        KL
    ):
        """
        Ads electric fields containing the reconstruction results to the station
        """
        if self.__efield_scaling:
            for i_channel, channel_id in enumerate(self.__used_channel_ids):
                efield = self.__get_reconstructed_efield(KL, i_channel)
                station.add_electric_field(efield)
        else:
            # The reconstructed electric field is the same for all channels, so it does not matter what we pick for
            # i_channel
            efield = self.__get_reconstructed_efield(KL, 0)
            efield.set_channel_ids(self.__used_channel_ids)
            station.add_electric_field(efield)

    def __get_reconstructed_efield(
        self,
        KL,
        i_channel
    ):
        """
        Creates an electric field object containing the reconstruction results.
        """
        median = KL.position
        efield_stat_calculators = [ift.StatCalculator(), ift.StatCalculator()]
        polarization_stat_calculator = ift.StatCalculator()
        energy_fluence_stat_calculator = ift.StatCalculator()
        slope_parameter_stat_calculator = ift.StatCalculator()
        rec_efield = np.zeros((3, self.__electric_field_template.get_number_of_samples()))
        sampling_rate = self.__electric_field_template.get_sampling_rate()
        times = np.arange(self.__data_traces.shape[1]) / sampling_rate
        freqs = np.fft.rfftfreq(rec_efield.shape[1], 1. / sampling_rate)
        for sample in KL.samples:
            efield_sample_pol = np.zeros_like(rec_efield)
            if self.__efield_trace_operators[i_channel][0] is not None:
                efield_sample_theta = self.__efield_trace_operators[i_channel][0].force(median + sample).val
                efield_stat_calculators[0].add(efield_sample_theta)
                efield_sample_pol[1] = efield_sample_theta
            if self.__efield_trace_operators[i_channel][1] is not None:
                efield_sample_phi = self.__efield_trace_operators[i_channel][1].force(median + sample).val
                efield_stat_calculators[1].add(efield_sample_phi)
                efield_sample_pol[2] = efield_sample_phi
            if self.__polarization == 'pol':
                energy_fluences = trace_utilities.get_electric_field_energy_fluence(
                    efield_sample_pol,
                    times
                )
                polarization_stat_calculator.add(np.arctan(np.sqrt(energy_fluences[2]) / np.sqrt(energy_fluences[1])))
            e_fluences = np.zeros((len(self.__energy_fluence_passbands), 3))
            for i_passband, passband in enumerate(self.__energy_fluence_passbands):
                filter_response = bandpass_filter.get_filter_response(freqs, passband, 'butter', 10)
                e_fluence = trace_utilities.get_electric_field_energy_fluence(
                    fft.freq2time(fft.time2freq(efield_sample_pol, sampling_rate) * filter_response, sampling_rate) * self.__scaling_factor / self.__gain_scaling,
                    times
                )
                e_fluence[0] = np.sum(np.abs(e_fluence))
                e_fluences[i_passband] = e_fluence
            energy_fluence_stat_calculator.add(e_fluences)
            slopes = np.zeros((len(self.__slope_passbands), 3))
            for i_passband, passbands in enumerate(self.__slope_passbands):
                filter_response_1 = bandpass_filter.get_filter_response(freqs, passbands[0], 'butter', 10)
                e_fluence_1 = trace_utilities.get_electric_field_energy_fluence(
                    fft.freq2time(fft.time2freq(efield_sample_pol, sampling_rate) * filter_response_1, sampling_rate) * self.__scaling_factor / self.__gain_scaling,
                    times
                )
                e_fluence_1[0] = np.sum(np.abs(e_fluence_1))
                filter_response_2 = bandpass_filter.get_filter_response(freqs, passbands[1], 'butter', 10)
                e_fluence_2 = trace_utilities.get_electric_field_energy_fluence(
                    fft.freq2time(fft.time2freq(efield_sample_pol, sampling_rate) * filter_response_2, sampling_rate) * self.__scaling_factor / self.__gain_scaling,
                    times
                )
                e_fluence_2[0] = np.sum(np.abs(e_fluence_2))
                if self.__polarization == 'pol':
                    slopes[i_passband] = e_fluence_1[0] / e_fluence_2[0]
                elif self.__polarization == 'theta':
                    slopes[i_passband] = e_fluence_1[1] / e_fluence_2[1]
                else:
                    slopes[i_passband] = e_fluence_1[2] / e_fluence_2[2]
            slope_parameter_stat_calculator.add(slopes)
        if self.__efield_trace_operators[i_channel][0] is not None:
            rec_efield[1] = efield_stat_calculators[0].mean * self.__scaling_factor / self.__gain_scaling
        if self.__efield_trace_operators[i_channel][1] is not None:
            rec_efield[2] = efield_stat_calculators[1].mean * self.__scaling_factor / self.__gain_scaling
        efield = NuRadioReco.framework.electric_field.ElectricField([self.__used_channel_ids[i_channel]])
        efield.set_trace(rec_efield, self.__electric_field_template.get_sampling_rate())
        if self.__polarization == 'pol':
            efield.set_parameter(efp.polarization_angle, polarization_stat_calculator.mean)
            efield.set_parameter_error(efp.polarization_angle, np.sqrt(polarization_stat_calculator.var))
        energy_fluence_dict = {}
        slope_dict = {}
        for i_passband, passband in enumerate(self.__energy_fluence_passbands):
            energy_fluence_dict['{:.0f}-{:.0f}'.format(passband[0] / units.MHz, passband[1] / units.MHz)] = energy_fluence_stat_calculator.mean[i_passband]
        for i_passband, passbands in enumerate(self.__slope_passbands):
            slope_dict['{:.0f}-{:.0f}, {:.0f}-{:.0f}'.format(passbands[0][0], passbands[0][1], passbands[1][0], passbands[1][1])] = slope_parameter_stat_calculator.mean[i_passband]
        energy_fluence_error = np.sqrt(energy_fluence_stat_calculator.var)
        efield.set_parameter(efp.signal_energy_fluence, energy_fluence_dict)
        efield.set_parameter_error(efp.signal_energy_fluence, energy_fluence_error)
        efield.set_parameter(efp.energy_fluence_ratios, slope_dict)
        efield.set_parameter_error(efp.energy_fluence_ratios, np.sqrt(slope_parameter_stat_calculator.var))
        return efield

    def __draw_priors(
        self,
        event,
        station,
        freq_space
    ):
        """
        Draws samples from the prior distribution of the electric field spectrum.
        """
        plt.close('all')
        fig1 = plt.figure(figsize=(12, 8))
        ax1_0 = fig1.add_subplot(3, 2, (1, 2))
        ax1_1 = fig1.add_subplot(323)
        ax1_2 = fig1.add_subplot(324)
        ax1_3 = fig1.add_subplot(325)
        ax1_4 = fig1.add_subplot(326)
        sampling_rate = station.get_channel(self.__used_channel_ids[0]).get_sampling_rate()
        times = np.arange(self.__data_traces.shape[1]) / sampling_rate
        freqs = freq_space.get_k_length_array().val / self.__data_traces.shape[1] * sampling_rate
        alpha = .8
        for i in range(8):
            x = ift.from_random('normal', self.__efield_trace_operators[0][0].domain)
            efield_spec_sample = self.__efield_spec_operators[0][0].force(x)
            ax1_1.plot(freqs / units.MHz, np.abs(efield_spec_sample.val) / np.max(np.abs(efield_spec_sample.val)), c='C{}'.format(i), alpha=alpha)
            efield_trace_sample = self.__efield_trace_operators[0][0].force(x)
            ax1_2.plot(times, efield_trace_sample.val / np.max(np.abs(efield_trace_sample.val)))
            channel_spec_sample = self.__channel_spec_operators[0].force(x)
            ax1_3.plot(freqs / units.MHz, np.abs(channel_spec_sample.val))  # / np.max(np.abs(channel_spec_sample.val)), c='C{}'.format(i), alpha=alpha)
            channel_trace_sample = self.__channel_trace_operators[0].force(x)
            ax1_4.plot(times, channel_trace_sample.val / np.max(np.abs(channel_trace_sample.val)), c='C{}'.format(i), alpha=alpha)
            a = self.__power_spectrum_operator.force(x).val
            power_freqs = self.__power_spectrum_operator.target[0].k_lengths / self.__data_traces.shape[1] * sampling_rate
            ax1_0.plot(power_freqs, a, c='C{}'.format(i), alpha=alpha)
        ax1_0.grid()
        ax1_0.set_xscale('log')
        ax1_0.set_yscale('log')
        ax1_0.set_title('Power Spectrum')
        ax1_0.set_xlabel('k')
        ax1_0.set_ylabel('A')
        ax1_1.grid()
        ax1_1.set_xlim([50, 750])
        ax1_2.grid()
        ax1_2.set_xlabel('t [ns]')
        ax1_2.set_ylabel('E [a.u.]')
        ax1_2.set_title('E-Field Trace')
        ax1_3.grid()
        ax1_3.set_xlim([50, 750])
        ax1_4.grid()
        ax1_4.set_xlim([0, 150])
        ax1_1.set_xlabel('f [MHz]')
        # ax1_2.set_xlabel('t [ns]')
        ax1_3.set_xlabel('f [MHz]')
        ax1_4.set_xlabel('t [ns]')
        ax1_1.set_ylabel('E [a.u.]')
        # ax1_2.set_ylabel('E [a.u.]')
        ax1_3.set_ylabel('U [a.u.]')
        ax1_4.set_ylabel('U [a.u.]')
        ax1_1.set_title('E-Field Spectrum')
        # ax1_2.set_title('E-Field Trace')
        ax1_3.set_title('Channel Spectrum')
        ax1_4.set_title('Channel Trace')
        fig1.tight_layout()
        fig1.savefig('priors_{}_{}.png'.format(event.get_id(), event.get_run_number()))

    def __draw_reconstruction(
        self,
        event,
        station,
        KL,
        suffix=''
    ):
        """
        Draw plots showing the results of the reconstruction.
        """
        plt.close('all')
        fontsize = 16
        n_channels = len(self.__used_channel_ids)
        median = KL.position
        sampling_rate = station.get_channel(self.__used_channel_ids[0]).get_sampling_rate()
        fig1 = plt.figure(figsize=(16, 4 * n_channels))
        fig2 = plt.figure(figsize=(16, 4 * n_channels))
        freqs = np.fft.rfftfreq(self.__data_traces.shape[1], 1. / sampling_rate)
        classic_mean_efield_spec = np.zeros_like(freqs)
        classic_mean_efield_spec /= len(self.__used_channel_ids)
        for i_channel, channel_id in enumerate(self.__used_channel_ids):
            times = np.arange(self.__data_traces.shape[1]) / sampling_rate + self.__trace_start_times[i_channel]
            trace_stat_calculator = ift.StatCalculator()
            amp_trace_stat_calculator = ift.StatCalculator()
            efield_stat_calculators = [ift.StatCalculator(), ift.StatCalculator()]
            amp_efield_stat_calculators = [ift.StatCalculator(), ift.StatCalculator()]
            if self.__polarization == 'pol':
                ax1_1 = fig1.add_subplot(n_channels, 3, 3 * i_channel + 1)
                ax1_2 = fig1.add_subplot(n_channels, 3, 3 * i_channel + 2)
                ax1_3 = fig1.add_subplot(n_channels, 3, 3 * i_channel + 3, sharey=ax1_2)
                ax1_2.set_title(r'$\theta$ component', fontsize=fontsize)
                ax1_3.set_title(r'$\varphi$ component', fontsize=fontsize)
            else:
                ax1_1 = fig1.add_subplot(n_channels, 2, 2 * i_channel + 1)
                ax1_2 = fig1.add_subplot(n_channels, 2, 2 * i_channel + 2)
            ax2_1 = fig2.add_subplot(n_channels, 1, i_channel + 1)
            for sample in KL.samples:
                for i_pol, efield_stat_calculator in enumerate(efield_stat_calculators):
                    channel_sample_trace = self.__channel_trace_operators[i_channel].force(median + sample).val
                    trace_stat_calculator.add(channel_sample_trace)
                    amp_trace = np.abs(fft.time2freq(channel_sample_trace, sampling_rate))
                    amp_trace_stat_calculator.add(amp_trace)
                    ax2_1.plot(times, channel_sample_trace * self.__scaling_factor / units.mV, c='k', alpha=.2)
                    ax1_1.plot(freqs / units.MHz, amp_trace * self.__scaling_factor / units.mV, c='k', alpha=.2)
                    if self.__efield_trace_operators[i_channel][i_pol] is not None:
                        efield_sample_trace = self.__efield_trace_operators[i_channel][i_pol].force(median + sample).val
                        efield_stat_calculator.add(efield_sample_trace)
                        amp_efield = np.abs(fft.time2freq(efield_sample_trace, sampling_rate))
                        amp_efield_stat_calculators[i_pol].add(amp_efield)
                        if self.__polarization == 'pol':
                            if i_pol == 0:
                                ax1_2.plot(freqs / units.MHz, amp_efield * self.__scaling_factor / self.__gain_scaling / (units.mV / units.m / units.GHz), c='k', alpha=.2)
                            else:
                                ax1_3.plot(freqs / units.MHz, amp_efield * self.__scaling_factor / self.__gain_scaling / (units.mV / units.m / units.GHz), c='k', alpha=.2)
                        else:
                            ax1_2.plot(freqs / units.MHz, amp_efield * self.__scaling_factor / self.__gain_scaling / (units.mV / units.m / units.GHz), c='k', alpha=.2)

            ax1_1.plot(freqs / units.MHz, np.abs(fft.time2freq(self.__data_traces[i_channel], sampling_rate)) * self.__scaling_factor / units.mV, c='C0', label='data')
            sim_efield_max = None
            channel_snr = None
            if station.has_sim_station():
                sim_station = station.get_sim_station()
                n_drawn_sim_channels = 0
                for ray_tracing_id in sim_station.get_ray_tracing_ids():
                    sim_channel_sum = None
                    for sim_channel in sim_station.get_channels_by_ray_tracing_id(ray_tracing_id):
                        if sim_channel.get_id() == channel_id:
                            if sim_channel_sum is None:
                                sim_channel_sum = sim_channel
                            else:
                                sim_channel_sum += sim_channel
                            ax1_1.plot(sim_channel.get_frequencies() / units.MHz, np.abs(sim_channel.get_frequency_spectrum()) / units.mV, c='C1', linestyle='--', alpha=.5)
                            ax2_1.plot(sim_channel.get_times(), sim_channel.get_trace() / units.mV, c='C1', linewidth=6, zorder=1, linestyle='--', alpha=.5)
                    if sim_channel_sum is not None:
                        if n_drawn_sim_channels == 0:
                            sim_channel_label = 'MC truth'
                        else:
                            sim_channel_label = None
                        snr = .5 * (np.max(sim_channel_sum.get_trace()) - np.min(sim_channel_sum.get_trace())) / (self.__noise_levels[i_channel] * self.__scaling_factor)
                        if channel_snr is None or snr > channel_snr:
                            channel_snr = snr
                        ax1_1.plot(
                            sim_channel_sum.get_frequencies() / units.MHz,
                            np.abs(sim_channel_sum.get_frequency_spectrum()) / units.mV,
                            c='C1',
                            label=sim_channel_label,
                            alpha=.8,
                            linewidth=2
                        )
                        ax2_1.plot(
                            sim_channel_sum.get_times(),
                            sim_channel_sum.get_trace() / units.mV,
                            c='C1',
                            linewidth=6,
                            zorder=1,
                            label=sim_channel_label
                        )
                        n_drawn_sim_channels += 1
                    efield_sum = None
                    for efield in station.get_sim_station().get_electric_fields_for_channels([channel_id]):
                        if efield.get_ray_tracing_solution_id() == ray_tracing_id:
                            if self.__polarization == 'theta':
                                ax1_2.plot(efield.get_frequencies() / units.MHz, np.abs(efield.get_frequency_spectrum()[1]) / (units.mV / units.m / units.GHz), c='C1', alpha=.2, linestyle='--')
                            if self.__polarization == 'phi':
                                ax1_2.plot(efield.get_frequencies() / units.MHz, np.abs(efield.get_frequency_spectrum()[2]) / (units.mV / units.m / units.GHz), c='C1', alpha=.2, linestyle='--')
                            if self.__polarization == 'pol':
                                ax1_2.plot(efield.get_frequencies() / units.MHz, np.abs(efield.get_frequency_spectrum()[1]) / (units.mV / units.m / units.GHz), c='C1', alpha=.2, linestyle='--')
                                ax1_3.plot(efield.get_frequencies() / units.MHz, np.abs(efield.get_frequency_spectrum()[2]) / (units.mV / units.m / units.GHz), c='C1', alpha=.2, linestyle='--')
                            if efield_sum is None:
                                efield_sum = efield
                            else:
                                efield_sum += efield
                    if efield_sum is not None:
                        if self.__polarization == 'theta':
                            ax1_2.plot(efield_sum.get_frequencies() / units.MHz, np.abs(efield_sum.get_frequency_spectrum()[1]) / (units.mV / units.m / units.GHz), c='C1', alpha=1.)
                        if self.__polarization == 'phi':
                            ax1_2.plot(efield_sum.get_frequencies() / units.MHz, np.abs(efield_sum.get_frequency_spectrum()[2]) / (units.mV / units.m / units.GHz), c='C1', alpha=1.)
                        if self.__polarization == 'pol':
                            ax1_2.plot(efield_sum.get_frequencies() / units.MHz, np.abs(efield_sum.get_frequency_spectrum()[1]) / (units.mV / units.m / units.GHz), c='C1', alpha=1.)
                            ax1_3.plot(efield_sum.get_frequencies() / units.MHz, np.abs(efield_sum.get_frequency_spectrum()[2]) / (units.mV / units.m / units.GHz), c='C1', alpha=1.)
                        if sim_efield_max is None or np.max(np.abs(efield_sum.get_frequency_spectrum())) > sim_efield_max:
                            sim_efield_max = np.max(np.abs(efield_sum.get_frequency_spectrum()))
            else:
                channel_snr = .5 * (np.max(station.get_channel(channel_id).get_trace()) - np.min(station.get_channel(channel_id).get_trace())) / (self.__noise_levels * self.__scaling_factor)
            ax2_1.plot(times, self.__data_traces[i_channel] * self.__scaling_factor / units.mV, c='C0', alpha=1., zorder=5, label='data')

            ax1_1.plot(freqs / units.MHz, amp_trace_stat_calculator.mean * self.__scaling_factor / units.mV, c='C2', label='IFT reco', linewidth=3, alpha=.6)
            ax2_1.plot(times, trace_stat_calculator.mean * self.__scaling_factor / units.mV, c='C2', linestyle='-', zorder=2, linewidth=4, label='IFT reconstruction')
            ax2_1.set_xlim([times[0], times[-1]])
            if channel_snr is not None:
                textbox = dict(boxstyle='round', facecolor='white', alpha=.5)
                ax2_1.text(.9, .05, 'SNR={:.1f}'.format(channel_snr), transform=ax2_1.transAxes, bbox=textbox, fontsize=18)
            if self.__polarization == 'theta':
                ax1_2.plot(freqs / units.MHz, amp_efield_stat_calculators[0].mean * self.__scaling_factor / self.__gain_scaling / (units.mV / units.m / units.GHz), c='C2', alpha=.6)
            if self.__polarization == 'phi':
                ax1_2.plot(freqs / units.MHz, amp_efield_stat_calculators[1].mean * self.__scaling_factor / self.__gain_scaling / (units.mV / units.m / units.GHz), c='C2', alpha=.6)
            if self.__polarization == 'pol':
                ax1_2.plot(freqs / units.MHz, np.abs(fft.time2freq(efield_stat_calculators[0].mean, sampling_rate)) * self.__scaling_factor / self.__gain_scaling / (units.mV / units.m / units.GHz), c='C2', alpha=.6)
                ax1_3.plot(freqs / units.MHz, np.abs(fft.time2freq(efield_stat_calculators[1].mean, sampling_rate)) * self.__scaling_factor / self.__gain_scaling / (units.mV / units.m / units.GHz), c='C2', alpha=.6)
            ax1_1.axvline(self.__passband[0] / units.MHz, c='k', alpha=.5, linestyle=':')
            ax1_1.axvline(self.__passband[1] / units.MHz, c='k', alpha=.5, linestyle=':')
            ax1_2.axvline(self.__passband[0] / units.MHz, c='k', alpha=.5, linestyle=':')
            ax1_2.axvline(self.__passband[1] / units.MHz, c='k', alpha=.5, linestyle=':')
            ax1_1.grid()
            ax1_2.grid()
            ax2_1.grid()
            if self.__polarization == 'pol':
                ax1_3.axvline(self.__passband[0] / units.MHz, c='k', alpha=.5, linestyle=':')
                ax1_3.axvline(self.__passband[1] / units.MHz, c='k', alpha=.5, linestyle=':')
                ax1_3.grid()
                ax1_3.set_xlim([0, 750])
                ax1_3.set_xlabel('f [MHz]')
            if i_channel == 0:
                ax2_1.legend(fontsize=fontsize)
                ax1_1.legend(fontsize=fontsize)
            ax1_1.set_xlim([0, 750])
            ax1_2.set_xlim([0, 750])
            ax1_1.set_title('Channel {}'.format(channel_id), fontsize=fontsize)
            ax2_1.set_title('Channel {}'.format(channel_id), fontsize=fontsize)
            ax1_1.set_xlabel('f [MHz]', fontsize=fontsize)
            ax1_2.set_xlabel('f [MHz]', fontsize=fontsize)
            ax1_1.set_ylabel('channel voltage [mV/GHz]', fontsize=fontsize)
            ax1_2.set_ylabel('E-Field [mV/m/GHz]', fontsize=fontsize)
            ax2_1.set_xlabel('t [ns]', fontsize=fontsize)
            ax2_1.set_ylabel('U [mV]', fontsize=fontsize)
            ax2_1_dummy = ax2_1.twiny()
            ax2_1_dummy.set_xlim(ax2_1.get_xlim())
            ax2_1_dummy.set_xticks(np.arange(times[0], times[-1], 10))

            def get_ticklabels(ticks):
                return ['{:.0f}'.format(tick) for tick in np.arange(times[0], times[-1], 10) - times[0]]
            ax2_1_dummy.set_xticklabels(get_ticklabels(np.arange(times[0], times[-1], 10)), fontsize=fontsize)
            ax1_1.tick_params(axis='both', labelsize=fontsize)
            ax1_2.tick_params(axis='both', labelsize=fontsize)
            ax2_1.tick_params(axis='both', labelsize=fontsize)
            if self.__polarization == 'pol':
                ax1_3.tick_params(axis='both', labelsize=fontsize)
            if sim_efield_max is not None:
                ax1_2.set_ylim([0, 1.2 * sim_efield_max / (units.mV / units.m / units.GHz)])
        fig1.tight_layout()
        fig1.savefig('{}_{}_spec_reco{}.png'.format(event.get_run_number(), event.get_id(), suffix))
        fig2.tight_layout()
        fig2.savefig('{}_{}_trace_reco{}.png'.format(event.get_run_number(), event.get_id(), suffix))
