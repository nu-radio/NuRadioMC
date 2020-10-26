import numpy as np
from NuRadioReco.utilities import units, fft, trace_utilities
import NuRadioReco.utilities.trace_utilities
import NuRadioReco.detector.antennapattern
import NuRadioReco.detector.RNO_G.analog_components
import NuRadioReco.detector.ARIANNA.analog_components
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import channelParameters as chp
import NuRadioReco.modules.iftElectricFieldReconstructor.operators
import NuRadioReco.framework.base_trace
import scipy
import nifty5 as ift
import copy
import matplotlib.pyplot as plt
import scipy.signal
import radiotools.helper


class IftElectricFieldReconstructor:
    def __init__(self):
        self.__antenna_pattern_provider = NuRadioReco.detector.antennapattern.AntennaPatternProvider()
        self.__passband = None
        self.__filter_type = None
        self.__debug = False
        self.__trace_length = None
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
        return

    def begin(
        self,
        electric_field_template,
        passband=None,
        filter_type='butter',
        amp_dct=None,
        phase_dct=None,
        trace_length=128,
        n_iterations=5,
        n_samples=20,
        polarization='theta',
        debug=False
    ):
        self.__passband = passband
        self.__filter_type = filter_type
        self.__debug = debug
        self.__trace_length = trace_length
        self.__n_iterations = n_iterations
        self.__n_samples = n_samples
        self.__trace_samples = len(electric_field_template.get_times())
        self.__polarization = polarization
        self.__electric_field_template = electric_field_template
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
        if phase_dct is None:
            self.__phase_dct = {
                'sm': -2.2,
                'sv': .5,
                'im': 0.,
                'iv': 3.5
            }
        else:
            self.__phase_dct = phase_dct
        return

    def run(self, event, station, detector, channel_ids, efield_scaling):
        self.__used_channel_ids = []    # only use channels with associated E-field and zenith
        self.__efield_scaling = efield_scaling
        self.__used_channel_ids = channel_ids
        # for channel_id in channel_ids:
        #     if len(list(station.get_electric_fields_for_channels([channel_id]))) > 0:
        #         electric_field = list(station.get_electric_fields_for_channels([channel_id]))[0]
        #         if electric_field.has_parameter(efp.zenith):
        #             self.__used_channel_ids.append(channel_id)
        if len(self.__used_channel_ids) == 0:
            return False
        # self.__trace_samples = int(station.get_channel(self.__used_channel_ids[0]).get_sampling_rate() * self.__trace_length)
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
        likelihood = self.__get_likelihood_operator(
            frequency_domain,
            large_frequency_domain,
            amp_operators,
            filter_operator
        )
        if self.__debug:
            self.__draw_priors(event, station, frequency_domain)
        ic_sampling = ift.GradientNormController(1E-8, iteration_limit=min(1000, likelihood.domain.size))
        H = ift.StandardHamiltonian(likelihood, ic_sampling)

        ic_newton = ift.DeltaEnergyController(name='newton',
                                              iteration_limit=100,
                                              tol_rel_deltaE=1e-7,
                                              convergence_level=3)
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
                if self.__debug:
                    self.__draw_reconstruction(
                        event,
                        station,
                        KL
                    )
        self.__store_reconstructed_efields(
            event, station, best_reco_KL
        )
        return True

    def __prepare_traces(
        self,
        event,
        station,
        det
    ):
        if self.__debug:
            plt.close('all')
            fig1 = plt.figure(figsize=(18, 12))
            ax1_1 = fig1.add_subplot(len(self.__used_channel_ids), 2, (1, 2 * len(self.__used_channel_ids) - 1))
            fig2 = plt.figure(figsize=(18, 12))

        n_padding_samples = int(40 * station.get_channel(self.__used_channel_ids[0]).get_sampling_rate())

        self.__noise_levels = np.zeros(len(self.__used_channel_ids))
        self.__snrs = np.zeros((len(self.__used_channel_ids)))

        self.__n_shifts = np.zeros_like(self.__used_channel_ids)
        self.__trace_start_times = np.zeros(len(self.__used_channel_ids))
        self.__data_traces = np.zeros((len(self.__used_channel_ids), self.__trace_samples))
        max_channel_length = 0
        passband = [100. * units.MHz, 300 * units.MHz]
        for channel_id in self.__used_channel_ids:
            channel = station.get_channel(channel_id)
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
            channel_spectrum_template = fft.time2freq(self.__electric_field_template.get_filtered_trace(passband), self.__electric_field_template.get_sampling_rate()) * \
                                        amp_response * (antenna_response['theta'] + antenna_response['phi'])
            channel_trace_template = fft.freq2time(channel_spectrum_template, self.__electric_field_template.get_sampling_rate())
            channel_trace_templates[i_channel] = channel_trace_template
            channel.apply_time_shift(-channel.get_parameter(chp.signal_time_offset), True)
            channel_trace = channel.get_filtered_trace(passband)
            correlation = radiotools.helper.get_normalized_xcorr(channel_trace_template, channel_trace)
            correlation = np.abs(correlation)
            correlation_sum[:len(correlation)] += correlation
            toffset = -(np.arange(0, correlation.shape[0]) - len(channel_trace)) / channel.get_sampling_rate()  # - propagation_times[i_channel, i_solution] - channel.get_trace_start_time()
            ax1_1.plot(toffset, correlation)

        for i_channel, channel_id in enumerate(self.__used_channel_ids):
            channel = station.get_channel(channel_id)
            time_offset = channel.get_parameter(chp.signal_time_offset)
            if self.__debug:
                channel_trace = channel.get_filtered_trace(passband)
                toffset = -(np.arange(0, correlation_sum.shape[0]) - len(channel_trace)) / channel.get_sampling_rate()
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
                    ax2_1.plot(sim_channel_sum.get_times(), sim_channel_sum.get_filtered_trace(passband) / units.mV, c='k', alpha=.5)
                    ax2_1.set_xlim([sim_channel_sum.get_trace_start_time() - 50, sim_channel_sum.get_times()[-1] + 50])
            channel.apply_time_shift(-toffset[np.argmax(correlation_sum)])
            self.__data_traces[i_channel] = channel.get_trace()[:self.__trace_samples]
            self.__noise_levels[i_channel] = np.sqrt(np.mean(channel.get_trace()[self.__trace_samples + 1:]**2))
            self.__snrs[i_channel] = .5 * (np.max(self.__data_traces[i_channel]) - np.min(self.__data_traces[i_channel])) / self.__noise_levels[i_channel]
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
            fig2.savefig('traces_{}_{}.png'.format(event.get_run_number(), event.get_id()))

    def __get_detector_operators(
        self,
        station,
        detector,
        frequency_domain,
        sampling_rate
    ):
        amp_operators = []
        self.__gain_scaling = []
        self.__classic_efield_recos = []
        self.__used_electric_fields = []
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
            electric_field = list(station.get_electric_fields_for_channels([channel_id]))[0]
            channel = station.get_channel(channel_id)
            receiving_zenith = channel.get_parameter(chp.signal_receiving_zenith)
            self.__used_electric_fields.append(electric_field)
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

            # calculate classic E-Field reco
            freqs = np.fft.rfftfreq(self.__data_traces.shape[1], 1. / station.get_channel(channel_id).get_sampling_rate())

            electric_field = list(station.get_electric_fields_for_channels([channel_id]))[0]
            receiving_zenith = station.get_channel(channel_id).get_parameter(chp.signal_receiving_zenith)
            receive_azimuth = 0.
            antenna_type = detector.get_antenna_model(station.get_id(), channel_id)
            antenna_pattern = self.__antenna_pattern_provider.load_antenna_pattern(antenna_type)
            antenna_orientation = detector.get_antenna_orientation(station.get_id(), channel_id)
            antenna_response = antenna_pattern.get_antenna_response_vectorized(
                freqs,
                receiving_zenith,
                receive_azimuth,
                antenna_orientation[0],
                antenna_orientation[1],
                antenna_orientation[2],
                antenna_orientation[3]
            )['theta']
            # amp_response = detector.get_amplifier_response(station.get_id(), channel_id, station.get_channel(channel_id).get_frequencies())
            # amp_gain = np.abs(amp_response)
            # amp_phase = np.unwrap(np.angle(amp_response))
            # self.__classic_efield_recos.append(fft.time2freq(self.__data_traces[i_channel], station.get_channel(channel_id).get_sampling_rate()) / antenna_response / amp_gain / amp_phase)
        return amp_operators, filter_operator

    def __get_likelihood_operator(
        self,
        frequency_domain,
        large_frequency_domain,
        hardware_operators,
        filter_operator
    ):
        power_domain = ift.RGSpace(large_frequency_domain.get_default_codomain().shape[0], harmonic=True)
        power_space = ift.PowerSpace(power_domain)
        self.__amp_dct['target'] = power_space
        A = ift.SLAmplitude(**self.__amp_dct)
        correlated_field = ift.CorrelatedField(large_frequency_domain.get_default_codomain(), A)
        realizer = ift.Realizer(self.__fft_operator.domain)
        realizer2 = ift.Realizer(self.__fft_operator.target)
        large_sp = correlated_field.target
        small_sp = ift.RGSpace(large_sp.shape[0] // 2, large_sp[0].distances)
        zero_padder = ift.FieldZeroPadder(small_sp, large_sp.shape, central=False)
        domain_flipper = NuRadioReco.modules.iftElectricFieldReconstructor.operators.DomainFlipper(zero_padder.domain, target=ift.RGSpace(small_sp.shape, harmonic=True))
        mag_S_h = (domain_flipper @ zero_padder.adjoint @ correlated_field)
        mag_S_h = NuRadioReco.modules.iftElectricFieldReconstructor.operators.SymmetrizingOperator(mag_S_h.target) @ mag_S_h
        mag_S_h = realizer2.adjoint @ mag_S_h.exp()
        fft_operator = ift.FFTOperator(frequency_domain.get_default_codomain())

        scaling_domain = ift.UnstructuredDomain(1)
        inserter = NuRadioReco.modules.iftElectricFieldReconstructor.operators.Inserter(realizer.target)
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
        median = KL.position
        for i_channel, channel_id in enumerate(self.__used_channel_ids):
            efield_stat_calculators = [ift.StatCalculator(), ift.StatCalculator()]
            polarization_stat_calculator = ift.StatCalculator()
            rec_efield = np.zeros((3, self.__electric_field_template.get_number_of_samples()))
            sampling_rate = self.__electric_field_template.get_sampling_rate()
            times = np.arange(self.__data_traces.shape[1]) / sampling_rate
            for sample in KL.samples:
                if self.__efield_trace_operators[i_channel][0] is not None:
                    efield_sample_theta = self.__efield_trace_operators[i_channel][0].force(median + sample).val
                    efield_stat_calculators[0].add(efield_sample_theta)
                if self.__efield_trace_operators[i_channel][1] is not None:
                    efield_sample_phi = self.__efield_trace_operators[i_channel][1].force(median + sample).val
                    efield_stat_calculators[1].add(efield_sample_phi)
                if self.__polarization == 'pol':
                    efield_sample_pol = np.zeros_like(rec_efield)
                    efield_sample_pol[1] = efield_sample_theta
                    efield_sample_pol[2] = efield_sample_phi
                    energy_fluences = trace_utilities.get_electric_field_energy_fluence(
                        efield_sample_pol,
                        times
                    )
                    polarization_stat_calculator.add(np.arctan(energy_fluences[2] / energy_fluences[1]))
            if self.__efield_trace_operators[i_channel][0] is not None:
                rec_efield[1] = efield_stat_calculators[0].mean * self.__scaling_factor / self.__gain_scaling
            if self.__efield_trace_operators[i_channel][1] is not None:
                rec_efield[2] = efield_stat_calculators[1].mean * self.__scaling_factor / self.__gain_scaling
            for efield in station.get_electric_fields_for_channels([channel_id]):
                efield.set_trace(rec_efield, sampling_rate)
                efield.set_trace_start_time(self.__trace_start_times[i_channel])
                if self.__polarization == 'pol':
                    efield.set_parameter(efp.polarization_angle, polarization_stat_calculator.mean)
                    efield.set_parameter_error(efp.polarization_angle, polarization_stat_calculator.var)
                for sim_efield in station.get_sim_station().get_electric_fields_for_channels([channel_id]):
                    sim_energy_fluence = trace_utilities.get_electric_field_energy_fluence(sim_efield.get_trace(), sim_efield.get_times())
                    sim_polarization = np.arctan(sim_energy_fluence[2] / sim_energy_fluence[1])
                break

    def __draw_priors(
        self,
        event,
        station,
        freq_space
    ):
        plt.close('all')
        fig1 = plt.figure(figsize=(8, 8))
        ax1_1 = fig1.add_subplot(221)
        ax1_2 = fig1.add_subplot(222)
        ax1_3 = fig1.add_subplot(223)
        ax1_4 = fig1.add_subplot(224)
        sampling_rate = station.get_channel(self.__used_channel_ids[0]).get_sampling_rate()
        times = np.arange(self.__data_traces.shape[1]) / sampling_rate
        freqs = freq_space.get_k_length_array().val / self.__data_traces.shape[1] * sampling_rate
        for i in range(5):
            x = ift.from_random('normal', self.__efield_trace_operators[0][0].domain)
            efield_spec_sample = self.__efield_spec_operators[0][0].force(x)
            ax1_1.plot(freqs / units.MHz, np.abs(efield_spec_sample.val))
            efield_trace_sample = self.__efield_trace_operators[0][0].force(x)
            ax1_2.plot(times, efield_trace_sample.val)
            channel_spec_sample = self.__channel_spec_operators[0].force(x)
            ax1_3.plot(freqs / units.MHz, np.abs(channel_spec_sample.val))
            channel_trace_sample = self.__channel_trace_operators[0].force(x)
            ax1_4.plot(times, channel_trace_sample.val / np.max(np.abs(channel_trace_sample.val)))
        for trace in self.__data_traces:
            ax1_4.plot(times, trace, c='k', alpha=.1)
        ax1_1.grid()
        ax1_1.set_xlim([0, 500])
        ax1_2.grid()
        ax1_3.grid()
        ax1_3.set_xlim([0, 500])
        ax1_4.grid()
        ax1_1.set_xlabel('f [MHz]')
        ax1_2.set_xlabel('t [ns]')
        ax1_3.set_xlabel('f [MHz]')
        ax1_4.set_xlabel('t [ns]')
        ax1_1.set_ylabel('E [a.u.]')
        ax1_2.set_ylabel('E [a.u.]')
        ax1_3.set_ylabel('U [a.u.]')
        ax1_4.set_ylabel('U [a.u.]')
        ax1_1.set_title('E-Field Spectrum')
        ax1_2.set_title('E-Field Trace')
        ax1_3.set_title('Channel Spectrum')
        ax1_4.set_title('Channel Trace')
        fig1.tight_layout()
        fig1.savefig('priors_{}.png'.format(event.get_id()))

    def __draw_reconstruction(
        self,
        event,
        station,
        KL
    ):
        plt.close('all')
        n_channels = len(self.__used_channel_ids)
        median = KL.position
        sampling_rate = station.get_channel(self.__used_channel_ids[0]).get_sampling_rate()
        fig1 = plt.figure(figsize=(12, 4 * n_channels))
        fig2 = plt.figure(figsize=(16, 4 * n_channels))
        freqs = np.fft.rfftfreq(self.__data_traces.shape[1], 1. / sampling_rate)
        classic_mean_efield_spec = np.zeros_like(freqs)
        # for i_channel, channel_id in enumerate(self.__used_channel_ids):
        #     classic_mean_efield_spec += np.abs(self.__classic_efield_recos[i_channel])
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
                ax1_2.set_title(r'$\theta$ component')
                ax1_3.set_title(r'$\varphi$ component')
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

            ax2_1.plot(times, self.__data_traces[i_channel] * self.__scaling_factor / units.mV, c='C0', alpha=1., zorder=5, label='data')
            ax2_1.plot(times, np.abs(scipy.signal.hilbert(self.__data_traces[i_channel] * self.__scaling_factor)) / units.mV, c='C0', alpha=.2, zorder=3)

            ax1_1.plot(freqs / units.MHz, amp_trace_stat_calculator.mean * self.__scaling_factor / units.mV, c='C2', label='IFT reco', linewidth=3)
            ax2_1.plot(times, trace_stat_calculator.mean * self.__scaling_factor / units.mV, c='C2', linestyle='-', zorder=2, linewidth=4, label='IFT reconstruction')
            ax2_1.plot(times, np.abs(scipy.signal.hilbert(trace_stat_calculator.mean * self.__scaling_factor)) / units.mV, c='C2', linestyle='-', zorder=2, linewidth=4, alpha=.5)
            ax2_1.set_xlim([times[0], times[-1]])
            if self.__polarization == 'theta':
                ax1_2.plot(freqs / units.MHz, amp_efield_stat_calculators[0].mean * self.__scaling_factor / self.__gain_scaling / (units.mV / units.m / units.GHz), c='C2')
            if self.__polarization == 'phi':
                ax1_2.plot(freqs / units.MHz, amp_efield_stat_calculators[1].mean * self.__scaling_factor / self.__gain_scaling / (units.mV / units.m / units.GHz), c='C2')
            if self.__polarization == 'pol':
                ax1_2.plot(freqs / units.MHz, np.abs(fft.time2freq(efield_stat_calculators[0].mean, sampling_rate)) * self.__scaling_factor / self.__gain_scaling / (units.mV / units.m / units.GHz), c='C2')
                ax1_3.plot(freqs / units.MHz, np.abs(fft.time2freq(efield_stat_calculators[1].mean, sampling_rate)) * self.__scaling_factor / self.__gain_scaling / (units.mV / units.m / units.GHz), c='C2')
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
                ax2_1.legend()
                ax1_1.legend()
            ax1_1.set_xlim([0, 750])
            ax1_2.set_xlim([0, 750])
            ax1_1.set_title('Channel {}'.format(channel_id))
            ax2_1.set_title('Channel {}'.format(channel_id))
            ax1_1.set_xlabel('f [MHz]')
            ax1_2.set_xlabel('f [MHz]')
            ax1_1.set_ylabel('channel voltage [mV/GHz]')
            ax1_2.set_ylabel('E-Field [mV/m/GHz]')
            ax2_1.set_xlabel('t [ns]')
            ax2_1.set_ylabel('U [mV]')
        fig1.tight_layout()
        fig1.savefig('spec_reco_{}_{}.png'.format(event.get_run_number(), event.get_id()))
        fig2.tight_layout()
        fig2.savefig('trace_reco_{}_{}.png'.format(event.get_run_number(), event.get_id()))
