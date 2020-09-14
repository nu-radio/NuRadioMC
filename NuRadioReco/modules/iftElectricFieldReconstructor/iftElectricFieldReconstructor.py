import numpy as np
from NuRadioReco.utilities import units, fft, geometryUtilities
import NuRadioReco.utilities.trace_utilities
import NuRadioReco.detector.antennapattern
import NuRadioReco.detector.RNO_G.analog_components
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import stationParameters as stnp
import NuRadioReco.modules.iftElectricFieldReconstructor.operators
import scipy
import nifty5 as ift
import copy
import matplotlib.pyplot as plt
import scipy.signal




class IftElectricFieldReconstructor:

    def __init__(self):
        self.__antenna_pattern_provider = NuRadioReco.detector.antennapattern.AntennaPatternProvider()
        return

    def begin(
            self,
            passband=None,
            filter_type='butter',
            amp_dct=None,
            phase_dct=None,
            trace_length=128,
            efield_scaling = True,
            debug = False
        ):
        self.__passband = passband
        self.__filter_type = filter_type
        self.__debug = debug
        self.__trace_length = trace_length
        self.__noiseless_spec = {}
        self.__noiseless_traces = None
        self.__efield_scaling = efield_scaling
        self.__n_event = 0
        if amp_dct is None:
            self.__amp_dct = {
                'n_pix': 64,  #spectral bins
                # Spectral smoothness (affects Gaussian process part)
                'a':  .01,
                'k0': 2.,
                # Power-law part of spectrum:
                'sm': -4.9,
                'sv': .5,
                'im':  2.,
                'iv':  .5
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

    def store_noiseless_traces(self, station):
        self.__noiseless_spec = {}
        self.__noiseless_traces = {}
        for channel in station.iter_channels():
            self.__noiseless_spec[str(channel.get_id())] = copy.copy(channel.get_frequency_spectrum())
            self.__noiseless_traces[str(channel.get_id())] = copy.copy(channel.get_trace())
            #plt.close('all')
            #plt.plot(np.abs(channel.get_frequency_spectrum()))
            #plt.show()
            self.__noiseless_freqs = channel.get_frequencies()

    def run(self, event, station, detector, channel_ids, use_sim=False):
        if use_sim:
            self.__vertex_position = station.get_sim_station().get_parameter(stnp.nu_vertex)
        else:
            self.__vertex_position = station.get_parameter(stnp.vertex_2D_fit)
        self.__used_channel_ids = []    #only use channels with associated E-field and zenith
        for channel_id in channel_ids:
            if len(list(station.get_electric_fields_for_channels([channel_id]))) > 0:
                electric_field = list(station.get_electric_fields_for_channels([channel_id]))[0]
                if electric_field.has_parameter(efp.zenith):
                    self.__used_channel_ids.append(channel_id)
        if len(self.__used_channel_ids) == 0:
            return False
        self.__trace_samples = int(station.get_channel(self.__used_channel_ids[0]).get_sampling_rate()*self.__trace_length)
        self.__prepare_traces(event, station, detector)
        ref_channel = station.get_channel(self.__used_channel_ids[0])
        frequencies = ref_channel.get_frequencies()
        sampling_rate = ref_channel.get_sampling_rate()
        time_domain = ift.RGSpace(self.__trace_samples)
        frequency_domain = time_domain.get_default_codomain()
        large_frequency_domain = ift.RGSpace(self.__trace_samples*2, harmonic=True)
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
                                              iteration_limit=1000,
                                              tol_rel_deltaE=1e-9,
                                              convergence_level=3)
        minimizer = ift.NewtonCG(ic_newton)
        median = ift.MultiField.full(H.domain, 0.)
        N_iterations = 5
        N_samples = 15
        min_energy = None
        for k in range(N_iterations):
            print('----------->>>   {}   <<<-----------'.format(k))
            KL = ift.MetricGaussianKL(median, H, N_samples, mirror_samples=True)
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
        self.__n_event += 1
        return True
    def __prepare_traces(self,
        event,
        station,
        det
    ):
        if self.__debug:
            plt.close('all')
            fig1 = plt.figure(figsize=(18,12))
            ax1_1 = fig1.add_subplot(len(self.__used_channel_ids),2,(1,2*len(self.__used_channel_ids)-1))

        n_padding_samples = int(30*station.get_channel(self.__used_channel_ids[0]).get_sampling_rate())

        signal_travel_times = np.zeros(len(self.__used_channel_ids))
        self.__noise_levels = np.zeros(len(self.__used_channel_ids))
        self.__snrs = np.zeros((len(self.__used_channel_ids)))

        trace_max = 0
        for channel_id in self.__used_channel_ids:
            if np.max(np.abs(station.get_channel(channel_id).get_trace())) > trace_max:
                trace_max = np.max(np.abs(station.get_channel(channel_id).get_trace()))
                ref_trace = copy.copy(station.get_channel(channel_id).get_trace())
                ref_trace = np.roll(ref_trace, -np.argmax(np.abs(ref_trace))+n_padding_samples)
                ref_trace[2*n_padding_samples:] = 0
        n_shifts = np.zeros_like(self.__used_channel_ids)
        self.__data_traces = np.zeros((len(self.__used_channel_ids), self.__trace_samples))
        for i_channel, channel_id in enumerate(self.__used_channel_ids):
            trace = station.get_channel(channel_id).get_trace()
            corr= scipy.signal.correlate(trace, ref_trace)
            corr_offset = -(np.arange(0, corr.shape[0]) - corr.shape[0] / 2.)
            n_shift = int(corr_offset[np.argmax(corr)])
            shifted_trace = copy.copy(np.roll(trace, n_shift))
            self.__data_traces[i_channel] = shifted_trace[:self.__trace_samples]
            self.__noise_levels[i_channel] = np.sqrt(np.mean(shifted_trace[self.__trace_samples+1:]**2))
            self.__snrs[i_channel] = .5*(np.max(self.__data_traces[i_channel])-np.min(self.__data_traces[i_channel]))/self.__noise_levels[i_channel]
            n_shifts[i_channel] = n_shift
        self.__scaling_factor = np.max(np.abs(self.__data_traces))
        self.__noise_levels /= self.__scaling_factor
        self.__data_traces /= self.__scaling_factor
        if self.__debug:
            for i_channel, channel_id in enumerate(self.__used_channel_ids):
                times = np.arange(len(self.__data_traces[i_channel]))/station.get_channel(channel_id).get_sampling_rate()
                ax1_1.plot(times, self.__data_traces[i_channel], c='C{}'.format(i_channel), label='Channel {}'.format(channel_id))
                ax1_2 = fig1.add_subplot(len(self.__used_channel_ids),2,2*i_channel+2)
                ax1_2.plot(station.get_channel(channel_id).get_times(), station.get_channel(channel_id).get_trace(), c='C{}'.format(i_channel))
                ax1_2.set_title('Channel {}'.format(channel_id))
                ax1_2.grid()
                #ax1_2.axvline(np.roll(station.get_channel(channel_id).get_times(), n_max-n_padding_samples-n_offsets[i_channel])[self.__trace_samples], c='k', linestyle='--')
                ax1_2.text(.05, .9, 'SNR={:.1f}'.format(self.__snrs[i_channel]), transform=ax1_2.transAxes, fontsize=12, bbox=dict(facecolor='w', alpha=.5))
                if self.__noiseless_traces is not None:
                    ax1_2.plot(station.get_channel(channel_id).get_times(), self.__noiseless_traces[str(channel_id)], c='k', linestyle=':')
                    self.__noiseless_traces[str(channel_id)] = np.roll(self.__noiseless_traces[str(channel_id)], n_shifts[i_channel])[:self.__trace_samples]
                    self.__noiseless_spec[str(channel_id)] = fft.time2freq(self.__noiseless_traces[str(channel_id)], station.get_channel(channel_id).get_sampling_rate())
            ax1_1.grid()
            ax1_1.legend()
            fig1.tight_layout()
            fig1.savefig('traces_{}_{}.png'.format(event.get_run_number(), event.get_id()))

    def __get_detector_operators(self,
        station,
        detector,
        frequency_domain,
        sampling_rate
    ):
        amp_operators = []
        self.__gain_scaling = []
        self.__classic_efield_recos = []
        self.__used_electric_fields = []
        frequencies = frequency_domain.get_k_length_array().val/self.__trace_samples*sampling_rate
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
            receiving_zenith = electric_field.get_parameter(efp.zenith)
            self.__used_electric_fields.append(electric_field)
            if electric_field.has_parameter(efp.azimuth):
                receive_azimuth = electric_field.get_parameter(efp.azimuth)
            else:
                receive_azimuth = 165.*units.deg
            antenna_type = detector.get_antenna_model(station.get_id(), channel_id)
            antenna_pattern = self.__antenna_pattern_provider.load_antenna_pattern(antenna_type)
            antenna_orientation = detector.get_antenna_orientation(station.get_id(), channel_id)
            antenna_response = NuRadioReco.utilities.trace_utilities.get_efield_antenna_factor(station, frequencies, [channel_id], detector, receiving_zenith, receive_azimuth, self.__antenna_pattern_provider)[0][0]
            amp_response_func = NuRadioReco.detector.RNO_G.analog_components.load_amp_response(detector.get_amplifier_type(station.get_id(), channel_id))
            amp_gain = amp_response_func['gain'](frequencies)
            amp_phase = np.unwrap(np.angle(amp_response_func['phase'](frequencies)))
            total_gain =  np.abs(antenna_response)
            total_gain = np.abs(amp_gain) * np.abs(antenna_response)
            max_gain = np.max(total_gain)
            self.__gain_scaling.append(max_gain)
            total_gain /= max_gain
            total_phase = np.unwrap(np.angle(antenna_response))+amp_phase + filter_phase
            total_phase[len(total_phase)//2:] *= -1
            total_phase[len(total_phase)//2+1] = 0
            total_phase *= -1
            amp_field = ift.Field(ift.DomainTuple.make(frequency_domain), total_gain*np.exp(1.j*total_phase))
            amp_operators.append(ift.DiagonalOperator(amp_field))

            #calculate classic E-Field reco
            freqs = np.fft.rfftfreq(self.__data_traces.shape[1], 1./station.get_channel(channel_id).get_sampling_rate())

            antenna_response = antenna_pattern.get_antenna_response_vectorized(
            freqs,
            receiving_zenith,
            receive_azimuth,
            antenna_orientation[0],
            antenna_orientation[1],
            antenna_orientation[2],
            antenna_orientation[3])['theta']
            amp_gain = amp_response_func['gain'](freqs)
            amp_phase = amp_response_func['phase'](freqs)
            self.__classic_efield_recos.append(fft.time2freq(self.__data_traces[i_channel], station.get_channel(channel_id).get_sampling_rate())/antenna_response/amp_gain/amp_phase)
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
        small_sp = ift.RGSpace(large_sp.shape[0]//2, large_sp[0].distances)
        zero_padder = ift.FieldZeroPadder(small_sp, large_sp.shape, central=False)
        domain_flipper = NuRadioReco.modules.iftElectricFieldReconstructor.operators.DomainFlipper(zero_padder.domain, target = ift.RGSpace(small_sp.shape, harmonic=True))
        mag_S_h =  (domain_flipper @ zero_padder.adjoint @ correlated_field)
        mag_S_h = NuRadioReco.modules.iftElectricFieldReconstructor.operators.SymmetrizingOperator(mag_S_h.target) @ mag_S_h
        mag_S_h = realizer2.adjoint @ mag_S_h.exp()
        fft_operator = ift.FFTOperator(frequency_domain.get_default_codomain())

        scaling_domain = ift.UnstructuredDomain(1)
        inserter = NuRadioReco.modules.iftElectricFieldReconstructor.operators.Inserter(realizer.target)
        add_one = ift.Adder(ift.Field(inserter.domain, 1))
        likelihood = None
        self.__efield_trace_operators = []
        self.__efield_spec_operators = []
        self.__channel_trace_operators = []
        self.__channel_spec_operators = []
        for i_channel, channel_id in enumerate(self.__used_channel_ids):
            phi_S_h = NuRadioReco.modules.iftElectricFieldReconstructor.operators.SlopeSpectrumOperator(frequency_domain.get_default_codomain(), self.__phase_dct['sm'], self.__phase_dct['im'], self.__phase_dct['sv'], self.__phase_dct['iv'])
            phi_S_h = realizer2.adjoint @ phi_S_h
            scaling_field = (inserter @ add_one @ (.1 * ift.FieldAdapter(scaling_domain, 'pol{}'.format(i_channel))))
            efield_spec_operator = ((filter_operator @ (mag_S_h * (1.j*phi_S_h).exp())))
            channel_spec_operator = (hardware_operators[i_channel] @ efield_spec_operator)
            if self.__efield_scaling:
                efield_trace_operator = ((realizer @ fft_operator.inverse @ efield_spec_operator)) * scaling_field
                channel_trace_operator = ((realizer @ fft_operator.inverse @ (channel_spec_operator))) * scaling_field
            else:
                efield_trace_operator = ((realizer @ fft_operator.inverse @ efield_spec_operator))
                channel_trace_operator = ((realizer @ fft_operator.inverse @ (channel_spec_operator)))
            noise_operator = ift.ScalingOperator(self.__noise_levels[i_channel]**2, frequency_domain.get_default_codomain())
            data_field = ift.Field(ift.DomainTuple.make(frequency_domain.get_default_codomain()), self.__data_traces[i_channel])
            self.__efield_spec_operators.append(efield_spec_operator)
            self.__efield_trace_operators.append(efield_trace_operator)
            self.__channel_spec_operators.append(channel_spec_operator)
            self.__channel_trace_operators.append(channel_trace_operator)
            if likelihood is None:
                likelihood = ift.GaussianEnergy(mean=data_field, inverse_covariance=noise_operator.inverse)(self.__channel_trace_operators[i_channel])
            else:
                likelihood += ift.GaussianEnergy(mean=data_field, inverse_covariance=noise_operator.inverse)(self.__channel_trace_operators[i_channel])
        return likelihood

    def __store_reconstructed_efields(self,
        event,
        station,
        KL
        ):
        median = KL.position
        for i_channel, channel_id in enumerate(self.__used_channel_ids):
            efield_stat_calculator = ift.StatCalculator()
            for sample in KL.samples:
                efield_sample = self.__efield_trace_operators[i_channel].force(median + sample).val
                efield_stat_calculator.add(efield_sample)
            sampling_rate = station.get_channel(channel_id).get_sampling_rate()
            rec_efield = efield_stat_calculator.mean*self.__scaling_factor/self.__gain_scaling[i_channel]
            trace = np.zeros((3, len(rec_efield)))
            trace[1] = rec_efield
            for efield in station.get_electric_fields_for_channels([channel_id]):
                efield.set_trace(trace, sampling_rate)

    def __draw_priors(
            self,
            event,
            station,
            freq_space
        ):
        plt.close('all')
        fig1 = plt.figure(figsize=(8,8))
        ax1_1 = fig1.add_subplot(221)
        ax1_2 = fig1.add_subplot(222)
        ax1_3 = fig1.add_subplot(223)
        ax1_4 = fig1.add_subplot(224)
        sampling_rate = station.get_channel(self.__used_channel_ids[0]).get_sampling_rate()
        times = np.arange(self.__data_traces.shape[1])/sampling_rate
        freqs = freq_space.get_k_length_array().val/self.__data_traces.shape[1]*sampling_rate
        for i in range(5):
            x = ift.from_random('normal', self.__efield_trace_operators[0].domain)
            efield_spec_sample = self.__efield_spec_operators[0].force(x)
            ax1_1.plot(freqs/units.MHz, np.abs(efield_spec_sample.val))
            efield_trace_sample = self.__efield_trace_operators[0].force(x)
            ax1_2.plot(times, efield_trace_sample.val)
            channel_spec_sample = self.__channel_spec_operators[0].force(x)
            ax1_3.plot(freqs/units.MHz, np.abs(channel_spec_sample.val))
            channel_trace_sample = self.__channel_trace_operators[0].force(x)
            ax1_4.plot(times, channel_trace_sample.val/np.max(np.abs(channel_trace_sample.val)))
        for trace in self.__data_traces:
            ax1_4.plot(times, trace, c='k', alpha=.1)
        ax1_1.grid()
        ax1_1.set_xlim([0,500])
        ax1_2.grid()
        ax1_3.grid()
        ax1_3.set_xlim([0,500])
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

    def __draw_reconstruction(self,
        event,
        station,
        KL
        ):
        plt.close('all')
        n_channels = len(self.__used_channel_ids)
        median = KL.position
        sampling_rate = station.get_channel(self.__used_channel_ids[0]).get_sampling_rate()
        fig1 = plt.figure(figsize=(12, 4*n_channels))
        fig2 = plt.figure(figsize=(16, 4*n_channels))
        times = np.arange(self.__data_traces.shape[1])/sampling_rate
        freqs = np.fft.rfftfreq(self.__data_traces.shape[1], 1./sampling_rate)
        classic_mean_efield_spec = np.zeros_like(freqs)
        for i_channel, channel_id in enumerate(self.__used_channel_ids):
            classic_mean_efield_spec += np.abs(self.__classic_efield_recos[i_channel]) * self.__scaling_factor
        classic_mean_efield_spec /= len(self.__used_channel_ids)
        for i_channel, channel_id in enumerate(self.__used_channel_ids):
            trace_stat_calculator = ift.StatCalculator()
            efield_stat_calculator = ift.StatCalculator()
            amp_trace_stat_calculator = ift.StatCalculator()
            amp_efield_stat_calculator = ift.StatCalculator()
            for sample in KL.samples:
                channel_sample_trace = self.__channel_trace_operators[i_channel].force(median + sample).val
                trace_stat_calculator.add(channel_sample_trace)
                amp_trace = np.abs(fft.time2freq(channel_sample_trace, sampling_rate))
                amp_trace_stat_calculator.add(amp_trace)
                efield_sample_trace = self.__efield_trace_operators[i_channel].force(median + sample).val
                efield_stat_calculator.add(efield_sample_trace)
                amp_efield = np.abs(fft.time2freq(efield_sample_trace, sampling_rate))
                amp_efield_stat_calculator.add(amp_efield)


            channel = station.get_channel(channel_id)
            ax1_1 = fig1.add_subplot(n_channels, 3, 3*i_channel+1)
            ax1_2 = fig1.add_subplot(n_channels, 3, 3*i_channel+2)
            ax1_3 = fig1.add_subplot(n_channels, 3, 3*i_channel+3)
            ax2_1 = fig2.add_subplot(n_channels, 4, (4*i_channel+1, 4*i_channel+2))
            ax2_2 = fig2.add_subplot(n_channels, 4, 4*i_channel+3)
            ax2_3 = fig2.add_subplot(n_channels, 4, 4*i_channel+4)
            ax1_1.plot(freqs/units.MHz, np.abs(fft.time2freq(self.__data_traces[i_channel], sampling_rate))*self.__scaling_factor/units.mV, c='C0', label='data')
            if self.__noiseless_traces is not None:
                ax1_1.plot(freqs/units.MHz, np.abs(self.__noiseless_spec[str(channel_id)])/units.mV, c='C1', label='MC truth')
            ax2_1.plot(times, self.__data_traces[i_channel]*self.__scaling_factor/units.mV, c='C0', alpha=1., zorder=5, label='data')
            ax2_1.plot(times, np.abs(scipy.signal.hilbert(self.__data_traces[i_channel]*self.__scaling_factor))/units.mV, c='C0', alpha=.2, zorder=3)
            #ax2_1.scatter(times, self.__data_traces[i_channel]*self.__scaling_factor/units.mV, c='C0', alpha=.5, zorder=0)

            ax1_1.plot(freqs/units.MHz, amp_trace_stat_calculator.mean*self.__scaling_factor/units.mV, c='C2', label='IFT reco')
            ax2_1.plot(times, trace_stat_calculator.mean*self.__scaling_factor/units.mV, c='C2', linestyle='-', zorder=2, linewidth=4, label='IFT reconstruction')
            ax2_1.plot(times, np.abs(scipy.signal.hilbert(trace_stat_calculator.mean*self.__scaling_factor))/units.mV, c='C2', linestyle='-', zorder=2, linewidth=4, alpha=.5)
            if self.__noiseless_traces is not None:
                ax2_1.plot(times, self.__noiseless_traces[str(channel_id)]/units.mV, c='C1', linewidth=6,zorder=1, label='MC truth')
                ax2_1.plot(times, np.abs(scipy.signal.hilbert(self.__noiseless_traces[str(channel_id)]))/units.mV, c='C1', linewidth=6,zorder=1, alpha=.5)
                ax2_2.plot(times, (self.__data_traces[i_channel]*self.__scaling_factor-self.__noiseless_traces[str(channel_id)])/units.mV, c='C0', label='data')
                ax2_2.plot(times, (trace_stat_calculator.mean*self.__scaling_factor-self.__noiseless_traces[str(channel_id)])/units.mV, c='C2', label='IFT reco')
                ax2_2.plot(times, (np.sum(self.__data_traces, axis=0)*self.__scaling_factor/len(self.__used_channel_ids)-self.__noiseless_traces[str(channel_id)])/units.mV, c='k', linestyle='-', label='classic reco', alpha=.5)
                ax2_3.plot(times, (np.abs(scipy.signal.hilbert(self.__data_traces[i_channel]*self.__scaling_factor))-np.abs(scipy.signal.hilbert(self.__noiseless_traces[str(channel_id)])))/units.mV, c='C0')
                ax2_3.plot(times, (np.abs(scipy.signal.hilbert(trace_stat_calculator.mean*self.__scaling_factor))-np.abs(scipy.signal.hilbert(self.__noiseless_traces[str(channel_id)])))/units.mV, c='C2')
                ax2_3.plot(times, (np.abs(scipy.signal.hilbert(np.sum(self.__data_traces, axis=0)*self.__scaling_factor/len(self.__used_channel_ids)))-np.abs(scipy.signal.hilbert(self.__noiseless_traces[str(channel_id)])))/units.mV, c='k', linestyle='-', alpha=.5)

            #ax2_1.plot(np.arange(len(self.__noiseless_traces[str(channel_id)]/channel.get_sampling_rate())), self.__noiseless_traces[str(channel_id)]/units.mV, c='C1')
            for efield in station.get_sim_station().get_electric_fields_for_channels([channel_id]):
                ax1_2.plot(efield.get_frequencies()/units.MHz, np.abs(efield.get_frequency_spectrum()[1]), c='C1')
            ax1_2.plot(freqs[(freqs>self.__passband[0])&(freqs<self.__passband[1])]/units.MHz, np.abs(self.__classic_efield_recos[i_channel])[(freqs>self.__passband[0])&(freqs<self.__passband[1])]*self.__scaling_factor, c='C0', alpha=.5)
            ax1_2.plot(freqs[(freqs>self.__passband[0])&(freqs<self.__passband[1])]/units.MHz, classic_mean_efield_spec[(freqs>self.__passband[0])&(freqs<self.__passband[1])], c='k', alpha=.5, label='classic reco')
            ax1_2.plot(freqs/units.MHz, amp_efield_stat_calculator.mean*self.__scaling_factor/self.__gain_scaling[i_channel], c='C2')
            ax1_3.plot(freqs[(freqs>self.__passband[0])&(freqs<self.__passband[1])]/units.MHz, (np.unwrap(np.angle(fft.time2freq(trace_stat_calculator.mean, station.get_channel(channel_id).get_sampling_rate())[(freqs>self.__passband[0])&(freqs<self.__passband[1])])))/np.pi, c='C2')
            if self.__noiseless_traces is not None:
                ax1_3.plot(freqs[(freqs>self.__passband[0])&(freqs<self.__passband[1])]/units.MHz, (np.unwrap(np.angle(self.__noiseless_spec[str(channel_id)][(freqs>self.__passband[0])&(freqs<self.__passband[1])])))/np.pi, c='C1')
            ax1_1.axvline(self.__passband[0]/units.MHz, c='k', alpha=.5, linestyle=':')
            ax1_1.axvline(self.__passband[1]/units.MHz, c='k', alpha=.5, linestyle=':')
            ax1_2.axvline(self.__passband[0]/units.MHz, c='k', alpha=.5, linestyle=':')
            ax1_2.axvline(self.__passband[1]/units.MHz, c='k', alpha=.5, linestyle=':')
            ax1_3.axvline(self.__passband[0]/units.MHz, c='k', alpha=.5, linestyle=':')
            ax1_3.axvline(self.__passband[1]/units.MHz, c='k', alpha=.5, linestyle=':')
            ax1_1.grid()
            #ax1_2.set_yscale('log')
            #ax1_2.set_ylim([0, 1.5*np.max(np.abs(amp_efield_stat_calculator.mean*self.__scaling_factor/self.__gain_scaling[i_channel]))])
            ax1_2.grid()
            ax1_3.grid()
            ax2_1.grid()
            ax2_2.grid()
            ax2_3.grid()
            if i_channel == 0:
                ax2_1.legend()
                ax2_2.legend()
                ax1_1.legend()
                ax1_2.legend()
            ax1_1.set_xlim([0,750])
            ax1_2.set_xlim([0,750])
            ax1_3.set_xlim([0,750])
            ax1_1.set_title('Channel {}'.format(channel_id))
            ax2_1.set_title('Channel {}'.format(channel_id))
            ax1_1.set_xlabel('f [MHz]')
            ax1_2.set_xlabel('f [MHz]')
            ax1_3.set_xlabel('f [MHz]')
            ax1_1.set_ylabel('channel voltage [mV/GHz]')
            ax1_2.set_ylabel('E-Field [V/m/GHz]')
            ax1_3.set_ylabel('channel phase /$\pi$')
            ax2_1.set_xlabel('t [ns]')
            ax2_2.set_xlabel('t [ns]')
            ax2_3.set_xlabel('t [ns]')
            ax2_1.set_ylabel('U [mV]')
            ax2_2.set_ylabel('residuals [mV]')
            ax2_3.set_ylabel('residuals of Hilbert envelopes [mV]')
        fig1.tight_layout()
        fig1.savefig('spec_reco_{}_{}.png'.format(event.get_run_number(), event.get_id()))
        fig2.tight_layout()
        fig2.savefig('trace_reco_{}_{}.png'.format(event.get_run_number(), event.get_id()))
