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
    print("draw reconstruction")
    print("channel trace operator len ", len(self.__channel_trace_operators))

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
        print("plot channel ", i_channel)
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
            #print(sample)
            for i_pol, efield_stat_calculator in enumerate(efield_stat_calculators):
                print("i_pol ", i_pol)

                channel_sample_trace = self.__channel_trace_operators[i_channel].force(median + sample).val
                trace_stat_calculator.add(channel_sample_trace)
                amp_trace = np.abs(fft.time2freq(channel_sample_trace, sampling_rate))
                amp_trace_stat_calculator.add(amp_trace)
                ax2_1.plot(times, channel_sample_trace * self.__scaling_factor / units.mV, c='k', alpha=.2)
                ax1_1.plot(freqs / units.MHz, amp_trace * self.__scaling_factor / units.mV, c='k', alpha=.2)
                if self.__efield_trace_operators[i_channel][i_pol] is not None:
                    print("self.__efield_trace_operators is not None")
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
            print("has sim_station")
            sim_station = station.get_sim_station()
            n_drawn_sim_channels = 0
            for ray_tracing_id in sim_station.get_ray_tracing_ids():
                sim_channel_sum = None
                for sim_channel in sim_station.get_channels_by_ray_tracing_id(ray_tracing_id):
                    if sim_channel.get_id() == channel_id:
                        print("sim channel == channel id")
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
            channel_snr = .5 * (np.max(station.get_channel(channel_id).get_trace()) - np.min(station.get_channel(channel_id).get_trace())) / (self.__noise_levels[i_channel] * self.__scaling_factor)
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
            ax1_3.set_xlim([self.__passband[0] / units.MHz - 50, self.__passband[1] / units.MHz + 50])
            ax1_3.set_xlabel('f [MHz]')
        if i_channel == 0:
            ax2_1.legend(fontsize=fontsize)
            ax1_1.legend(fontsize=fontsize)
        ax1_1.set_xlim([self.__passband[0] / units.MHz - 50, self.__passband[1] / units.MHz + 50])
        ax1_2.set_xlim([self.__passband[0] / units.MHz - 50, self.__passband[1] / units.MHz + 50])
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
    fig1.savefig('{}/{}_{}_spec_reco_{}_{}_{}.png'.format(self.__plot_folder, event.get_run_number(), event.get_id(), suffix, self.__ray_type, self.__plot_title))
    fig2.tight_layout()
    fig2.savefig('{}/{}_{}_trace_reco_{}_{}_{}.png'.format(self.__plot_folder, event.get_run_number(), event.get_id(), suffix, self.__ray_type, self.__plot_title))
