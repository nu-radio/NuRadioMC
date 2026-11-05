import os
from NuRadioReco.utilities import units, fft
import glob
import numpy as np
import matplotlib.pyplot as plt

from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.likelihood_reconstruction.likelihood_calculator
import NuRadioReco.modules.channelAddCableDelay

channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelBandPassFilter.begin()
channelCableDelayAdder = NuRadioReco.modules.channelAddCableDelay.channelAddCableDelay()
channelCableDelayAdder.begin()
hardwareResponseIncorporator = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()
hardwareResponseIncorporator.begin()


def get_noise_spectrum_from_FT_data(data_directory, event, use_channels, det, filter_settings_list, n_min_ft_events=100, plot_llh_dist=False, output_directory=".", mingainlin=0.001, return_traces=False):

    event_id = event._id
    run_number = event.get_run_number()
    station_id = event.get_station().get_id()

    n_channels = len(use_channels)
    n_samples = event.get_station().get_channel(use_channels[0]).get_number_of_samples()
    sampling_rate = event.get_station().get_channel(use_channels[0]).get_sampling_rate()

    n_ft_events = 0
    traces_all = []
    while len(traces_all) <= n_min_ft_events:
        print("r",run_number)
        run_directory = os.path.join(data_directory, f"station{station_id}/run{run_number}/")

        trigger_selector = lambda eventInfo: eventInfo.triggerType == "FORCE"
        
        data_reader = readRNOGData()
        try:
            data_reader.begin(
                    dirs_files=run_directory,
                    read_calibrated_data=False,
                    select_triggers=None,
                    select_runs=False,
                    apply_baseline_correction='auto',
                    convert_to_voltage=True,
                    selectors=[trigger_selector],
                    run_types=["physics"],
                    run_time_range=None,
                    max_trigger_rate=0 * units.Hz,
                    mattak_kwargs={},
                    overwrite_sampling_rate=None,
                    max_in_mem=256,
                    use_fallback_time=True)
        except:
            run_number += 1
            continue

        for i_event, event in enumerate(data_reader.run()):

            station = event.get_station(station_id)

            # Filters:
            for filt_settings in filter_settings_list:
                channelBandPassFilter.run(event, station, det, **filt_settings)

            # Remove cable delays:
            channelCableDelayAdder.run(event, station, det, mode='subtract')

            # Hardware response:
            hardwareResponseIncorporator.run(event, station, det, sim_to_data=False, mingainlin=mingainlin)

            i_trace = 0
            traces = np.zeros([n_channels, n_samples])
            for i_ch, channel in enumerate(station.iter_channels()):
                if channel.get_id() not in use_channels:
                    continue
                traces[i_trace, :] = channel.get_trace()
                i_trace += 1
            traces_all.append(np.copy(traces))

        n_ft_events = len(traces_all)

        run_number += 1

    traces_all = np.array(traces_all)

    nm = NuRadioReco.modules.likelihood_reconstruction.likelihood_calculator.LikelihoodCalculator(n_channels, n_samples, sampling_rate, matrix_inversion_method="pseudo_inv", threshold_amplitude=0.1, increase_cov_diagonal=0, ignore_llh_normalization=True)
    nm.initialize_with_data(traces_all)

    if plot_llh_dist:
        fig, ax = plt.subplots(1, len(use_channels), figsize=(len(use_channels)*3, 5))

        for i in range(len(use_channels)):
            n_samples = station.get_channel(0).get_number_of_samples()
            sampling_rate = station.get_channel(0).get_sampling_rate()
            frequencies = np.fft.rfftfreq(n_samples, 1/sampling_rate)
            plt.sca(ax[i])
            plt.title("Channel " + str(use_channels[i]))
            plt.plot(frequencies, nm.spectra[i], "b-", linewidth=1)
            plt.xlabel("Frequency [GHz]")
            plt.ylabel("Amplitude [mV/GHz]")
            plt.axis([min(frequencies), max(frequencies), 0, np.max(nm.spectra)*1.1])

        plt.tight_layout()
        plt.savefig("spectrum_FT.png")
        plt.close()

    if plot_llh_dist:
        
        plt.figure()
        for i in range(n_channels):
            plt.plot(nm.frequencies, nm.spectra[i])
        plt.title(f"Noise Spectrum for Event {event_id}")
        plt.xlabel("Frequency [GHz]")
        plt.ylabel("Amplitude [V/GHz]")
        plt.grid()
        plt.savefig(f"{output_directory}/noise_spectrum_event{event_id}.png")
        plt.show()
        plt.close()
        
        n_dof = 0
        for i in range(n_channels):
            n_dof += np.sum(nm.spectra[i]>np.max(nm.spectra[i])*nm.threshold_amplitude)*2
        nm.plot_llh_distribution(traces_all, n_dof=n_dof)
        plt.savefig(f"{output_directory}/llh_distribution_event{event_id}.png")
        plt.show()
        plt.close()

    minus_two_llh_dist = nm.calculate_minus_two_delta_llh(traces_all)

    if not return_traces:
        return n_ft_events, nm.spectra, nm.Vrms, minus_two_llh_dist
    elif return_traces:
        return n_ft_events, nm.spectra, nm.Vrms, minus_two_llh_dist, traces_all
