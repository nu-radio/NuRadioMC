import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
import glob
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as opt
import datetime
#from NuRadioReco.detector import detector
from NuRadioReco.utilities import units
from NuRadioMC.utilities import medium
from NuRadioMC.SignalProp import propagation
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelSignalReconstructor
import NuRadioReco.modules.voltageToEfieldConverter
import NuRadioReco.modules.electricFieldSignalReconstructor
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelAddCableDelay
import NuRadioReco.modules.channelSignalReconstructor
import NuRadioReco.modules.voltageToEfieldConverter
import NuRadioReco.modules.electricFieldSignalReconstructor
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
import NuRadioReco.modules.io.eventReader as eventReader
import NuRadioReco.modules.likelihood_reconstruction.electricFieldLikelihoodReconstructor
from NuRadioReco.detector.RNO_G import rnog_detector
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import stationParameters as stp
from NuRadioReco.utilities import fft, trace_utilities
import argparse
import scipy as scp
import glob

channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelBandPassFilter.begin()
channelCableDelayAdder = NuRadioReco.modules.channelAddCableDelay.channelAddCableDelay()
channelCableDelayAdder.begin()
hardwareResponseIncorporator = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()
hardwareResponseIncorporator.begin()
channelSignalReconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
channelSignalReconstructor.begin()
voltageToEfieldConverter = NuRadioReco.modules.voltageToEfieldConverter.voltageToEfieldConverter()
voltageToEfieldConverter.begin()
electricFieldSignalReconstructor = NuRadioReco.modules.electricFieldSignalReconstructor.electricFieldSignalReconstructor()
electricFieldSignalReconstructor.begin(signal_window_pre=15 * units.ns, signal_window_post=30 * units.ns, noise_window=200 * units.ns,)
efield_reconstructor = NuRadioReco.modules.likelihood_reconstruction.electricFieldLikelihoodReconstructor.electricFieldLikelihoodReconstructor()


ABS_PATH_HERE = str(os.path.dirname(os.path.realpath(__file__)))


def get_array_of_channels(station, use_channels, n_channels, n_samples, n_frequencies, frequencies, event, output_dir=None, signal=None, n_channels_total=24, plot=True, zoom=False, zenith=None, azimuth=None):
    """
    Plots time and frequency traces for upward-facing LPDAs and all channels.

    Args:
        station: NuRadioReco station object.
        use_channels: List of channel indices to use.
        n_channels: Number of channels to plot.
        n_samples: Number of samples per trace.
        n_frequencies: Number of frequency bins.
        frequencies: Frequency array for FFT.
        event: NuRadioReco event object.
        output_dir: Directory to save the plot.
        n_channels_total: Total number of channels in the station (for full array plot).
    """
    traces = np.zeros([n_channels, n_samples])
    traces_fft = np.zeros([n_channels, n_frequencies])
    trace_start_times = np.zeros([n_channels])
    max_voltage = 0
    max_amplitude = 0
    for i, i_channel in enumerate(use_channels):
        channel = station.get_channel(i_channel)
        traces[i, :] = channel.get_trace()
        trace_start_times[i] = channel.get_trace_start_time()
        if max(abs(channel.get_trace())) > max_voltage:
            max_voltage = max(abs(channel.get_trace()))
        traces_fft[i, :] = np.abs(fft.time2freq(channel.get_trace(), channel.get_sampling_rate()))
        if max(abs(traces_fft[i, :])) > max_amplitude:
            max_amplitude = max(abs(traces_fft[i, :]))

    # Plot the traces (Upwardfacing LPDAs)
    if plot:
        fig, ax = plt.subplots(len(use_channels), 2, figsize=[len(use_channels)*2, 4])
        for i, i_channel in enumerate(use_channels):
            channel = station.get_channel(i_channel)

            plt.sca(ax[i, 0])
            plt.plot(channel.get_times(), traces[i, :], "b-", linewidth=1, label="Data")
            if signal is not None:
                plt.plot(channel.get_times(), signal[i, :], "r-", linewidth=1, label="Fit")

            plt.axis([min(trace_start_times), max(trace_start_times + channel.get_number_of_samples()/channel.get_sampling_rate()), -max_voltage * 1.1, max_voltage * 1.1])
            plt.xlim(min(trace_start_times), max(trace_start_times + channel.get_number_of_samples()/channel.get_sampling_rate()))
            # if not zoom:
            #     plt.axis([min(trace_start_times), max(trace_start_times + channel.get_number_of_samples()/channel.get_sampling_rate()), -max_voltage * 1.1, max_voltage * 1.1])
            # else:
            #     plt.axis([min(channel.get_times()), max(channel.get_times()), -max_voltage * 1.1, max_voltage * 1.1])
            if i == len(use_channels)-1:
                plt.xlabel("Time [ns]")
            if i == len(use_channels)//2:
                plt.ylabel("Voltage [V]")
            if i != len(use_channels)-1:
                plt.xticks([])
            if i == 0:
                plt.legend()
            plt.title("Antenna " + str(channel.get_id()), y=1.05, pad=-14, fontsize=9)

            plt.sca(ax[i, 1])
            plt.plot(frequencies, traces_fft[i, :], "b-", linewidth=1, label="Data")
            if signal is not None:
                plt.plot(frequencies, np.abs(fft.time2freq(signal[i, :], channel.get_sampling_rate())), "r-", linewidth=1, label="Fit")
            axis = plt.axis()
            plt.axis([min(frequencies), max(frequencies), 0, max_amplitude*1.1])
            if i == len(use_channels)-1:
                plt.xlabel("Frequency [GHz]")
            if i == len(use_channels)//2:
                plt.ylabel("Ampl. [V/GHz]")
            if i != len(use_channels)-1:
                plt.xticks([])
            plt.title("Antenna " + str(channel.get_id()), y=1.05, pad=-14, fontsize=9)
    
        title = "Station: " + str(station.get_id()) + " - Run: " + str(event.get_run_number()) + " - Event: " + str(event.get_id())
        if zenith is not None and azimuth is not None:
            title += " - Zenith: " + str(round(zenith/units.deg, 1)) + " deg" + " - Azimuth: " + str(round(azimuth/units.deg, 1)) + " deg"
        fig.suptitle(title)
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.30, hspace=0.0)
        if output_dir is not None:
            plt.savefig(os.path.join(
                output_dir,
                f"Station{station.get_id()}_Run{event.get_run_number()}_Event{event.get_id()}_traces.png"
            ))
        plt.close(fig)

        # Plot the traces for all channels
        fig, ax = plt.subplots(6, 4, figsize=[15, 7])
        ax = ax.flatten()
        for i, channel in enumerate(station.iter_channels()):
            plt.sca(ax[i])
            plt.plot(channel.get_times(), channel.get_trace(), "b-", linewidth=1)
            axis = plt.axis()
            #plt.axis([0, max(channel.get_times()), -max_voltage*1.1, max_voltage*1.1])
            if i >= n_channels_total-4:
                plt.xlabel("Time [ns]")
            if i in [0, 4, 8, 12, 16, 20]:
                plt.ylabel("Voltage [mV]")
            #if i == 11: plt.legend(loc=4, fontsize=9)
            if i < n_channels_total-4:
                x_ticks = ax[i].get_xticks()
                ax[i].set_xticks(x_ticks[1:-1], [])
            if i not in [0, 4, 8, 12, 16, 20]:
                ax[i].set_yticks([], [])
            plt.title("Antenna " + str(channel.get_id()), y=1.05, pad=-14, fontsize=9)

        fig.suptitle("Station: " + str(station.get_id()) + " - Run: " + str(event.get_run_number()) + " - Event: " + str(event.get_id()))
        fig.subplots_adjust(wspace=0.00, hspace=0.0)
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.00, hspace=0.0)
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, f"Station{station.get_id()}_Run{event.get_run_number()}_Event{event.get_id()}.png"))
        plt.close(fig)

    return traces, traces_fft, trace_start_times, max_voltage



def ray_trace_to_surface(propagator, z_start, zenith, azimuth, i_solution=0):
    """
    Ray traces from a point in the ice to the surface along a given direction.

    Args:
        propagator: NuRadioMC SignalProp propagation module.
        xyz_start: Starting point of the ray trace (x, y, z) in meters.
        zenith: Zenith angle of the ray trace in radians
    """
    xyz_start = np.array([0, 0, z_start])

    # Define initial guess:
    xy_end_0 = np.tan(zenith) * abs(z_start)

    # Perform ray tracing from the start point to the end point
    def function_to_minimize(params):
        xy_end = params[0]
        xyz_end = np.array([xy_end, 0, -5 * units.m])
        propagator.set_start_and_end_point(xyz_start, xyz_end)
        propagator.find_solutions()
        launch_vector_solution = propagator.get_launch_vector(i_solution)
        zenith_solution = np.arccos(launch_vector_solution[2] / np.linalg.norm(launch_vector_solution))
        return (zenith_solution - zenith)**2
    
    result = opt.minimize(function_to_minimize, x0=xy_end_0)

    xyz_end = np.array([result["x"][0], 0, -5 * units.m])
    xy_solution = result["x"][0]
    # propagator.set_start_and_end_point(xyz_start, xyz_end)
    # propagator.find_solutions()
    # launch_vector_solution = propagator.get_launch_vector(i_solution)
    # zenith_solution = np.arccos(launch_vector_solution[2] / np.linalg.norm(launch_vector_solution))

    return xy_solution  # Return None if we did not reach the surface

def get_travel_time_delays(propagator, det, station_id, use_channels, zenith, azimuth, reference_channel=0, i_solution=0):

    channel_info = det.get_channel(station_id, reference_channel)
    xyz_antenna = np.array(channel_info["channel_position"]["position"])
    z_antenna = xyz_antenna[2]

    xy_solution = ray_trace_to_surface(propagator, z_antenna, zenith, azimuth, i_solution=0)
    x_vertex = np.cos(azimuth) * xy_solution
    y_vertex = np.sin(azimuth) * xy_solution
    z_vertex = -5 * units.m
    xyz_start = np.array([x_vertex, y_vertex, z_vertex])

    travel_times = np.zeros(len(use_channels))
    for i_channel, channel_id in enumerate(use_channels):

        channel_info = det.get_channel(station_id, channel_id)
        xyz_end = np.array(channel_info["channel_position"]["position"])

        propagator.set_start_and_end_point(xyz_start, xyz_end)
        propagator.find_solutions()
        travel_times[i_channel] = propagator.get_travel_time(i_solution)
    
    return travel_times - travel_times[np.where(np.atleast_1d(use_channels) == reference_channel)[0][0]]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", type=str, nargs="+", help="Input .nur files")
    parser.add_argument("output_folder", type=str, default="results", help="Save results to folder with this name")
    parser.add_argument("station_ids", type=int, nargs="+", default=[23], help="Station IDs to process")
    #parser.add_argument("zenith_azimuth_free", type=int, default=False, help="Free zenith and azimuth angles")
    parser.add_argument("use_spectrum", type=int, default=False, help="If 0 use flat spectrum. If 1 use spectrum of first half of trace")
    parser.add_argument("flow", type=float, default=80, help="Lowest frequency [MHz]")
    parser.add_argument("fhigh", type=float, default=750, help="Highest frequency [MHz]")
    parser.add_argument("cont", type=int, default=False, help="If 1, skip events that are already in the output file from previous runs")
    parser.add_argument("--plot_traces", type=int, default=False, help="If 1, plot traces for each event")


    args = parser.parse_args()
    output_folder = args.output_folder
    #zenith_azimuth_free = False if args.zenith_azimuth_free == 0 else True
    use_spectrum = False if args.use_spectrum == 0 else True
    flow = args.flow
    fhigh = args.fhigh
    continue_flag = False if args.cont == 0 else True
    plot_traces = False if args.plot_traces == 0 else True

    # Make output:
    output_dir = os.path.join(ABS_PATH_HERE, "./results", f"{output_folder}")
    if not os.path.exists(output_dir):
        print("Making dir", output_dir)
        os.makedirs(output_dir)
        
    # Path to data:
    filenames = args.filenames # glob.glob("./data/lgE17.0_*.nur")

    station_ids = args.station_ids

    evtReader = eventReader.eventReader()
    evtReader.begin(filename=filenames, read_detector=True)
    det = rnog_detector.Detector(select_stations=station_ids, database_time=datetime.datetime(2026, 3, 1, tzinfo=datetime.timezone.utc)) # Not evtReader.get_detector() since rnog_detector is not saved in nur file

    n_events = sum(1 for x in evtReader.run())
    evtReader.begin(filename=filenames, read_detector=True)
    for event in evtReader.run():
        n_samples = event.get_station().get_channel(0).get_number_of_samples()
        sampling_rate = event.get_station().get_channel(0).get_sampling_rate()
        frequencies = np.fft.rfftfreq(n_samples, 1/sampling_rate)
        noise_spectra = np.ones_like(frequencies)  # placeholder, we don't use noise spectra for now
        # det.update(event.get_station().get_station_time())
        # station = event.get_station()
        # hardwareResponseIncorporator.run(event, station, det, sim_to_data=False, mingainlin=0.001)
        # Vrms = np.std(station.get_channel(0).get_trace()[n_samples//2:])  # estimate noise level from first half of trace
        #print(station.get_channel(0).get_trace())
        break

    use_channels = [0, 1, 2, 3, 4, 8]
    n_channels_total = 24
    n_channels = len(use_channels)

    filters_llh = None

    # initialize arrays:# load previous results:
    if not continue_flag:
        snr_array = np.zeros([n_events])
        polarization_llh_array = np.zeros([n_events])
        fluence_array = np.zeros([n_events])
        zenith_initial_array = np.zeros([n_events])
        azimuth_initial_array = np.zeros([n_events])
        zenith_reco_array = np.zeros([n_events])
        azimuth_reco_array = np.zeros([n_events])
        fluence_uf_array = np.zeros([n_events])  # [Theta, Phi, Total]
        polarization_true_array = np.zeros([n_events])  # [Theta, Phi, Total]
        polarization_ref_array = np.zeros([n_events])  # [Theta, Phi, Total]
        polarization_h1_array = np.zeros([n_events])  # [Theta, Phi, Total]
        polarization_h2_array = np.zeros([n_events])  # [Theta, Phi, Total]
        polarization_uf_array = np.zeros([n_events])  # [Theta, Phi, Total]
        fluence_uf_all_array = np.zeros([n_events])  # [Theta, Phi, Total]
        polarization_uf_error_array = np.zeros([n_events])  # [Theta, Phi, Total]
        polarization_uf_all_array = np.zeros([n_events])  # [Theta, Phi, Total]
        params_array = np.zeros([n_events, 8]) #, 6 if not zenith_azimuth_free else 8])
        llh_array = np.zeros([n_events])
        polarization_error_array = np.zeros([n_events])
        fluence_error_array = np.zeros([n_events])
        p_value_array = np.zeros([n_events])
        n_processed = 0
    # else:
    #     results_file = os.path.join(output_dir, f"results_run_number_{run_number}.npz")
    #     data = np.load(results_file, allow_pickle=True)
    #     polarization_llh_array = data['polarization']
    #     fluence_array = data['fluence']
    #     zenith_initial_array = data['zenith_initial']
    #     azimuth_initial_array = data['azimuth_initial']
    #     zenith_reco_array = data['zenith_reco']
    #     azimuth_reco_array = data['azimuth_reco']
    #     fluence_uf_array = data['fluence_uf']
    #     polarization_uf_array = data['polarization_uf']
    #     fluence_uf_all_array = data['fluence_uf_all']
    #     polarization_uf_all_array = data['polarization_uf_all']
    #     A_theta_array = data['A_theta']
    #     A_phi_array = data['A_phi']
    #     params_array = data['params']
    #     llh_array = data['llh']
    #     polarization_error_array = data['polarization_error']
    #     fluence_error_array = data['fluence_error']
    #     n_processed = sum(polarization_llh_array != 0) - sum(np.isnan(polarization_llh_array))

    filt_settings_low = {'passband': [0 * units.MHz, fhigh * units.MHz],
                            'filter_type': 'butter', #'rectangular', butter
                            'order': 8}
    filt_settings_high = {'passband': [flow * units.MHz, 1000 * units.MHz],
                            'filter_type': 'butter', #'rectangular', butter
                            'order': 3}
    filt_settings_rectangular = {'passband': [75 * units.MHz, 700* units.MHz], 'filter_type': 'rectangular'}

    ice = medium.get_ice_model('greenland_simple')
    propagator = propagation.get_propagation_module("analytic")(ice, attenuation_model="SP1", n_frequencies_integration=25, n_reflections=0) #, detector=det)

    for i_event, event in enumerate(evtReader.run()):
        if i_event < n_processed and continue_flag:
            print("skipped:", i_event, event.get_station().get_id(), event.get_run_number()) #, event_ids[i_event])
            continue
        station = event.get_station()
        station.set_is_cosmic_ray()
        #n_channels = station.get_number_of_channels()
        n_samples = station.get_channel(0).get_number_of_samples()
        sampling_rate = station.get_channel(0).get_sampling_rate()
        frequencies = np.fft.rfftfreq(n_samples, 1/sampling_rate)
        n_frequencies = len(frequencies)
        det.update(station.get_station_time())

        # Flip VPol antenna traces for test dataset:
        # for i_channel in [0,1,2,3]:
        #     trace = station.get_channel(i_channel).get_trace()
        #     trace *= -1
        #     station.get_channel(i_channel).set_trace(trace, sampling_rate)

        print("i_event:", i_event, "- Station:", station.get_id(), "- Run:", event.get_run_number(), "- Event:", event.get_id())

        # Apply bandpass filters:
        channelBandPassFilter.run(event, station, det, **filt_settings_low)
        channelBandPassFilter.run(event, station, det, **filt_settings_high)
        channelBandPassFilter.run(event, station, det, **filt_settings_rectangular)

        # Hardware response:
        hardwareResponseIncorporator.run(event, station, det, sim_to_data=False, mingainlin=0.001)

        # Get noise RMS for each channel and save max SNR:
        snrs = np.zeros(len(use_channels))
        Vrms = np.array([np.std(station.get_channel(i_ch).get_trace()[n_samples//2:]) for i_ch in use_channels])  # estimate noise level from second half of trace
        for i_ch, ch_id in enumerate(use_channels):
            snrs[i_ch] = trace_utilities.get_signal_to_noise_ratio(station.get_channel(i_ch).get_trace(), Vrms[i_ch])
        snr_array[i_event] = np.max(snrs)

        if use_spectrum:
            noise_spectra = np.ones_like(frequencies)
            filter_1 = channelBandPassFilter.get_filter(frequencies, station.get_id(), use_channels[0], det, **filt_settings_low)
            filter_2 = channelBandPassFilter.get_filter(frequencies, station.get_id(), use_channels[0], det, **filt_settings_high)
            filter_3 = channelBandPassFilter.get_filter(frequencies, station.get_id(), use_channels[0], det, **filt_settings_rectangular)
            noise_spectra = abs(filter_1 * filter_2 * filter_3)

        traces, traces_fft, trace_start_times, max_voltage = get_array_of_channels(station, use_channels, n_channels, n_samples, n_frequencies, frequencies, event, output_dir, plot=plot_traces)

        # Get and save true polarization:
        sim_station = event.get_station().get_sim_station()
        efields_ch_0 = list(sim_station.get_electric_fields_for_channels([0]))
        largest_amplitude_efield = None
        max_amplitude_efield = 0
        for i_efield, sim_channel in enumerate(sim_station.get_channels_by_channel_id(0)):
            if largest_amplitude_efield is None or np.max(sim_channel.get_trace()) > max_amplitude_efield:
                largest_amplitude_efield = i_efield
                max_amplitude_efield = np.max(sim_channel.get_trace())
        zenith = efields_ch_0[largest_amplitude_efield][efp.zenith] #+ np.random.normal(0, 5*units.deg)
        azimuth = efields_ch_0[largest_amplitude_efield][efp.azimuth] #+ np.random.uniform(0, 360*units.deg)
        sim_station.set_parameter(stp.zenith, zenith)
        sim_station.set_parameter(stp.azimuth, azimuth)
        polararization = efields_ch_0[largest_amplitude_efield][efp.polarization_angle]

        polarization_true_array[i_event] = polararization
        polarization_ref_array[i_event] = efields_ch_0[1][efp.polarization_angle] if len(efields_ch_0) > 0 else np.nan

        # Determine ray-traced tracvel times to each antenna based on the (true) reconstructed zenith angle (or vertex position?):
        travel_time_shifts = get_travel_time_delays(propagator, det, sim_station.get_id(), use_channels, zenith, azimuth, reference_channel=0, i_solution=0)

        # Unfolding:
        channelSignalReconstructor.run(event, station, det)
        voltageToEfieldConverter.run(event, station, det, use_channels=use_channels, use_MC_direction=True, travel_time_delays=travel_time_shifts)
        times = station.get_channel(use_channels[0]).get_times()
        signal_search_window = [times[0], times[-1]]
        electricFieldSignalReconstructor.run(event, station, det, signal_search_window=signal_search_window, debug=True, fluence_method="rice") #, theta_phi_rotation=-45 * units.deg) #, theta=zenith, phi=azimuth)
        efield = station.get_electric_fields()[0]
        fluence_uf_array[i_event] = sum(efield[efp.signal_energy_fluence])
        polarization_uf_array[i_event] = efield[efp.polarization_angle]
        polarization_uf_error_array[i_event] = efield.get_parameter_error(efp.polarization_angle)

        # Add cable delays again since voltageToEfieldConverter adds it in efield_reconstructor:
        channelCableDelayAdder.run(event, station, det, mode='add')

        # Likelihood reconstruction:
        efield_reconstructor.begin(n_channels, n_samples, sampling_rate, noise_spectra, Vrms, [filt_settings_low, filt_settings_high, filt_settings_rectangular], use_chi2=False, zenith_azimuth_free=False, debug=True, travel_time_shifts=travel_time_shifts) #filt_settings_2
        trace_hilbert_envelope = scp.signal.hilbert(station.get_channel(use_channels[0]).get_trace())
        t_max = np.argmax(trace_hilbert_envelope) / sampling_rate + station.get_channel(use_channels[0]).get_trace_start_time()
        signal_fit, params_fit, minus_two_llh_best, p_value_fit = efield_reconstructor.run(event, station, det, use_channels=use_channels, signal_search_window=[t_max-50-30, t_max+50-30], use_MC_direction=True, full_output=True, second_order=False)

        # Plot results:
        if plot_traces:
            fig, ax = plt.subplots(len(use_channels), 2, figsize=[len(use_channels)*2, 4])
            for i, i_channel in enumerate(use_channels):
                ax[i, 0].set_title("z: " + str(round(zenith,2)) + " a: " + str(round(azimuth,2)), y=1.05, pad=-14, fontsize=8)
                ax[i, 0].plot(station.get_channel(i_channel).get_times(), signal_fit[i, :], "r-", linewidth=1)
                ax[i, 0].set_xlabel("Time [ns]")
                ax[i, 0].set_ylabel("Voltage [mV]")

                ax[i, 1].plot(frequencies, np.abs(fft.time2freq(signal_fit[i, :], station.get_channel(i_channel).get_sampling_rate())), "r-", linewidth=1)
                ax[i, 1].set_xlabel("Frequency [GHz]")
                ax[i, 1].set_ylabel("Ampl. [mV/GHz]")
            plt.savefig(os.path.join(output_dir, f"Station{station.get_id()}_Run{event.get_run_number()}_Event{event.get_id()}_fit.png"))
            plt.close()

        get_array_of_channels(station, use_channels, n_channels, n_samples, n_frequencies, frequencies, event, output_dir, signal=signal_fit, n_channels_total=n_channels_total, plot=plot_traces, zoom=False, zenith=zenith, azimuth=azimuth)

        # Save results:
        efiled_llh = station.get_electric_fields()[1]

        # Plot efield:
        if plot_traces:
            plt.figure(figsize=[15,3])
            plt.title(str("Parameters: ") + str(np.round(params_fit, 2)), y=1.05, pad=-14, fontsize=8)
            plt.plot(efiled_llh.get_times(), efiled_llh.get_trace()[1], "b-", linewidth=1, label="Theta")
            plt.plot(efiled_llh.get_times(), efiled_llh.get_trace()[2], "g--", linewidth=1, label="Phi")
            plt.xlim(min(efiled_llh.get_times()), max(efiled_llh.get_times()))
            plt.legend()
            plt.xlabel("Time [ns]")
            plt.ylabel("Amplitude [V/m]")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"Station{station.get_id()}_Run{event.get_run_number()}_Event{event.get_id()}_efield.png"))
            plt.close()

        polarization_llh_array[i_event] = efiled_llh[efp.polarization_angle]
        fluence_array[i_event] = efiled_llh[efp.signal_energy_fluence]
        zenith_initial_array[i_event] = zenith
        azimuth_initial_array[i_event] = azimuth
        zenith_reco_array[i_event] = efiled_llh[efp.zenith]
        azimuth_reco_array[i_event] = efiled_llh[efp.azimuth]
        params_array[i_event, :] = params_fit
        llh_array[i_event] = minus_two_llh_best
        p_value_array[i_event] = p_value_fit
        polarization_error_array[i_event] = efiled_llh.get_parameter_error(efp.polarization_angle)
        fluence_error_array[i_event] = efiled_llh.get_parameter_error(efp.signal_energy_fluence)

        np.savez(
            os.path.join(output_dir, f"results.npz"),
            polarization_true=polarization_true_array,
            polarization_llh=polarization_llh_array,
            fluence=fluence_array,
            zenith_initial=zenith_initial_array,
            azimuth_initial=azimuth_initial_array,
            zenith_reco=zenith_reco_array,
            azimuth_reco=azimuth_reco_array,
            fluence_uf=fluence_uf_array,
            polarization_uf=polarization_uf_array,
            fluence_uf_all=fluence_uf_all_array,
            polarization_uf_all=polarization_uf_all_array,
            polarization_uf_error=polarization_uf_error_array,
            params=params_array,
            llh=llh_array,
            p_value=p_value_array,
            polarization_error=polarization_error_array,
            fluence_error=fluence_error_array,
            snr=snr_array
        )



        # Plot true polarization distribution:
        plt.figure(figsize=[6, 4])
        plt.hist(polarization_true_array[polarization_true_array!=0]/units.deg, bins=20, range=[-90, 90], histtype='step', linewidth=2, ls='-', label="True polarization")
        #plt.hist(polarization_ref_array/units.deg, bins=20, range=[-90, 90], histtype='step', linewidth=2, ls='--', label="Refracted")
        # plt.hist(polarization_h1_array/units.deg, bins=20, range=[-90, 90], histtype='step', linewidth=2, ls='--', label="H1")
        # plt.hist(polarization_h2_array/units.deg, bins=20, range=[-90, 90], histtype='step', linewidth=2, ls=':', label="H2")
        plt.xlabel("Polarization angle [deg]")
        plt.ylabel("Counts")
        plt.legend()
        plt.title("True polarization distribution")
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"polarization_true_distribution.png"))
        plt.show()
        plt.close()



        # Unfolding results:
        plt.figure(figsize=[6, 4])
        plt.hist(polarization_true_array[polarization_true_array!=0]/units.deg, bins=20, range=[-180, 180], histtype='step', linewidth=2, ls='-', label="True polarization")
        plt.hist(polarization_uf_array[polarization_uf_array!=0]/units.deg, bins=50, range=[-180, 180], histtype='step', linewidth=2, ls='--', label="Unfolded polarization")
        plt.xlabel("Polarization angle [deg]")
        plt.ylabel("Counts")
        plt.legend()
        plt.title("True vs Unfolded polarization distribution")
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"polarization_uf.png"))
        plt.show()
        plt.close()

        plt.figure(figsize=[6, 4])
        plt.scatter(polarization_true_array[polarization_true_array!=0]/units.deg, polarization_uf_array[polarization_true_array!=0]/units.deg, label="Unfolded", alpha=0.5)
        plt.xlabel("Polarization angle 0 [deg]")
        plt.ylabel("Polarization angle unfolded [deg]")
        plt.legend()
        plt.title("True vs Unfolded polarization")
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"polarization_uf_2D.png"))
        plt.show()
        plt.close()


        # LLH reconstruction results:
        plt.figure(figsize=[6, 4])
        plt.hist(polarization_true_array[polarization_true_array!=0]/units.deg, bins=20, range=[-180, 180], histtype='step', linewidth=2, ls='-', label="True polarization")
        plt.hist(polarization_llh_array[polarization_llh_array!=0]/units.deg, bins=50, range=[-180, 180], histtype='step', linewidth=2, ls='--', label="LLH reco polarization")
        plt.xlabel("Polarization angle [deg]")
        plt.ylabel("Counts")
        plt.legend()
        plt.title("True vs LLH reco polarization distribution")
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"polarization_llh.png"))
        plt.show()
        plt.close()

        plt.figure(figsize=[6, 4])
        plt.scatter(polarization_true_array[polarization_true_array!=0]/units.deg, polarization_llh_array[polarization_true_array!=0]/units.deg, label="LLH reco", alpha=0.5)
        plt.xlabel("Polarization angle 0 [deg]")
        plt.ylabel("Polarization angle LLH reco [deg]")
        plt.legend()
        plt.title("True vs LLH reco polarization")
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"polarization_llh_2D.png"))
        plt.show()


        print()
        print(polarization_uf_array[polarization_uf_array!=0])
        print()