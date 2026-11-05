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
import NuRadioReco.modules.likelihood_reconstruction.likelihood_calculator
from NuRadioReco.detector.RNO_G import rnog_detector
from NuRadioReco.detector.detector import Detector
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import stationParameters as stp
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.utilities import fft, trace_utilities
import NuRadioReco.modules.likelihood_reconstruction.neutrinoLikelihoodReconstructor
import argparse
import scipy as scp
import glob
from get_noise_spectrum_FT import get_noise_spectrum_from_FT_data

channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelBandPassFilter.begin()
channelCableDelayAdder = NuRadioReco.modules.channelAddCableDelay.channelAddCableDelay()
channelCableDelayAdder.begin()
hardwareResponseIncorporator = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()
hardwareResponseIncorporator.begin()
channelSignalReconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
channelSignalReconstructor.begin()
voltageToEfieldConverter = NuRadioReco.modules.voltageToEfieldConverter.voltageToEfieldConverter()
voltageToEfieldConverter.begin(debug=True)
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
        fig, ax = plt.subplots(len(use_channels), 2, figsize=[12, len(use_channels)*2])
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
                plt.legend(loc=1)
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
            if zoom:
                if signal is None:
                    channel = station.get_channel(use_channels[0])
                    t_array = trace_start_times[0] + np.arange(0, n_samples) * (1 / channel.get_sampling_rate())
                    t_max = t_array[np.argmax(np.abs(channel.get_trace()))]
                for i, i_channel in enumerate(use_channels):
                    if signal is not None:
                        channel = station.get_channel(use_channels[i])
                        t_array = trace_start_times[i] + np.arange(n_samples) * (1 / channel.get_sampling_rate())
                        t_max = t_array[np.argmax(np.abs(abs(signal[i, :])))]
                    plt.sca(ax[i, 0])
                    plt.xlim(t_max - 75, t_max + 75)
                plt.savefig(os.path.join(
                    output_dir,
                    f"Station{station.get_id()}_Run{event.get_run_number()}_Event{event.get_id()}_traces_zoom.png"
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


def get_noise_spectrum_from_traces(filenames, det, use_channels, filter_settings_list, plot=False, mingainlin=0.001, use_first_half=True, n_events_max=1000):

    evtReader = eventReader.eventReader()
    evtReader.begin(filename=filenames, read_detector=True)

    traces_array = []
    n_events = sum(1 for x in evtReader.run())

    for i_event, event in enumerate(evtReader.run()):
        if i_event >= n_events_max:
            break
        print(i_event, "out of", n_events)

        station = event.get_station()
        station.set_is_cosmic_ray()
        #n_channels = station.get_number_of_channels()
        n_samples = station.get_channel(0).get_number_of_samples()
        sampling_rate = station.get_channel(0).get_sampling_rate()
        frequencies = np.fft.rfftfreq(n_samples, 1/sampling_rate)
        n_frequencies = len(frequencies)
        det.update(station.get_station_time())

        # Filters:
        for filt_settings in filter_settings_list:
            channelBandPassFilter.run(event, station, det, **filt_settings)

        # Hardware response:
        hardwareResponseIncorporator.run(event, station, det, sim_to_data=False, mingainlin=mingainlin)

        traces, traces_fft, trace_start_times, max_voltage = get_array_of_channels(station, use_channels, len(use_channels), n_samples, n_frequencies, frequencies, event, output_dir=None)

        traces_array.append(traces)

        #sids[i_event] = station.get_id()

    traces_array = np.array(traces_array)

    # Split into first and second half:
    traces_array_first_half = traces_array[:,:,:n_samples//2]
    traces_array_second_half = traces_array[:,:,n_samples//2:]
    if use_first_half:
        traces_array_combined = np.append(traces_array_first_half, np.roll(traces_array_first_half, 1, axis=0), axis=2) #np.tile(traces_array_first_half, (1, 1, 2))
    else:
        traces_array_combined = np.append(traces_array_second_half, np.roll(traces_array_second_half, 1, axis=0), axis=2)

    # initialize noise model:
    noise_model = NuRadioReco.modules.likelihood_reconstruction.likelihood_calculator.LikelihoodCalculator(n_antennas=len(use_channels), n_samples=n_samples, sampling_rate=sampling_rate, matrix_inversion_method="pseudo_inv", threshold_amplitude=0.1)
    noise_model.initialize_with_data(traces_array_combined)

    spectra = noise_model.spectra

    # Remove artefacts in frequency domain from truncation:
    for filt_settings in filter_settings_list:
        if filt_settings["filter_type"] == "rectangular":
            for i in range(len(use_channels)):
                spectra[i, frequencies > filt_settings["passband"][1]] = 0
                spectra[i, frequencies < filt_settings["passband"][0]] = 0

    if plot:
        fig, ax = plt.subplots(1, len(use_channels), figsize=(len(use_channels)*3, 5))

        for i in range(len(use_channels)):
            plt.sca(ax[i])
            plt.title("Channel " + str(use_channels[i]))
            plt.plot(frequencies, spectra[i], "b-", linewidth=1)
            plt.xlabel("Frequency [GHz]")
            plt.ylabel("Amplitude [mV/GHz]")
            plt.axis([min(frequencies), max(frequencies), 0, np.max(spectra)*1.1])

        plt.tight_layout()
        plt.savefig("spectrum.png")
        plt.close()

    noise_model.plot_llh_distribution(traces_array_combined, np.sum(np.linalg.matrix_rank(noise_model.cov_inv)), frequency_domain=False, make_new_figure=True)
    plt.savefig("llh_distribution_1.png")
    plt.close()

    noise_model.initialize_with_spectra(spectra)
    noise_model.plot_llh_distribution(traces_array_combined, np.sum(np.linalg.matrix_rank(noise_model.cov_inv)), frequency_domain=False, make_new_figure=True)
    plt.savefig("llh_distribution_2.png")
    plt.close()

    # stds = np.std(traces_array_combined, axis=2)
    # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # s = np.unique(sids)
    # for i in range(3):
    #     plt.sca(ax[i])
    #     plt.plot(stds[sids==s[0],i], ".", c="r")
    #     plt.plot(stds[sids==s[1],i], ".", c="y")
    #     plt.plot(stds[sids==s[2],i], ".", c="b")
    #     plt.ylim(0,np.max(stds)*1.1)
    # plt.tight_layout()
    # plt.savefig("stds.png")
    # quit()

    return spectra


import csv
PA_VERTEX_LOOKUP_FILE = "/home/mravn/likelihood-reconstruction/deep_cr_reco/data/pa_vertex_lookup.txt"
def _load_pa_vertex_lookup(table_path=PA_VERTEX_LOOKUP_FILE):
    zenith_by_event = {}
    azimuth_by_event = {}
    x_relative_PA = {}
    y_relative_PA = {}
    z_relative_PA = {}
    with open(table_path, newline="", encoding="utf-8") as table_file:
        reader = csv.DictReader(table_file)
        for row in reader:
            key = (int(row["station"]), int(row["run"]), int(row["event"]))
            zenith_by_event[key] = float(row["zenith_PA_deg"])
            azimuth_by_event[key] = float(row["azimuth_PA_deg"])
            x_relative_PA[key] = float(row["x_PA_m"])
            y_relative_PA[key] = float(row["y_PA_m"])
            z_relative_PA[key] = float(row["z_PA_m"])
    return zenith_by_event, azimuth_by_event, x_relative_PA, y_relative_PA, z_relative_PA

def get_zenith_and_azimuth_from_table(station, run, event, table_path=PA_VERTEX_LOOKUP_FILE):
    """Return the phased-array zenith and azimuth for a station/run/event tuple."""
    zenith_by_event, azimuth_by_event, x_relative_PA, y_relative_PA, z_relative_PA = _load_pa_vertex_lookup(table_path)
    key = (int(station), int(run), int(event))
    try:
        return zenith_by_event[key] * units.deg, azimuth_by_event[key] * units.deg, x_relative_PA[key] * units.m, y_relative_PA[key] * units.m, z_relative_PA[key] * units.m
    except KeyError as exc:
        raise KeyError(
            f"No zenith or azimuth found for station={station}, run={run}, event={event} in {table_path}"
        ) from exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", type=str, nargs="+", help="Input .nur files")
    parser.add_argument("--output_folder", type=str, default="results", help="Save results to folder with this name")
    parser.add_argument("--station_ids", type=int, nargs="+", default=[23], help="Station IDs to process")
    #parser.add_argument("--zenith_azimuth_free", type=int, default=False, help="Free zenith and azimuth angles")
    parser.add_argument("--use_spectrum", type=int, default=False,
                        help="If 0: use flat spectrum. 1: Use flat spectrum + bandpass filters. 2: Use noise spectrum second half of trace. 3: Use noise spectrum from FT data.")
    parser.add_argument("--flow", type=float, default=80, help="Lowest frequency [MHz]")
    parser.add_argument("--fhigh", type=float, default=750, help="Highest frequency [MHz]")
    parser.add_argument("--cont", type=int, default=False, help="If 1, skip events that are already in the output file from previous runs")
    parser.add_argument("--plot_traces", type=int, default=False, help="If 1, plot traces for each event")
    parser.add_argument("--real_data", type=int, default=False, help="If 1, use real data instead of simulation")
    parser.add_argument("--use_alvarez", type=int, default=False, help="If 1, use Alvarez model for zenith angle")


    args = parser.parse_args()
    output_folder = args.output_folder
    #zenith_azimuth_free = False if args.zenith_azimuth_free == 0 else True
    use_spectrum = args.use_spectrum
    flow = args.flow
    fhigh = args.fhigh
    continue_flag = False if args.cont == 0 else True
    plot_traces = False if args.plot_traces == 0 else True
    real_data = False if args.real_data == 0 else True
    use_alvarez = args.use_alvarez

    # Make output:
    output_dir = os.path.join(ABS_PATH_HERE, "./results", f"{output_folder}")
    if not os.path.exists(output_dir):
        print("Making dir", output_dir)
        os.makedirs(output_dir)

    # Dump arguments to file:
    with open(os.path.join(output_dir, "args.txt"), "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")

    # Path to data:
    filenames = args.filenames # glob.glob("./data/lgE17.0_*.nur")

    station_ids = args.station_ids

    evtReader = eventReader.eventReader()
    evtReader.begin(filename=filenames, read_detector=True)
    #det = rnog_detector.Detector(select_stations=station_ids) #, database_time=datetime.datetime(2026, 3, 1, tzinfo=datetime.timezone.utc)) # Not evtReader.get_detector() since rnog_detector is not saved in nur file
    #det = evtReader.get_detector()
    det = Detector(source="rnog_mongo", detector_file="/home/mravn/NuRadioMC/NuRadioReco/modules/likelihood_reconstruction/deep_cr_reco/detector_calibrated_response_season2022_2026_07_11.json.xz")

    n_events = sum(1 for x in evtReader.run())
    evtReader.begin(filename=filenames, read_detector=True)
    for event in evtReader.run():
        n_samples = event.get_station().get_channel(0).get_number_of_samples()
        sampling_rate = event.get_station().get_channel(0).get_sampling_rate()
        frequencies = np.fft.rfftfreq(n_samples, 1/sampling_rate)
        #noise_spectra = np.ones_like(frequencies)  # placeholder, we don't use noise spectra for now
        station = event.get_station()
        det.update(station.get_station_time())

        for ch_id in [0,1,2,3,4,8]:
            channel_info = det.get_channel(station_ids[0], ch_id)
            print(f"Channel {ch_id}: {channel_info['ant_type']}, {channel_info['channel_position']['position']}, {channel_info['channel_position']['orientation']}")
            print(f"Cable delay for Channel {ch_id}: {det.get_cable_delay(station_ids[0], ch_id)}")

        # station = event.get_station()
        # hardwareResponseIncorporator.run(event, station, det, sim_to_data=False, mingainlin=0.001)
        # Vrms = np.std(station.get_channel(0).get_trace()[n_samples//2:])  # estimate noise level from first half of trace
        #print(station.get_channel(0).get_trace())
        break

    filters_llh = None

    # initialize arrays:# load previous results:
    if not continue_flag:
        snr_array = np.zeros([n_events])
        vertex_initial_array = np.zeros([n_events, 3])
        params_array = np.zeros([n_events, 7]) #, 6 if not zenith_azimuth_free else 8])
        uncertainties_array = np.zeros([n_events, 7])
        minus_two_llh_array = np.zeros([n_events])
        p_value_array = np.zeros([n_events])
        energy_array = np.zeros([n_events])
        zenith_sh_array = np.zeros([n_events])
        azimuth_sh_array = np.zeros([n_events])
        vertex_sh_array = np.zeros([n_events, 3])
        vertex_time = np.zeros([n_events])
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

    apply_butterworth_filter = True #True
    if apply_butterworth_filter:
        filter_settings_list = [filt_settings_low, filt_settings_high, filt_settings_rectangular]
    else:
        filter_settings_list = [filt_settings_rectangular]


    if use_spectrum == 0:
        noise_spectra = np.ones_like(frequencies)
    elif use_spectrum == 1:
        noise_spectra = np.ones_like(frequencies)
        filter_1 = channelBandPassFilter.get_filter(frequencies, station.get_id(), 0, det, **filt_settings_low)
        filter_2 = channelBandPassFilter.get_filter(frequencies, station.get_id(), 0, det, **filt_settings_high)
        filter_3 = channelBandPassFilter.get_filter(frequencies, station.get_id(), 0, det, **filt_settings_rectangular)
        noise_spectra = abs(filter_1 * filter_2 * filter_3) * 1.1
    elif use_spectrum == 11:
        noise_spectra = np.ones_like(frequencies)
        filter_3 = channelBandPassFilter.get_filter(frequencies, station.get_id(), 0, det, **filt_settings_rectangular)
        noise_spectra = abs(filter_3) * 1.1
    elif use_spectrum == 2:
        noise_spectra = get_noise_spectrum_from_traces(
            filenames, det, use_channels,
            filter_settings_list = filter_settings_list,
            plot=True, mingainlin=0.001, use_first_half=False, n_events_max=100
        )

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

        use_channels = [0, 1, 2, 3, 9, 10, 22, 23] # 9, 10, 11, 23, 22, 21]
        ref_channel = 2
        if station.get_id() == 13:
            use_channels = [1, 2, 3, 9, 10, 22, 23] #9, 10, 11, 23, 22, 21]
            ref_channel = 2
        n_channels_total = 24
        n_channels = len(use_channels)

        print("i_event:", i_event, "- Station:", station.get_id(), "- Run:", event.get_run_number(), "- Event:", event.get_id())

        # Apply bandpass filters:
        if apply_butterworth_filter:
            channelBandPassFilter.run(event, station, det, **filt_settings_low)
            channelBandPassFilter.run(event, station, det, **filt_settings_high)
        channelBandPassFilter.run(event, station, det, **filt_settings_rectangular)

        # Remove cable delays:
        if not event._has_been_processed_by_module('channelAddCableDelay', station.get_id()):
            channelCableDelayAdder.run(event, station, det, mode='subtract')

        # Hardware response:
        hardwareResponseIncorporator.run(event, station, det, sim_to_data=False, mingainlin=0.001)

        # Get noise RMS for each channel and save max SNR:
        snrs = np.zeros(len(use_channels))
        Vrms = np.array([np.std(station.get_channel(ch_id).get_trace()[n_samples//2:]) for ch_id in use_channels])  # estimate noise level from second half of trace
        for i_ch, ch_id in enumerate(use_channels):
            snrs[i_ch] = trace_utilities.get_signal_to_noise_ratio(station.get_channel(ch_id).get_trace(), Vrms[i_ch])
        snr_array[i_event] = np.max(snrs)

        if use_spectrum == 3:
            n_ft_events, noise_spectra, Vrms_FT, minus_two_llh_dist = get_noise_spectrum_from_FT_data(
                "/mnt/md0/data/RNO-G/inbox/", event, use_channels, det,
                filter_settings_list = filter_settings_list,
                n_min_ft_events=100, plot_llh_dist=True, output_directory=output_dir, mingainlin=0.001, return_traces=False)

            # debug llh calculation:
            from NuRadioReco.modules.likelihood_reconstruction import likelihood_calculator
            spectra_second_half = noise_spectra[:, ::2]
            llh_calculator = likelihood_calculator.LikelihoodCalculator(len(use_channels), station.get_channel(0).get_number_of_samples()//2, station.get_channel(0).get_sampling_rate(), threshold_amplitude=0.1)
            traces_first_half = np.array([station.get_channel(ch_id).get_trace()[:n_samples//2] for ch_id in use_channels])
            traces_second_half = np.array([station.get_channel(ch_id).get_trace()[n_samples//2:] for ch_id in use_channels])
            llh_calculator.initialize_with_spectra(spectra_second_half, np.std(traces_second_half, axis=1))
            llh_first_half = llh_calculator.calculate_minus_two_delta_llh(traces_first_half, np.zeros_like(traces_first_half), frequency_domain=True)
            llh_second_half = llh_calculator.calculate_minus_two_delta_llh(traces_second_half, np.zeros_like(traces_second_half), frequency_domain=True)
            plt.figure()
            plt.hist(minus_two_llh_dist, bins=50)
            plt.axvline(llh_first_half*2, color="r", label="Event LLH from first half of trace")
            plt.axvline(llh_second_half*2, color="b", label="Event LLH from second half of trace")
            plt.xlabel("-2 Delta LLH")
            plt.ylabel("Counts")
            plt.title(f"Station {station.get_id()} - Run {event.get_run_number()} - Event {event.get_id()}")
            plt.legend()
            plt.savefig(os.path.join(output_dir, f"Station{station.get_id()}_Run{event.get_run_number()}_Event{event.get_id()}_llh_distribution.png"))
            plt.close()

        traces, traces_fft, trace_start_times, max_voltage = get_array_of_channels(station, use_channels, n_channels, n_samples, n_frequencies, frequencies, event, output_dir, plot=plot_traces)

        # Get and save true polarization:
        if not real_data:
            sim_station = event.get_station().get_sim_station()
            efields_ch_0 = list(sim_station.get_electric_fields_for_channels([ref_channel]))
            largest_amplitude_efield = None
            max_amplitude_efield = 0
            for i_efield, sim_channel in enumerate(sim_station.get_channels_by_channel_id(ref_channel)):
                if largest_amplitude_efield is None or np.max(sim_channel.get_trace()) > max_amplitude_efield:
                    largest_amplitude_efield = i_efield
                    max_amplitude_efield = np.max(sim_channel.get_trace())
            zenith = efields_ch_0[largest_amplitude_efield][efp.zenith] #+ np.random.normal(0, 5*units.deg)
            azimuth = efields_ch_0[largest_amplitude_efield][efp.azimuth] #+ np.random.uniform(0, 360*units.deg)
            sim_station.set_parameter(stp.zenith, zenith)
            sim_station.set_parameter(stp.azimuth, azimuth)
            polarization = efields_ch_0[largest_amplitude_efield][efp.polarization_angle]
        else:
            # zenith = event.get_station().get_parameter(stp.zenith)
            # azimuth = event.get_station().get_parameter(stp.azimuth)
            zenith_vert, azimuth_vert, x_relative_PA, y_relative_PA, z_relative_PA = get_zenith_and_azimuth_from_table(station.get_id(), event.get_run_number(), event.get_id())
            #z_vert = -5 * units.m
            z_vert_local = det.get_relative_position(station.get_id(), ref_channel)[2] + z_relative_PA
            xy_vert_local = np.tan(zenith_vert) * abs(z_relative_PA) #det.get_relative_position(station.get_id(), ref_channel)[2] - z_vert)
            xyz_ant_local = [0, 0, det.get_relative_position(station.get_id(), ref_channel)[2]]
            r_vertex_local = np.linalg.norm([x_relative_PA, y_relative_PA, z_relative_PA])
            propagator.set_start_and_end_point(np.array([xy_vert_local, 0, z_vert_local]), xyz_ant_local)
            propagator.find_solutions()
            launch_vector = propagator.get_launch_vector(0)
            zenith_launch = np.arccos(launch_vector[2] / np.linalg.norm(launch_vector))
            receive_vector = propagator.get_receive_vector(0)
            zenith = np.arccos(receive_vector[2] / np.linalg.norm(receive_vector))
            azimuth = azimuth_vert
            station.set_parameter(stp.zenith, zenith)
            station.set_parameter(stp.azimuth, azimuth)
            polarization = 0
            print("zenith_vert:", zenith_vert/units.deg, "azimuth_vert:", azimuth_vert/units.deg, "zenith:", zenith/units.deg, "azimuth:", azimuth/units.deg)
            xyz_vertex_initial = np.array([x_relative_PA, y_relative_PA, z_relative_PA]) + np.array(det.get_relative_position(station.get_id(), ref_channel)) + det.get_absolute_position(station.get_id())

        # Determine ray-traced tracvel times to each antenna based on the (true) reconstructed zenith angle (or vertex position?):
        travel_time_shifts = get_travel_time_delays(propagator, det, station.get_id(), use_channels, zenith, azimuth, reference_channel=ref_channel, i_solution=0)

        # Unfolding:
        # if real_data:
        #     channelCableDelayAdder.run(event, station, det, mode='subtract')
        channelSignalReconstructor.run(event, station, det)
        voltageToEfieldConverter.run(event, station, det, use_channels=use_channels, use_MC_direction=False if real_data else True, travel_time_delays=travel_time_shifts)
        times = station.get_channel(ref_channel).get_times()
        signal_search_window = [times[0], times[-1]]
        electricFieldSignalReconstructor.run(event, station, det, signal_search_window=signal_search_window, debug=True, fluence_method="rice") #, theta_phi_rotation=-45 * units.deg) #, theta=zenith, phi=azimuth)
        efield = station.get_electric_fields()[0]

        # Add cable delays again since EfieldToVoltageConverter adds it in efield_reconstructor:
        channelCableDelayAdder.run(event, station, det, mode='add')

        # Likelihood reconstruction:
        # Initialize likelihood reconstructor:
        def detector_simulation_filter_amp(evt, station, det):
            for filt_settings in filter_settings_list:
                channelBandPassFilter.run(evt, station, det, **filt_settings)
        reco = NuRadioReco.modules.likelihood_reconstruction.neutrinoLikelihoodReconstructor.neutrinoLikelihoodReconstructor()
        reco.begin(
            n_channels,
            n_samples,
            sampling_rate,
            noise_spectra,
            Vrms,
            config_file = "../../../examples/likelihood_reconstruction/neutrino_reco_sim_config.yaml",
            detector_simulation_filter_amp = detector_simulation_filter_amp,
            debug = True
        )
        cherenkov_angle = np.arccos(1/ice.get_index_of_refraction([0,0,-5 * units.m]))
        zenith_guess = 180 * units.deg - zenith_launch - cherenkov_angle
        pulse_time_guess = np.argmax(traces[0]) / sampling_rate
        parameters_initial = [
            1 * units.PeV,
            zenith_guess,
            azimuth_vert,
            r_vertex_local,
            zenith_vert,
            azimuth_vert,
            pulse_time_guess
        ]
        print(parameters_initial)
        parameters_fit, uncertainties_fit, signal_fit, minus_two_llh_initial, minus_two_llh_fit, p_value_fit = reco.run(
            event, station, det, parameters_initial, use_channels=use_channels, reference_channel=ref_channel, full_output=True)

        shower_reconstructed = list(event.get_showers())[0]

        # Plot results:
        if plot_traces:
            fig, ax = plt.subplots(len(use_channels), 2, figsize=[len(use_channels)*2, 4])
            for i, i_channel in enumerate(use_channels):
                if i == 0:
                    ax[i, 0].set_title("parameters: " + str(parameters_fit), fontsize=8)
                    ax[i, 0].set_title("shower: " + str(shower_reconstructed), fontsize=8)
                ax[i, 0].set_title("z: " + str(round(zenith,2)) + " a: " + str(round(azimuth,2)), y=1.05, pad=-14, fontsize=8)
                ax[i, 0].plot(station.get_channel(i_channel).get_times(), signal_fit[i, :], "r-", linewidth=1)
                ax[i, 0].set_xlabel("Time [ns]")
                ax[i, 0].set_ylabel("Voltage [mV]")

                ax[i, 1].plot(frequencies, np.abs(fft.time2freq(signal_fit[i, :], station.get_channel(i_channel).get_sampling_rate())), "r-", linewidth=1)
                ax[i, 1].set_xlabel("Frequency [GHz]")
                ax[i, 1].set_ylabel("Ampl. [mV/GHz]")
            plt.savefig(os.path.join(output_dir, f"Station{station.get_id()}_Run{event.get_run_number()}_Event{event.get_id()}_fit.png"))
            plt.close()

        get_array_of_channels(station, use_channels, n_channels, n_samples, n_frequencies, frequencies, event, output_dir, signal=signal_fit, n_channels_total=n_channels_total, plot=plot_traces, zoom=True, zenith=zenith, azimuth=azimuth)

        vertex_initial_array[i_event, :] = xyz_vertex_initial
        params_array[i_event, :] = parameters_fit
        uncertainties_array[i_event, :] = uncertainties_fit
        minus_two_llh_array[i_event] = minus_two_llh_fit
        p_value_array[i_event] = p_value_fit
        energy_array[i_event] = shower_reconstructed.get_parameter(shp.energy)
        zenith_sh_array[i_event] = shower_reconstructed.get_parameter(shp.zenith)
        azimuth_sh_array[i_event] = shower_reconstructed.get_parameter(shp.azimuth)
        vertex_sh_array[i_event, :] = shower_reconstructed.get_parameter(shp.vertex)
        vertex_time[i_event] = shower_reconstructed.get_parameter(shp.vertex_time)

        np.savez(
            os.path.join(output_dir, f"results.npz"),
            snr=snr_array,
            vertex_initial=vertex_initial_array,
            params=params_array,
            uncertainties=uncertainties_array,
            minus_two_llh=minus_two_llh_array,
            p_value=p_value_array,
            energy=energy_array,
            zenith_sh=zenith_sh_array,
            azimuth_sh=azimuth_sh_array,
            vertex_sh=vertex_sh_array,
        )

        if i_event > 20:
            plot_traces = False