import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
import datetime
from radiotools import helper as hp

from NuRadioReco.utilities import units, fft, signal_processing, minimization, matched_filter
from NuRadioReco.modules.channelGenericNoiseAdder import channelGenericNoiseAdder
from NuRadioReco.modules.likelihood_reconstruction import likelihood_calculator, shower_simulator, neutrinoLikelihoodReconstructor
from NuRadioReco.framework.event import Event
import NuRadioReco.modules.channelBandPassFilter

channelGenericNoiseAdder = channelGenericNoiseAdder()
channelGenericNoiseAdder.begin()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()

det = NuRadioReco.detector.detector.Detector(json_filename='../../../NuRadioReco/detector/RNO_G/RNO_single_station.json', antenna_by_depth=False)

evt = Event(1, 1)
station_id = 11 #det.get_station_ids()[0]
n_channels_total = det.get_number_of_channels(station_id)
n_samples = det.get_number_of_samples(station_id, 0)
sampling_rate = det.get_sampling_frequency(station_id, 0)
use_channels = [0,1,2,3,4,5,6,7,8,9,10,11,21,22,23]
n_channels = len(use_channels)

filter_type = "butter"
min_freq = 80 * units.MHz
order_high_pass = 2
max_freq = 500 * units.MHz
order_low_pass = 8
frequencies = np.fft.rfftfreq(n_samples, 1/sampling_rate)
filt = signal_processing.get_filter_response(frequencies, [min_freq, max_freq], "butter", 8)
bandwidth = np.trapz(np.abs(filt) ** 2, frequencies)
noise_amplitude = signal_processing.calculate_vrms_from_temperature(300 * units.kelvin, bandwidth)

filter_settings_low = {'passband': [0 * units.MHz, max_freq],
                            'filter_type': 'butter',
                            'order': 10}
filter_settings_high = {'passband': [min_freq, 1000 * units.MHz],
                            'filter_type': 'butter',
                            'order': 5}

def detector_simulation_filter_amp(evt, station, det):

    channelBandPassFilter.run(evt, station, det, passband=[min_freq, 1000 * units.GHz],
                                filter_type=filter_type, order=order_high_pass)
    channelBandPassFilter.run(evt, station, det, passband=[0, max_freq],
                                filter_type=filter_type, order=order_low_pass)

signal_model = shower_simulator.ShowerSimulator(
            config_file="./neutrino_reco_sim_config.yaml", #"../../../NuRadioMC/examples/07_RNO_G_simulation/RNO_config.yaml",
            det = det,
            station_id=station_id,
            reference_channel=0,
            evt_time=datetime.datetime(2022, 7, 1),
            use_channels=use_channels,
            detector_simulation_filter_amp=detector_simulation_filter_amp,
            pre_pulse_time=100 * units.ns
        )

# Simple neutrino event that is likely to give a strong signal in the detector:
E_shower = 200 * units.PeV
zenith = 90 * units.deg
azimuth = 45 * units.deg
vertex_r = 1 * units.km
vertex_zenith = 90 * units.deg + 56 * units.deg # the same as zenith plus Cherenkov angle
vertex_azimuth = 45 * units.deg # the same as azimuth
vertex_xyz = hp.spherical_to_cartesian(vertex_zenith, vertex_azimuth) * vertex_r
vertex_xyz[2] -= 100 * units.m # assuming ~100 m antenna depth
vertex_time = 0


n_events = 100
llh_initial_array = np.zeros(n_events)
llh_fitted_array = np.zeros(n_events)
fitted_parameters_array = np.zeros((n_events, 7))
uncertainties_fit_array = np.zeros((n_events, 7))

plots_only = False

for i_event in range(n_events):
    if plots_only:
        vertex_zenith, vertex_azimuth = hp.cartesian_to_spherical(vertex_xyz[0], vertex_xyz[1], vertex_xyz[2])
        vertex_r = np.linalg.norm(vertex_xyz)
        parameters_initial = [E_shower, zenith, azimuth, vertex_r, vertex_zenith, vertex_azimuth, vertex_time] # initialize at true parameters
        break

    # Simulate the event:
    station, traces, trace_start_times = signal_model.simulate_single_shower(
        energy=E_shower,
        zenith=zenith,
        azimuth=azimuth,
        vertex=vertex_xyz,
        vertex_time=vertex_time,
        type="HAD",
        charge_excess_profile_id=5,
        trace_start_times=None # <- Automatically calculates start times based on pulse in reference antenna
    )

    # Add noise to the traces:
    for i_channel, channel in enumerate(station.iter_channels()):
        trace = channel.get_trace()
        trace += channelGenericNoiseAdder.bandlimited_noise_from_spectrum(
            len(trace), channel.get_sampling_rate(), filt, amplitude=noise_amplitude, type='rayleigh')
        channel.set_trace(trace, sampling_rate=channel.get_sampling_rate())

    # Plot the traces:
    fig, ax = plt.subplots(n_channels, 1, figsize=[10, 2*n_channels], sharex=True)
    for i_channel, channel in enumerate(station.iter_channels()):
        trace = channel.get_trace()
        time_axis = np.arange(len(trace)) * 1/channel.get_sampling_rate() + channel.get_trace_start_time()
        ax[i_channel].plot(time_axis, trace, label=f"Channel {channel.get_id()}")
        ax[i_channel].legend()
        #ax[i_channel].set_xlim(0, max(time_axis))
        if i_channel == n_channels - 1: ax[i_channel].set_xlabel("Time [ns]")
        ax[i_channel].set_ylabel("Voltage [V]")
    plt.tight_layout()
    plt.savefig("simulated_traces.png", dpi=300)
    plt.close()


    # Initialize likelihood reconstructor:
    reco = neutrinoLikelihoodReconstructor.neutrinoLikelihoodReconstructor()
    reco.begin(
        n_channels,
        n_samples,
        sampling_rate,
        np.abs(filt),
        noise_amplitude,
        config_file="./neutrino_reco_sim_config.yaml",
        detector_simulation_filter_amp=detector_simulation_filter_amp,
        use_chi2=False,
        debug=True
    )
    vertex_zenith, vertex_azimuth = hp.cartesian_to_spherical(vertex_xyz[0], vertex_xyz[1], vertex_xyz[2])
    vertex_r = np.linalg.norm(vertex_xyz)
    pulse_time = np.argmax(traces[0]) / sampling_rate + 6.615 * units.ns
    parameters_initial = [E_shower, zenith, azimuth, vertex_r, vertex_zenith, vertex_azimuth, pulse_time] # initialize at true parameters
    initial_likelihood, fitted_signal, fitted_parameters, minus_two_llh, uncertainties_fit = reco.run(evt, station, det, parameters_initial, use_channels=use_channels, reference_channel=0, full_output=True)


    print()
    print("Initial parameters:", parameters_initial)
    print("Initial likelihood:", initial_likelihood)
    print("Fitted parameters:", fitted_parameters)
    print("Uncertainties on fitted parameters:", uncertainties_fit)
    print("Minus two delta LLH:", minus_two_llh)

    llh_initial_array[i_event] = initial_likelihood
    llh_fitted_array[i_event] = minus_two_llh
    fitted_parameters_array[i_event] = fitted_parameters
    uncertainties_fit_array[i_event] = uncertainties_fit

if not plots_only:
    np.savez("llh_reco_results.npz", llh_initial_array=llh_initial_array, llh_fitted_array=llh_fitted_array, fitted_parameters_array=fitted_parameters_array, uncertainties_fit_array=uncertainties_fit_array)
elif plots_only:
    data = np.load("llh_reco_results.npz")
    llh_initial_array = data["llh_initial_array"]
    llh_fitted_array = data["llh_fitted_array"]
    fitted_parameters_array = data["fitted_parameters_array"]
    uncertainties_fit_array = data["uncertainties_fit_array"]

# plot results:
plt.figure(figsize=[10,6])
plt.subplot(2,1,1)
bins = np.linspace(0,50,50)
hist = plt.hist(llh_initial_array - llh_fitted_array, bins=20, alpha=0.5, label="-2 delta LLH")
ndof = 7 # number of fitted parameters
import scipy as scp
dist = scp.stats.chi2(ndof)
x = np.linspace(0,max(bins),1000)
y = dist.pdf(x) * len(llh_fitted_array) * (hist[1][1] - hist[1][0]) * 1.0
plt.plot(x,y,"y-",label=f"$\chi^2($dof$={str(ndof)})$")
plt.xlabel("-2 delta LLH")
plt.ylabel("Number of events")
plt.legend()

# coverage:
plt.subplot(2,1,2)
llh = llh_initial_array - llh_fitted_array
#llh = np.delete(llh, np.where(llh<0)[0])
n_x = 200

x = np.linspace(0,max(bins),n_x)
dist = scp.stats.chi2(ndof)
expected_coverage = dist.cdf(x)

real_coverage = np.zeros(n_x)
real_coverage_chi2 = np.zeros(n_x)
for i in range(n_x):
    real_coverage[i] = np.sum(llh<x[i]) / len(llh)

plt.plot([-1,2],[-1,2],"k--",label=f"1:1")
plt.plot(expected_coverage,real_coverage,"b-",label=f"Likelihood")
plt.plot(expected_coverage,real_coverage_chi2,"m:",label=f"$\chi^2$")
plt.axis([0,1,0,1])

plt.tight_layout()
plt.savefig("llh_reco_results_coverage.png", dpi=300)


# plot fitted parameters corner plot:
fig, ax = plt.subplots(7, 7, figsize=[20,20])
parameter_names = ["Energy [PeV]", "Zenith [deg]", "Azimuth [deg]", "Vertex r [km]", "Vertex zenith [deg]", "Vertex azimuth [deg]", "Vertex time [ns]"]
for i in range(7):
    for j in range(7):
        if i == j:
            ax[i,j].hist(fitted_parameters_array[:,i], bins=20, alpha=0.5)
            ax[i,j].axvline(parameters_initial[i], color="r", linestyle="--", label="True parameter")
            ax[i,j].set_xlabel(parameter_names[i])
            if i==0 and j==0:
                ax[i,j].legend()
        elif i > j:
            ax[i,j].scatter(fitted_parameters_array[:,j], fitted_parameters_array[:,i], alpha=0.5)
            ax[i,j].plot(parameters_initial[j], parameters_initial[i], "rx", label="True parameters")
            ax[i,j].set_xlabel(parameter_names[j])
            ax[i,j].set_ylabel(parameter_names[i])
            if i==1 and j==0:
                ax[i,j].legend()
        else:
            ax[i,j].axis("off")
plt.tight_layout()
plt.savefig("llh_reco_results_fitted_parameters.png", dpi=300)

# pull distributions:
fig, ax = plt.subplots(7, 1, figsize=[10,20])
for i in range(7):
    pull = (fitted_parameters_array[:,i] - parameters_initial[i]) / uncertainties_fit_array[:,i]
    ax[i].hist(pull, bins=np.linspace(-3,3,50), alpha=0.5, label=f"STD: {np.std(pull):.2f}")
    ax[i].set_xlabel(f"Pull for {parameter_names[i]}")
    ax[i].set_ylabel("Number of events")
    ax[i].legend()
plt.tight_layout()
plt.savefig("llh_reco_results_pull_distributions.png", dpi=300)


plt.figure()
plt.hist(fitted_parameters_array[:,-1], bins=np.linspace(-1,1,50), alpha=0.5)
plt.xlabel("Fitted vertex time [ns]")
plt.ylabel("Number of events")
plt.tight_layout()
plt.savefig("test_1.png", dpi=300)

plt.figure()
plt.hist(fitted_parameters_array[:,3], bins=50, alpha=0.5)
plt.xlabel("Fitted vertex r [m]")
plt.ylabel("Number of events")
plt.tight_layout()
plt.savefig("test_2.png", dpi=300)



print(fitted_parameters_array[:,3])