"""
This is an advanced versio of neutrino_signal_reconstruction.py. Here, we generate 100 versions of 
the same event with different realizations of noise. The resulting -2 delta LLH distributions should
follow the expected chi-square distributions with 7 degrees of freedom (indicating correct coverage).
Additionally, the fitted parameter distributions are plotted. There may be outliers because the fits
are not initialized exactly at the true values and may fail. The 1-st order uncertainties estimated
using the Fisher-information of the optimum are validated through pull plots. Finally, the p-values
for the fitted signals are evaluated, which should be a uniform distribution between 0 and 1 (exept
for a few outliers).
This script takes a few hours to run.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
import datetime
from radiotools import helper as hp

from NuRadioReco.utilities import units, signal_processing
from NuRadioReco.modules.channelGenericNoiseAdder import channelGenericNoiseAdder
from NuRadioReco.modules.likelihood_reconstruction import shower_simulator, neutrinoLikelihoodReconstructor
from NuRadioReco.framework.event import Event
import NuRadioReco.modules.channelBandPassFilter

channelGenericNoiseAdder = channelGenericNoiseAdder()
channelGenericNoiseAdder.begin()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()

det = NuRadioReco.detector.detector.Detector(json_filename='../../../NuRadioReco/detector/RNO_G/RNO_single_station.json', antenna_by_depth=False)

station_id = 11 #det.get_station_ids()[0]
n_channels_total = det.get_number_of_channels(station_id)
n_samples = det.get_number_of_samples(station_id, 0)
sampling_rate = det.get_sampling_frequency(station_id, 0)
use_channels = [0,1,2,3,4,5,6,7,8,9,10,11,21,22,23] # or [12,13,14,15,16,17,18,19,20] for shallow station
ref_ch = 0 # or 12 for shallow station
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
            station_id = station_id,
            reference_channel = ref_ch,
            evt_time = datetime.datetime(2022, 7, 1),
            use_channels = use_channels,
            detector_simulation_filter_amp = detector_simulation_filter_amp,
            pre_pulse_time = 100 * units.ns
        )

# Simple neutrino event that is likely to give a strong signal in the detector:
E_shower = 200 * units.PeV
zenith = 90 * units.deg
azimuth = 45 * units.deg
vertex_r = 1 * units.km
vertex_zenith = 90 * units.deg + 56 * units.deg # the same as zenith plus Cherenkov angle
vertex_azimuth = 45 * units.deg # the same as azimuth
vertex_xyz = hp.spherical_to_cartesian(vertex_zenith, vertex_azimuth) * vertex_r
vertex_xyz[2] -= 100 * units.m # assuming ~100 m antenna depth. Remove this for shallow station.
vertex_time = 0


n_events = 100
llh_true_array = np.zeros(n_events)
minus_two_llh_initial_array = np.zeros(n_events)
minus_two_llh_fit_array = np.zeros(n_events)
fitted_parameters_array = np.zeros((n_events, 7))
uncertainties_fit_array = np.zeros((n_events, 7))

# Set this to true if the script has already run (partially) and only the plotting is needed:
plots_only = False

for i_event in range(n_events):

    if plots_only:
        # initialize reconstruction module (for plotting) and skip loop:
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
        break

    evt = Event(1, i_event)

    # Simulate the event:
    station, traces, trace_start_times = signal_model.simulate_single_shower(
        energy = E_shower,
        zenith = zenith,
        azimuth = azimuth,
        vertex = vertex_xyz,
        vertex_time = vertex_time,
        type = "HAD",
        charge_excess_profile_id = 5,
        trace_start_times = None # <- Automatically calculates start times based on pulse in reference antenna
    )

    # Add noise to the traces:
    signal_true = np.copy(traces)
    for i_channel, channel in enumerate(station.iter_channels()):
        trace = channel.get_trace()
        trace += channelGenericNoiseAdder.bandlimited_noise_from_spectrum(
            len(trace), channel.get_sampling_rate(), filt, amplitude=noise_amplitude, type='rayleigh')
        channel.set_trace(trace, sampling_rate=channel.get_sampling_rate())
        traces[i_channel] = trace

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
        config_file = "./neutrino_reco_sim_config.yaml",
        detector_simulation_filter_amp = detector_simulation_filter_amp,
        use_chi2 = False, # Set to True to see that using a chi2 gives under-coverage
        debug = True
    )
    minus_two_llh_true = reco._function_to_minimize_llh(traces, signal_true)

    # The reconstructor class uses different parameters (that are better for minimization) than the
    # ones we used to simulate the event. Here we convert the vertex position to the spherical coordinates
    # relative to the reference antenna, and find the pulse time relative to the start of the trace:
    vertex_xyz_rel = vertex_xyz - det.get_relative_position(station_id, ref_ch)
    vertex_zenith_rel, vertex_azimuth_rel = hp.cartesian_to_spherical(vertex_xyz_rel[0], vertex_xyz_rel[1], vertex_xyz_rel[2])
    vertex_r_rel = np.linalg.norm(vertex_xyz_rel)
    pulse_time_guess = np.argmax(traces[use_channels.index(ref_ch)]) / sampling_rate

    # Save true parameters:
    parameters_true = [E_shower, zenith, azimuth, vertex_r_rel, vertex_zenith_rel, vertex_azimuth_rel, 100 * units.ns]

    # Set initial parameters close to the MC true parameters:
    parameters_initial = [
        E_shower * 1.5,
        zenith + 5 * units.deg,
        azimuth - 10 * units.deg,
        vertex_r_rel + 20 * units.m,
        vertex_zenith_rel + 0.5 * units.deg,
        vertex_azimuth_rel - 0.25 * units.deg,
        pulse_time_guess]

    # Run reconstruction:
    parameters_fit, uncertainties_fit, signal_fit, minus_two_llh_initial, minus_two_llh_fit, p_value_fit = reco.run(
        evt, station, det, parameters_initial, use_channels=use_channels, reference_channel=ref_ch, full_output=True)

    print()
    print("-2 LLH for true signal:", minus_two_llh_true)
    print("Initial parameters:", parameters_initial)
    print("Initial -2 LLH:", minus_two_llh_initial)
    print("Fitted parameters:", parameters_fit)
    print("Uncertainties on fitted parameters:", uncertainties_fit)
    print("Fitted -2 LLH:", minus_two_llh_fit)
    print("p-value for fitted signal:", p_value_fit)

    llh_true_array[i_event] = minus_two_llh_true
    minus_two_llh_initial_array[i_event] = minus_two_llh_initial
    minus_two_llh_fit_array[i_event] = minus_two_llh_fit
    fitted_parameters_array[i_event] = parameters_fit
    uncertainties_fit_array[i_event] = uncertainties_fit

    if i_event % 10 == 0 and i_event > 0:
        np.savez(
            "llh_reco_results.npz",
            minus_two_llh_initial_array=minus_two_llh_initial_array,
            minus_two_llh_fit_array=minus_two_llh_fit_array,
            llh_true_array=llh_true_array,
            fitted_parameters_array=fitted_parameters_array,
            uncertainties_fit_array=uncertainties_fit_array,
            parameters_initial=parameters_initial,
            parameters_true=parameters_true
        )

if plots_only:
    data = np.load("llh_reco_results.npz")
    valid_indices = np.where(data["minus_two_llh_fit_array"] > 0)[0]
    minus_two_llh_initial_array = data["minus_two_llh_initial_array"][valid_indices]
    minus_two_llh_fit_array = data["minus_two_llh_fit_array"][valid_indices]
    llh_true_array = data["llh_true_array"][valid_indices]
    fitted_parameters_array = data["fitted_parameters_array"][valid_indices]
    uncertainties_fit_array = data["uncertainties_fit_array"][valid_indices]
    parameters_initial = data["parameters_initial"]
    parameters_true = data["parameters_true"]
fitted_parameters_array[:,2] = fitted_parameters_array[:,2] % (360 * units.deg)
parameters_true[6] = 100 * units.ns

# Plot -2 delta LLH dristribution:
plt.figure(figsize=[10,6])
plt.subplot(2,1,1)
bins = np.linspace(0,50,50)
hist = plt.hist(llh_true_array - minus_two_llh_fit_array, bins=20, alpha=0.5, label="-2 delta LLH")
ndof = 7 # number of fitted parameters
import scipy as scp
dist = scp.stats.chi2(ndof)
x = np.linspace(0,max(bins),1000)
y = dist.pdf(x) * len(minus_two_llh_fit_array) * (hist[1][1] - hist[1][0]) * 1.0
plt.plot(x,y,"y-",label=fr"$\chi^2($dof$={str(ndof)})$")
plt.xlabel("-2 delta LLH")
plt.ylabel("Number of events")
plt.legend()

# Coverage:
plt.subplot(2,1,2)
llh = llh_true_array - minus_two_llh_fit_array
#llh = np.delete(llh, np.where(llh<0)[0])
n_x = 200

x = np.linspace(0,max(bins),n_x)
dist = scp.stats.chi2(ndof)
expected_coverage = dist.cdf(x)

real_coverage = np.zeros(n_x)
for i in range(n_x):
    real_coverage[i] = np.sum(llh<x[i]) / len(llh)

plt.plot([-1,2],[-1,2],"k--",label=f"1:1")
plt.plot(expected_coverage, real_coverage,"b-", label=f"Likelihood")
plt.axis([0,1,0,1])
plt.xlabel("Confidence level")
plt.ylabel("Coverage")
plt.legend()

plt.tight_layout()
plt.savefig("llh_reco_results_coverage.png", dpi=300)


# Plot fitted parameters corner plot:
fig, ax = plt.subplots(7, 7, figsize=[20,20])
parameter_names = ["Energy [eV]", "Zenith [deg]", "Azimuth [deg]", "Vertex r [m]", "Vertex zenith [deg]", "Vertex azimuth [deg]", "Pulse time [ns]"]
scaling = [units.eV, units.deg, units.deg, units.m, units.deg, units.deg, units.ns]
for i in range(7):
    for j in range(7):
        if i == j:
            ax[i,j].hist(fitted_parameters_array[:,i] / scaling[i], bins=20, alpha=0.5)
            ax[i,j].axvline(parameters_initial[i] / scaling[i], color="g", linestyle="--", label="Initial parameter")
            ax[i,j].axvline(parameters_true[i] / scaling[i], color="r", linestyle="-", label="True parameter")
            ax[i,j].set_xlabel(parameter_names[i])
            if i==0 and j==0:
                ax[i,j].legend()
        elif i > j:
            ax[i,j].scatter(fitted_parameters_array[:,j] / scaling[j], fitted_parameters_array[:,i] / scaling[i], alpha=0.5)
            ax[i,j].plot(parameters_initial[j] / scaling[j], parameters_initial[i] / scaling[i], "gx", label="Initial parameters")
            ax[i,j].plot(parameters_true[j] / scaling[j], parameters_true[i] / scaling[i], "r*", label="True parameters")
            ax[i,j].set_xlabel(parameter_names[j])
            ax[i,j].set_ylabel(parameter_names[i])
            if i==1 and j==0:
                ax[i,j].legend()
        else:
            ax[i,j].axis("off")
plt.tight_layout()
plt.savefig("llh_reco_results_fitted_parameters.png", dpi=300)

# Pull distributions:
fig, ax = plt.subplots(7, 1, figsize=[10,20])
for i in range(7):
    pull = (fitted_parameters_array[:,i] - parameters_true[i]) / uncertainties_fit_array[:,i]
    std = np.std(pull)
    quantiles = (np.quantile(pull, 0.84) - np.quantile(pull, 0.16)) / 2
    ax[i].hist(pull, bins=np.linspace(-5,5,50), alpha=0.5, label=f"STD: {std:.2f}, 68% quantile: {quantiles:.2f}") #np.linspace(-3,3,50)
    # ax[i].set_xlabel(f"Pull for {parameter_names[i][:-5]}")
    import re
    pname = re.sub(r' \[.*\]', '', parameter_names[i])
    ax[i].set_xlabel(f"""Pull for {pname}""")
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


# Goodness of fit:
plt.figure(figsize=[10,6])
plt.subplot(2,1,1)
# chi2 distribution:
hist = plt.hist(minus_two_llh_fit_array, bins=20, alpha=0.5, label="Fitted signal -2 LLH")
n_dof_total = reco.likelihood_calculator.get_dof() - 7 # n_channels * n_samples - 7
dist = scp.stats.chi2(n_dof_total)
x = np.linspace(0, max(minus_two_llh_fit_array)*1.2, 1000)
y = dist.pdf(x) * len(minus_two_llh_fit_array) * (hist[1][1] - hist[1][0]) * 1.0
#plt.hist(llh_true_array, bins=20, alpha=0.5, label="True signal")
plt.hist(minus_two_llh_initial_array, bins=20, alpha=0.2, label="Initial parameters -2 LLH")
plt.plot(x, y, "y-", label=rf"$\chi^2($dof$={str(n_dof_total)})$")
plt.xlabel("Chi2 of fit")
plt.ylabel("Number of events")
plt.legend()

# p-value distribution:
plt.subplot(2,1,2)
p_values = 1 - scp.stats.chi2.cdf(minus_two_llh_fit_array, n_dof_total)
p_values_initial = 1 - scp.stats.chi2.cdf(minus_two_llh_initial_array, n_dof_total)
plt.hist(p_values, bins=np.linspace(0,1,20), alpha=0.5, label="Fitted signal")
#plt.hist(p_values_initial, bins=20, alpha=0.2, label="Initial parameters")
plt.xlabel("Goodness-of-fit p-value")
plt.ylabel("Number of events")
#plt.legend()
plt.tight_layout()
plt.savefig("llh_reco_results_goodness_of_fit.png", dpi=300)
