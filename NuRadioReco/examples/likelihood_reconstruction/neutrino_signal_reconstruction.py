"""
This example script demonstrates how to simulate a user-defined a neutrino event with ShowerSimulater and
add band-limited noise to it. The event is then reconstructed using the NeutrinoLikelihoodReconstructor.

In the current implementation, the reconstruction needs to be initialized close to the true
values of the parameters (or close to the global minimum of the likelihood), and we hence 
derive a good initial guess from the true parameters. In a realistic implementation, a good
guess of the vertex zenith, vertex azimuth, and (maybe) vertex r can be found usinginterferometry.
The amplitude (energy) and times are profiled over internally in the reconstruction algorithm.
The rest of the parameters (nu_zenith, nu_azimuth, and maybe r vertex) can then be scanned over
to find the global minimum.
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
            config_file="./neutrino_reco_sim_config.yaml",
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
vertex_zenith_temp = 90 * units.deg + 56 * units.deg # the same as zenith plus Cherenkov angle
vertex_azimuth_temp = 45 * units.deg # the same as azimuth
vertex_xyz = hp.spherical_to_cartesian(vertex_zenith_temp, vertex_azimuth_temp) * vertex_r
vertex_xyz[2] -= 100 * units.m # assuming ~100 m antenna depth
vertex_time = 0

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

# Save true signal:
signal_true = np.copy(traces)

# Add noise to the traces:
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
    use_chi2 = False,
    debug = True
)
minus_two_llh_true = reco._function_to_minimize_llh(traces, signal_true)

# The reconstructor class uses different parameters (that are better for minimization) than the
# ones we used to simulate the event. Here we convert the vertex position to the spherical coordinates
# relative to the reference antenna, and find the pulse time relative to the start of the trace:
vertex_xyz_rel = vertex_xyz - det.get_relative_position(station_id, 0)
vertex_zenith_rel, vertex_azimuth_rel = hp.cartesian_to_spherical(vertex_xyz_rel[0], vertex_xyz_rel[1], vertex_xyz_rel[2])
vertex_r_rel = np.linalg.norm(vertex_xyz_rel)
pulse_time_guess = np.argmax(traces[0]) / sampling_rate

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
    evt, station, det, parameters_initial, use_channels=use_channels, reference_channel=0, full_output=True)

shower_reconstructed = list(evt.get_showers())[0]

# If the fitted -2 LLH is smaller that the -2 LLH of the true signal, the reconstruction
# is likely to have found the global minimum of the likelihood landscape. If the fit failed,
# the p-value will be very close to 0:
print()
print("-2 LLH of true signal:", minus_two_llh_true)
print("Initial -2 LLH:", minus_two_llh_initial)
print("Fitted -2 LLH:", minus_two_llh_fit)
print("p-value for fitted signal:", p_value_fit)

print()
print("True parameters:", [E_shower, zenith, azimuth, vertex_r_rel, vertex_zenith_rel, vertex_azimuth_rel, 100*units.ns])
print("Initial parameters:", parameters_initial)
print("Fitted parameters:", parameters_fit)
print("Uncertainties on fitted parameters:", uncertainties_fit)

print()
print("Reconstructed shower parmaeters (standard NuRadioMC units):")
for i_param, key in enumerate(shower_reconstructed.get_parameters().keys()):
    if shower_reconstructed.has_parameter_error(key):
        print(key, ":", shower_reconstructed.get_parameter(key), "+/-", shower_reconstructed.get_parameter_error(key))
    else:
        print(key, ":", shower_reconstructed.get_parameter(key))