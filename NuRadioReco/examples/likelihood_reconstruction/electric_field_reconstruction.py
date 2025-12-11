import numpy as np
import matplotlib.pyplot as plt
import copy
import datetime

from NuRadioReco.utilities import units, fft, signal_processing, noise_model, trace_minimizer, matched_filter, analytic_pulse, trace_utilities
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.event import Event
from NuRadioReco.framework.station import Station
from NuRadioReco.framework.sim_station import SimStation
from NuRadioReco.framework.electric_field import ElectricField
from NuRadioReco.framework.channel import Channel
from NuRadioReco.detector import detector
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.electricFieldBandPassFilter
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelLengthAdjuster
import NuRadioReco.modules.stationElectricFieldLikelihoodReconstructor


channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
channelGenericNoiseAdder.begin()
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin(debug=False, pre_pulse_time=0, post_pulse_time=0, caching=False)
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelBandPassFilter.begin()
electricFieldBandPassFilter = NuRadioReco.modules.electricFieldBandPassFilter.electricFieldBandPassFilter()
channelLengthAdjuster = NuRadioReco.modules.channelLengthAdjuster.channelLengthAdjuster()


# Set parameters:
det = detector.Detector(json_filename="./dual_LPDA.json", assume_inf=False, antenna_by_depth=False)
det.update(datetime.datetime.now())
station_id = det.get_station_ids()[0]
n_channels = det.get_number_of_channels(station_id)
channel_ids = det.get_channel_ids(station_id)
n_samples = det.get_number_of_samples(station_id, channel_ids[0])
sampling_rate = det.get_sampling_frequency(station_id, channel_ids[0])
t_array = np.arange(n_samples) * 1/sampling_rate
frequencies = np.fft.rfftfreq(n_samples, 1/sampling_rate)
min_freq = 50 * units.MHz
max_freq = 150 * units.MHz
noise_amplitude = 1 * units.muV
filter_settings_low = {'passband': [0 * units.MHz, max_freq],
                            'filter_type': 'butter',
                            'order': 10}
filter_settings_high = {'passband': [min_freq, 1000 * units.MHz],
                            'filter_type': 'butter',
                            'order': 5}



# Make true electric field and data traces:
zenith_arrival = 80 * units.degree
azimuth_arrival = 40 * units.degree
polarization = 65 * units.degree  # polarization angle with respect to theta direction
amplitude = 5000
slope = -3
phase = np.pi / 4
time = 100 * units.ns
efield_trace = analytic_pulse.get_analytic_pulse(
    amp_p0 = amplitude,
    amp_p1 = slope,
    phase_p0 = phase,
    phase_p1 = -time * 2*np.pi,
    n_samples_time = n_samples,
    sampling_rate = sampling_rate,
    bandpass = None
)

electric_field = ElectricField(channel_ids, position=None, shower_id=None, ray_tracing_id=None)
electric_field_theta = np.cos(polarization) * efield_trace
electric_field_phi = np.sin(polarization) * efield_trace
electric_field.set_trace(np.array([np.zeros_like(efield_trace), electric_field_theta, electric_field_phi]), sampling_rate, trace_start_time=0)
electric_field[efp.zenith] = zenith_arrival
electric_field[efp.azimuth] = azimuth_arrival
electric_field[efp.ray_path_type] = "direct"


# Make a copy of the electric field and apply filter:
efield_filtered = copy.copy(electric_field)
sim_station = SimStation(0)
sim_station.add_electric_field(efield_filtered)
evt = Event(1, 1)
electricFieldBandPassFilter.run(evt, sim_station, det=None, **filter_settings_low)
electricFieldBandPassFilter.run(evt, sim_station, det=None, **filter_settings_high)

# Get true (filtered) fluence and polarization:
f_R, f_theta_true, f_phi_true = trace_utilities.get_electric_field_energy_fluence(efield_filtered.get_trace(), efield_filtered.get_times())
polarization_true = np.arctan2(np.sqrt(f_phi_true), np.sqrt(f_theta_true))

print("True fluence (filtered):", round(f_theta_true + f_phi_true, 3), "eV/m^2")
print("True polarization angle (filtered):", round(polarization_true / units.degree, 3), "degree")


# Fold (unfiltered) electric field through detector to get data traces:
sim_station = SimStation(station_id)
sim_station.add_electric_field(electric_field)
sim_station.set_is_cosmic_ray()
sim_station[stnp.zenith] = zenith_arrival
sim_station[stnp.azimuth] = azimuth_arrival

evt = Event(1, 1)
station = NuRadioReco.framework.station.Station(station_id)
station.add_sim_station(sim_station)
efieldToVoltageConverter.run(evt, station, det)

# Add noise and filter:
channelGenericNoiseAdder.run(evt, station, det, min_freq=0*units.GHz, max_freq=max(frequencies), amplitude=noise_amplitude)
channelBandPassFilter.run(evt, station, det, **filter_settings_low)
channelBandPassFilter.run(evt, station, det, **filter_settings_high)


# Reconstruct electric field from data traces:
filter_low = channelBandPassFilter.get_filter(frequencies, station_id, channel_ids[0], det, **filter_settings_low)
filter_high = channelBandPassFilter.get_filter(frequencies, station_id, channel_ids[0], det, **filter_settings_high)
filter = abs(filter_low * filter_high)
reco = NuRadioReco.modules.stationElectricFieldLikelihoodReconstructor.stationElectricFieldLikelihoodReconstructor()
reco.begin(n_channels, n_samples, sampling_rate, filter, noise_amplitude, filter_settings_low, filter_settings_high)
signal_fit = reco.run(evt, station, det, use_MC_direction=True, return_signal=True)

efield_reco = station.get_electric_fields()[0]
print("Reconstructed fluence: (", np.round(efield_reco[efp.signal_energy_fluence], 3), "+/-", np.round(efield_reco.get_parameter_error(efp.signal_energy_fluence), 3), ") eV/m^2")
print("Reconstructed polarization angle: (", np.round(efield_reco[efp.polarization_angle] / units.degree % 180, 3), "+/-", np.round(efield_reco.get_parameter_error(efp.polarization_angle) / units.degree, 3), ") degree")


# Plot data traces and reconstructed signal:
fig, ax = plt.subplots(n_channels, 1, figsize=[8,5])
for i_antenna, channel_id in enumerate(channel_ids):
    channel = station.get_channel(channel_id)
    data_trace = channel.get_trace()
    ax[i_antenna].plot(t_array, data_trace, label="Data (Signal + Noise)", color="b")
    ax[i_antenna].plot(t_array, signal_fit[i_antenna], label="Reconstructed Signal", color='y')
    ax[i_antenna].set_xlim(0, max(t_array))
    ax[i_antenna].set_title(f"Channel {channel_id}")
    if i_antenna == 0: ax[i_antenna].legend()
    if i_antenna == n_channels - 1: ax[i_antenna].set_xlabel("Time [ns]")
    ax[i_antenna].set_ylabel("Voltage [V]")
plt.tight_layout()
plt.show()
