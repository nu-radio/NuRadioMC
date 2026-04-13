
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from radiotools import helper as hp

#import NuRadioReco.detector.RNO_G.rnog_detector
import NuRadioReco.detector.detector
import NuRadioReco.modules.channelBandPassFilter
from NuRadioReco.utilities import units
from NuRadioMC.examples.neutrino_signal_model.neutrino_simulator import NeutrinoSimulator
#import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator

#hardware_response = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()


#det = NuRadioReco.detector.RNO_G.rnog_detector.Detector()
det = NuRadioReco.detector.detector.Detector(json_filename='../../../NuRadioReco/detector/RNO_G/RNO_single_station.json', antenna_by_depth=False)

def detector_simulation_filter_amp(evt, station, det):

    # hardware_response.run(evt, station, det, sim_to_data=True)

    channelBandPassFilter.run(evt, station, det, passband=[80 * units.MHz, 1000 * units.GHz],
                                filter_type='butter', order=2)
    channelBandPassFilter.run(evt, station, det, passband=[0, 500 * units.MHz],
                                filter_type='butter', order=10)

station_id = 11

signal_model = NeutrinoSimulator(
            config_file="../07_RNO_G_simulation/RNO_config.yaml",
            det = det,
            station_id=station_id,
            reference_channel=0,
            evt_time=datetime(2022, 7, 1),
            use_channels=None,
            detector_simulation_filter_amp=detector_simulation_filter_amp,
            pre_pulse_time=100 * units.ns
        )

# Simple neutrino event that is likely to give a strong signal in the detector:
E_shower = 1 * units.EeV
zenith = 90 * units.deg
azimuth = 45 * units.deg
vertex_r = 1 * units.km
vertex_zenith = 90 * units.deg + 56 * units.deg # the same as zenith plus Cherenkov angle
vertex_azimuth = 45 * units.deg # the same as azimuth
vertex_xyz = hp.spherical_to_cartesian(vertex_zenith, vertex_azimuth) * vertex_r
vertex_xyz[2] -= 100 * units.m # assuming ~100 m antenna depth
interaction_time = 0

station, traces, trace_start_times = signal_model.simulate(
    energy=E_shower,
    zenith=zenith,
    azimuth=azimuth,
    vertex=vertex_xyz,
    interaction_time=interaction_time,
    type="HAD",
    charge_excess_profile_id=None,
)


# Plot results:
n_channels = len(traces)
fig, ax = plt.subplots(6, 4, figsize=(20, 12))
ax = ax.flatten()
channel_ids = station.get_channel_ids(station_id)
for i_ch, channel_id in enumerate(channel_ids):
    trace = traces[i_ch]
    times = np.arange(len(trace)) / station.get_channel(channel_id).get_sampling_rate() + trace_start_times[i_ch]
    ax[i_ch].plot(times, trace)
    ax[i_ch].set_title(f"Channel {channel_id}")
    ax[i_ch].set_xlabel("Time [ns]")
    ax[i_ch].set_ylabel("Voltage [V]")
plt.tight_layout()
plt.savefig("simulated_traces.png")
