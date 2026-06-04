import os
import numpy as np
import matplotlib.pyplot as plt
from NuRadioMC.simulation import simulation as sim
from NuRadioReco.detector import detector
from NuRadioMC.SignalProp import propagation
from NuRadioMC.utilities import medium
import NuRadioReco.framework.radio_shower
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelReadoutWindowCutter
import NuRadioReco.modules.trigger.simpleThreshold
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.utilities import units
from datetime import datetime
import logging
logger = logging.getLogger('calculate_channel')

"""
This script is an example of how to calculate the observed voltage traces in a detector
for a list of a showers using the `claculate_sim_efield` and `apply_det_response` functions.
The observer positions are defined in the detector object.
The showers are defined in the shower objects.
General config settings are defined in the NuRadioMC yaml config file,
which includes the medium model (i.e. ice model) and the
propagation module to use (e.g. the analytic ray tracer).
"""
# initialize the detector description (from the json file)
kwargs = dict(json_filename=os.path.join(os.path.dirname(__file__), 'surface_station_1GHz.json'), antenna_by_depth=False)
det = detector.Detector(**kwargs)
det.update(datetime.now())

# get the general config settings
cfg = sim.get_config(os.path.join(os.path.dirname(__file__), 'config.yaml'))

# set the ice model
ice = medium.get_ice_model(cfg['propagation']['ice_model'])
# set the propagation module
# it is important to pass the detector object to the propagator for an accurate calculation of the attenuation length
# (if the detector is availble, the sampling rate is used to determine the maximum frequency for an efficient interpolation
# of the attenuation length, otherwise the internal sampling rate of the simulation is used which is much higher, this would 
# lead to inaccuracies at low frequencies)
propagator = propagation.get_propagation_module(cfg['propagation']['module'])(ice, detector=det, config=cfg)

# set the station id and channel id
sid = 101
cid = 1

# define the showers that should be simulated
showers = []
shower = NuRadioReco.framework.radio_shower.RadioShower(0)
# according to our convention, the shower direction is the direction of
# where the shower is coming from.
shower[shp.zenith] = 89 * units.deg # propagation downwards
shower[shp.azimuth] = 180 * units.deg # propagation into the positive x direction
shower[shp.energy] = 1e17 * units.eV
shower[shp.vertex] = np.array([-700*units.m, 0, -1*units.km])
shower[shp.vertex_time] = 0 * units.ns
shower[shp.type] = 'had'
showers.append(shower)

# calculate the electric fields at the observer positions from the showers
sim_station = sim.calculate_sim_efield(showers, sid, cid,
                         det, propagator, ice, cfg)

# now let's apply the detector response (antennas and signal chain)

# first we define the analog signal chain
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
def detector_simulation_filter_amp(evt, station, det):
    channelBandPassFilter.run(evt, station, det, passband=[80 * units.MHz, 1000 * units.GHz],
                                filter_type='butter', order=2)
    channelBandPassFilter.run(evt, station, det, passband=[0, 500 * units.MHz],
                                filter_type='butter', order=10)

# calculate the signal in each channel after antenna and detector response
evt = NuRadioReco.framework.event.Event(0, 0)
stn = NuRadioReco.framework.station.Station(sid)
stn.set_sim_station(sim_station)
evt.set_station(stn)
sim.apply_det_response(evt, det, cfg, detector_simulation_filter_amp, add_noise=False, channel_ids=[cid])

# the previous call of sim.apply_det_response folds the simulated efields through the antenna responses,
# combines them into one trace per antenna, and saves them to the station object. In many simulation we
# will later add noise to the station channels. If also we want to save the noiseless simulated traces,
# we can call sim.apply_det_response_sim. This will apply the antenna response to each simulated efield
# and save the traces individually to the sim_station object instead of the station object
sim.apply_det_response_sim(sim_station, det, cfg, detector_simulation_filter_amp)

# the simulated traces are longer than the readout windows defined in the detector description. We can either
# run a trigger simulation, cut the traces to a user defined readout window, or or leave them as they are if
# the full traces are desired for the analysis. Try setting cut_traces to "using_trigger" or
# "using_trace_start_time" here instead
cut_traces = False
if cut_traces == "using_trigger":
    simpleThreshold = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()
    Vrms_dummy = 1e-5 * units.V
    simpleThreshold.run(evt, stn, det,
                        threshold=3 * Vrms_dummy,
                        triggered_channels=[cid],
                        number_concidences=1,
    )
    channelReadoutWindowCutter = NuRadioReco.modules.channelReadoutWindowCutter.channelReadoutWindowCutter()
    channelReadoutWindowCutter.run(evt, stn, det)

elif cut_traces == "using_trace_start_time":
    trace_start_time = 7000 * units.ns # rough guess of the vertex_time + travel time + cable delay - 100 ns
    channelReadoutWindowCutter = NuRadioReco.modules.channelReadoutWindowCutter.channelReadoutWindowCutter()
    channelReadoutWindowCutter.cut_using_trace_start_times(evt, stn, det, trace_start_times=trace_start_time)

# Resample to detector sampling rate:
# for i, channel in enumerate(stn.iter_channels()):
#     channel.resample(det.get_sampling_frequency(stn.get_id(), channel.get_id()))


# let's plot the result for the channel we simulated:
fig, (ax, ax2) = plt.subplots(1,2)
for i, channel in enumerate(stn.iter_channels()):
    print("Plotting channel", channel.get_id())
    trace = channel.get_trace()
    ax.plot(channel.get_times(), trace/units.V, f"-C{i}", label=f'channel id {channel.get_id()}')

    ax2.plot(channel.get_frequencies()/units.MHz, np.abs(channel.get_frequency_spectrum()/units.V*units.MHz),
             f"-C{i}",label=f'channel id {channel.get_id()}')

ax.set_xlabel('Time (ns)')
ax.set_ylabel('Voltage (V)')
ax.legend()
ax2.set_xlabel('Frequency (MHz)')
ax2.set_ylabel('Amplitude (V/MHz)')
ax2.legend()
fig.tight_layout()
# only show plot if running interactively (not in CI environment)
if os.environ.get('DISPLAY') or os.environ.get('CI') is None:
    plt.show()


# alternatively, we can sum the traces of the sim channels and we should get equivalent results
for i, channel in enumerate(stn.iter_channels()):
    sim_channel_sum = None
    for sim_channel in sim_station.get_channels_by_channel_id(channel.get_id()):
        if sim_channel_sum is None:
            sim_channel_sum = sim_channel
        else:
            sim_channel_sum += sim_channel
    if sim_channel_sum is not None:
        fig, (ax, ax2) = plt.subplots(1,2)
        ax.plot(channel.get_times(), trace/units.V, f"-C{i}", label=f'channel id {channel.get_id()}')
        ax.plot(sim_channel_sum.get_times(), sim_channel_sum.get_trace()/units.V, f"--C{i+1}",
                label=f'sim_channel_sum')
        ax2.plot(channel.get_frequencies()/units.MHz, np.abs(channel.get_frequency_spectrum()/units.V*units.MHz),
            f"-C{i}",label=f'channel id {channel.get_id()}')
        ax2.plot(sim_channel_sum.get_frequencies()/units.MHz, np.abs(sim_channel_sum.get_frequency_spectrum()/units.V*units.MHz),
                f"--C{i+1}", label=f'sim_channel_sum')
ax.set_xlabel('Time (ns)')
ax.set_ylabel('Voltage (V)')
ax.legend()
ax2.set_xlabel('Frequency (MHz)')
ax2.set_ylabel('Amplitude (V/MHz)')
ax2.legend()
fig.suptitle("Comparison of station channel and sim_channel_sum", fontsize=14)
fig.tight_layout()
if os.environ.get('DISPLAY') or os.environ.get('CI') is None:
    plt.show()