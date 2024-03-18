import os
import yaml
import numpy as np
from NuRadioMC.simulation import simulation2 as sim
from NuRadioReco.detector import detector
from NuRadioMC.SignalProp import propagation
from NuRadioMC.utilities import medium
import NuRadioReco.framework.radio_shower
import NuRadioReco.modules.channelBandPassFilter
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.utilities import units
from datetime import datetime

"""
This script is an example of how to calculate the efield at observer positions
for a list of a showers using the `claculate_sim_efield` function.
The observer positions are defined in the detector object.
The showers are defined in the shower objects.
General config settings are defined in the NuRadioMC yaml config file.
The user also needs to specify the medium model (i.e. ice model) and the
propagation module to use (e.g. the analytic ray tracer).
"""

# set the ice model
ice = medium.get_ice_model('southpole_simple')
# set the propagation module
propagator = propagation.get_propagation_module("analytic")(ice)

# set the station id and channel id
sid = 101
cid = 1

# get the general config settings
cfg = sim.get_config("config.yaml")

# initialize the detector description (from the json file)
kwargs = dict(json_filename="surface_station_1GHz.json", antenna_by_depth=False)
det = detector.Detector(**kwargs)
det.update(datetime.now())

# define the showers that should be simulated
showers = []
shower = NuRadioReco.framework.radio_shower.RadioShower(0)
# according to our convention, the shower direction is the direction of
# where the shower is coming from.
shower[shp.zenith] = 89 * units.deg # propagation downwards
shower[shp.azimuth] = 180 * units.deg # propagation into the positive x direction
shower[shp.energy] = 1e17 * units.eV
shower[shp.vertex] = np.array([-700*units.m, 0, -1*units.km])
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

# applies the detector response to the electric fields (the antennas are defined
# in the json detector description file)
sim.apply_det_response_sim(sim_station, det, cfg, detector_simulation_filter_amp)


# calculate the signal in each channel after antenna and detector response
evt = NuRadioReco.framework.event.Event(0, 0)
stn = NuRadioReco.framework.station.Station(sid)
stn.set_sim_station(sim_station)
evt.set_station(stn)
sim.apply_det_response(evt, det, cfg, detector_simulation_filter_amp, add_noise=False)

import matplotlib.pyplot as plt
# let's plot the results
fig, (ax, ax2) = plt.subplots(1,2)
for i, channel in enumerate(stn.iter_channels()):
    if channel.get_id() != cid:
        continue
    print(channel.get_id())
    trace = channel.get_trace()
    ax.plot(channel.get_times(), trace/units.V, f"-C{i}", label=f'channel id {channel.get_id()}')

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Voltage (V)')
    ax2.plot(channel.get_frequencies()/units.MHz, np.abs(channel.get_frequency_spectrum()/units.V*units.MHz),
             f"-C{i}",label=f'channel id {channel.get_id()}')

    # alternatively, we can sum the traces of the sim channels and we should get equivalent results
    sim_channel_sum = None
    for sim_channel in sim_station.get_channels_by_channel_id(channel.get_id()):
        if sim_channel_sum is None:
            sim_channel_sum = sim_channel
        else:
            sim_channel_sum += sim_channel
    if sim_channel_sum is not None:
        ax.plot(sim_channel_sum.get_times(), sim_channel_sum.get_trace()/units.V, f"--C{i+1}",
                label=f'channel id {channel.get_id()}')
        ax2.plot(sim_channel_sum.get_frequencies()/units.MHz, np.abs(sim_channel_sum.get_frequency_spectrum()/units.V*units.MHz),
             f"--C{i+1}",label=f'channel id {channel.get_id()}')
    ax2.set_xlabel('Frequency (MHz)')
    ax2.set_ylabel('Voltage (V/MHz)')
ax.legend()
ax2.legend()
fig.tight_layout()
plt.show()
