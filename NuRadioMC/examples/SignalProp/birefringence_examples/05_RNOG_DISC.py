from NuRadioMC.utilities import medium
from NuRadioReco.utilities import units
from NuRadioReco.framework import base_trace
import NuRadioReco.framework.electric_field
from NuRadioReco.detector import antennapattern
import datetime
import matplotlib.pyplot as plt
import numpy as np
from radiotools import helper as hp
from scipy import signal
from NuRadioMC.SignalProp import analyticraytracing

import NuRadioReco.detector.detector as detector
import NuRadioReco.modules.efieldToVoltageConverterPerEfield
import NuRadioReco.framework.sim_station
"""
###-----------------------------------------
#   EXAMPLE: Script to simulate the effects of birefringence on the vpol and hpol amplitude at the RNOG stations. 
###-----------------------------------------
"""

DISC_in_BIG_cord = np.array([168.81 , 432.705 , 0.075575 ])* units.m
BIG_in_BIG_cord = np.array([0 , 0 , 0])* units.m
DISC_in_DISC_cord = np.array([0 , 0 , 0])* units.m
loc_1_in_BIG_cord = np.array([-97.759100, 1915.1097 , -0.2])* units.m
loc_2_in_BIG_cord = np.array([-450.218, 1907.4 , -0.2])* units.m
loc_3_in_BIG_cord = np.array([-238.953, 1150.35 , -0.2])* units.m
loc_4_in_BIG_cord = np.array([-693.31, 1658.41 , -0.2])* units.m


BIG_in_DISC_cord = BIG_in_BIG_cord - DISC_in_BIG_cord
loc_1_in_DISC_cord = loc_1_in_BIG_cord - DISC_in_BIG_cord
loc_2_in_DISC_cord = loc_2_in_BIG_cord - DISC_in_BIG_cord
loc_3_in_DISC_cord = loc_3_in_BIG_cord - DISC_in_BIG_cord
loc_4_in_DISC_cord = loc_4_in_BIG_cord - DISC_in_BIG_cord

detector_file = detector.Detector(json_filename='../../../../NuRadioReco/detector/RNO_G/RNO_season_2022.json', antenna_by_depth=False)
evt_time=datetime.datetime(2023, 1, 1)
detector_file.update(evt_time)
sim_station = NuRadioReco.framework.sim_station
evt = NuRadioReco.framework.event.Event(0, 0)


ref_index_model = 'greenland_simple'

ice = medium.get_ice_model(ref_index_model)
rays = analyticraytracing.ray_tracing(ice)
ray_tracing_solution = 0

config = {'propagation': {}}
config['propagation']['attenuate_ice'] = True
config['propagation']['focusing_limit'] = 2
config['propagation']['focusing'] = False
config['propagation']['birefringence'] = True

emitter = 'DISC'
ice_models = ['B', 'A', 'C']
flow_directions = [-10, 0, +10]
stations = [11, 12, 13, 21, 22, 23, 24]
channels = [0]

if emitter == 'DISC':
    depths = np.arange(-600, 0, 10)
    stations = [11, 12, 21, 22]

elif (emitter == 'loc1') or (emitter == 'loc2') or (emitter == 'loc3') or (emitter == 'loc4'):
    depths = np.arange(-400, 0, 10)
    stations = [11, 12, 13, 21, 22, 23]

time_delays = np.zeros((len(flow_directions), len(ice_models), len(stations), len(channels), len(depths)))
time_delays_simple = np.zeros((len(flow_directions), len(ice_models), len(stations), len(channels), len(depths)))
path_length = np.zeros((len(flow_directions), len(ice_models), len(stations), len(channels), len(depths)))

for direction_id in range(len(flow_directions)):
    #print('flow direction: ', flow_directions[direction_id])

    th = flow_directions[direction_id] * np.pi / 180
    rot = np.matrix([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

    for model_id in range(len(ice_models)):
        #print('model: model_', ice_models[model_id])

        bire_ice_model = 'greenland_'+ ice_models[model_id]

        for station_id in range(len(stations)):
            #print('station: ', stations[station_id])
            station = NuRadioReco.framework.station.Station(station_id)
            station.set_sim_station(sim_station)
            evt.set_station(station)

            for channel_id in range(len(channels)):
                #print('channel: ', channels[channel_id])
                
                for depth in range(len(depths)):
                    #print('depth: ', depths[depth])

                    if emitter == 'DISC':
                        emitter_position = np.array([0 , 0 , depths[depth] ])* units.m  #DISC hole

                    elif emitter == 'loc1':
                        emitter_position = np.array([loc_1_in_DISC_cord[0] , loc_1_in_DISC_cord[1] , depths[depth] ])* units.m #loc1 hole
                    elif emitter == 'loc2':
                        emitter_position = np.array([loc_2_in_DISC_cord[0] , loc_2_in_DISC_cord[1] , depths[depth] ])* units.m #loc1 hole                    
                    elif emitter == 'loc3':
                        emitter_position = np.array([loc_3_in_DISC_cord[0] , loc_3_in_DISC_cord[1] , depths[depth] ])* units.m #loc1 hole
                    elif emitter == 'loc4':
                        emitter_position = np.array([loc_4_in_DISC_cord[0] , loc_4_in_DISC_cord[1] , depths[depth] ])* units.m #loc1 hole  

                    antenna_position = detector_file.get_relative_position(stations[station_id], channels[channel_id]) + detector_file.get_absolute_position(stations[station_id])

                    emitter_position[:2] = np.matmul(rot, emitter_position[:2])
                    antenna_position[:2] = np.matmul(rot, antenna_position[:2])

                    rays.set_start_and_end_point(emitter_position, antenna_position)
                    rays.find_solutions()

                    if rays.get_number_of_solutions() == 0:
                        #print('not found')
                        time_delays[direction_id, model_id, station_id, channel_id, depth] = None
                        path_length[direction_id, model_id, station_id, channel_id, depth] = None
                        time_delays_simple[direction_id, model_id, station_id, channel_id, depth] = None
                        continue

                    length = rays.get_path_length(ray_tracing_solution)
                    shift = 10
                    
                    path_properties = rays.get_path_properties_birefringence(ray_tracing_solution, bire_model = bire_ice_model)

                    time_delays_simple[direction_id, model_id, station_id, channel_id, depth] = np.sum(path_properties['second_time_delay']) - np.sum(path_properties['first_time_delay'])
                    path_length[direction_id, model_id, station_id, channel_id, depth] = length

fig, ax = plt.subplots(len(flow_directions), len(ice_models), figsize=(11, 11))

for direction_id in range(len(flow_directions)):
    #print('flow direction: ', flow_directions[direction_id])

    for model_id in range(len(ice_models)):
        #print('model: model_', ice_models[model_id])


        for station_id in range(len(stations)):
            #print('station: ', stations[station_id])

            for channel_id in range(len(channels)):
                #print('channel: ', channels[channel_id])
                shift = 7

                first = ax[direction_id, model_id].plot(depths, path_length[direction_id, model_id, station_id, channel_id, :]*shift/1000, '--', label= str(shift) + 'ns/km: station = ' + str(stations[station_id])) #+ ', channel = ' + str(channels[channel_id]))
                ax[direction_id, model_id].plot(depths, time_delays_simple[direction_id, model_id, station_id, channel_id, :], color=first[0].get_color(), label= 'station = ' + str(stations[station_id])) #+ ', channel = ' + str(channels[channel_id]))

        if ice_models[model_id] == 'B':
            ax[direction_id, model_id].set_ylabel('time delay [ns]')

        if flow_directions[direction_id] == 10:
            ax[direction_id, model_id].set_xlabel('emitter depth [m]')

        if (flow_directions[direction_id] == -10) and (ice_models[model_id] == 'A'):
            ax[direction_id, model_id].legend()     

        ax[direction_id, model_id].set_ylim(0, 20)
        ax[direction_id, model_id].grid(True)
        ax[direction_id, model_id].set_xlim(np.min(depths), np.max(depths))
        ax[direction_id, model_id].set_title('direction: ' + str(flow_directions[direction_id]) + ', model: ' + str(ice_models[model_id]))

plt.tight_layout() 
plt.savefig(emitter + '_greenland', dpi=600)

