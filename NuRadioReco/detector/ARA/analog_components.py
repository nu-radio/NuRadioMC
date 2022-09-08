import numpy as np
import os
import pandas as pd
import datetime
from NuRadioReco.utilities import units
from scipy.interpolate import interp1d
from NuRadioReco.detector.ARA import ARA_stations_configuration
import logging

logger = logging.getLogger("analog_components")

def load_system_response(path=os.path.dirname(os.path.realpath(__file__)), **kwargs):
    """
    Default file was imported from:
    https://github.com/bhokansonfasig/pyrex/tree/master/pyrex/custom/ara/data

    Parameters
    ----------
    path: string
        Path to the file containing the system response
    """

    station_id = kwargs.get("station_id")
    config  = kwargs.get("config")
    if config == 'C0':
       data = np.loadtxt(os.path.join(path, "HardwareResponses/default_gain.txt"),
                      skiprows=3, delimiter=' ') # Proceed with default gain (ARAsim default gain)
    else:
       data = np.loadtxt(os.path.join(path, "HardwareResponses/In_situ_Electronics_A"+str(station_id)+"_"+str(config)+".txt"),
               skiprows=3, delimiter=' ') # individual hardware response for A2 and A3 channels taken from[https://github.com/MyoungchulK/MF_filters/tree/main/data/in_situ_sc_gain]
    
    number_of_columns = len((pd.DataFrame(data)).columns)
    default = {}
    default['gain'] = []
    default['phase'] = []
    default['frequencies'] = data[:, 0] * units.MHz
    for i in range((number_of_columns)):
        if i%2 !=0 :    ###### odd numbered columns contain gain values ######
           per_channel_gain = data[:, int(i)]  # unitless
           default['gain'].append(per_channel_gain)
        elif i%2 ==0 and i!=0:    ###### even numbered columns contain phase values ######
           per_channel_phase = data[:, int(i)] 
           default['phase'].append(per_channel_phase)

    return default


system_response = {}
collect_station_id = []
system_response_collection = []


def get_system_response(frequencies, station, evt_time):
    station_id = station.get_id()
    channels = station.iter_channels()
    global config

    ## Check if stationi_id is already saved or not and read config only once for given station
    if station_id in collect_station_id:
       system_response['default'] = system_response_collection[collect_station_id.index(station_id)]
    else:
       config = ARA_stations_configuration.pass_configuration(station_id, evt_time)
       system_response['default'] = load_system_response(station_id= station_id,  config = config)
       collect_station_id.append(station_id)
       system_response_collection.append(system_response['default'])
       logger.warning(" Processing with the config {}".format(config)) 

    default_frequencies = system_response['default']['frequencies']
    phase = system_response['default']['phase']
    gain = system_response['default']['gain']
    
    system = {}
    system['gain'] = []
    system['phase'] = []

    for channel in channels:
        channel_id = channel.get_id()
        interp_gain = interp1d(default_frequencies, gain[channel_id], bounds_error=False, fill_value=0)
        interp_phase = interp1d(default_frequencies, np.unwrap(phase[channel_id]), bounds_error=False, fill_value=0) 
        gain_value = interp_gain(frequencies)
        phase_value = np.exp(1j * interp_phase(frequencies))
        system['gain'].append(gain_value)
        system['phase'].append(phase_value)

    return system
