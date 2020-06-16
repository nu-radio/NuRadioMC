from __future__ import absolute_import, division, print_function
import NuRadioReco.framework.base_station
import NuRadioReco.framework.channel
import NuRadioReco.framework.sim_channel
import numpy as np
from six import iteritems
import collections
try:
    import cPickle as pickle
except ImportError:
    import pickle
import logging
logger = logging.getLogger('SimStation')


class SimStation(NuRadioReco.framework.base_station.BaseStation):

    def __init__(self, station_id, sampling_rate=None, trace=None):
        NuRadioReco.framework.base_station.BaseStation.__init__(self, station_id)
        self.__magnetic_field_vector = None
        self.__simulation_weight = None
        if(trace is not None and sampling_rate is not None):
            self.set_electric_fields(trace, sampling_rate)
        self.__channels = collections.OrderedDict()

    def get_magnetic_field_vector(self):
        return self.__magnetic_field_vector

    def set_magnetic_field_vector(self, magnetic_field_vector):
        self.__magnetic_field_vector = magnetic_field_vector

    def get_simulation_weight(self):
        return self.__simulation_weight

    def set_simulation_weight(self, simulation_weight):
        self.__simulation_weight = simulation_weight

    def iter_channels(self):
        for channel in self.__channels.values():
            yield channel

    def add_channel(self, channel):
        """
        adds a NuRadioReco.framework.channel ot the SimStation object
        """
        if not isinstance(channel, NuRadioReco.framework.sim_channel.SimChannel):
            raise AttributeError(f"channel needs to be of type NuRadioReco.framework.sim_channel")
        if(channel.get_unique_identifier() in self.__channels):
            raise AttributeError(f"channel with the unique identifier {channel.get_unique_identifier()} is already present in SimStation")
        self.__channels[channel.get_unique_identifier()] = channel

    def get_channel(self, unique_identifier):
        """
        returns channel identified by the triple (channel_id, shower_id, ray_tracing_id)
        """
        return self.__channels[unique_identifier]

    def serialize(self, save_channel_traces, save_efield_traces):
        base_station_pkl = NuRadioReco.framework.base_station.BaseStation.serialize(self, save_efield_traces=save_efield_traces)
        channels_pkl = []
        for channel in self.iter_channels():
            channels_pkl.append(channel.serialize(save_trace=save_channel_traces))
        data = {'__magnetic_field_vector': self.__magnetic_field_vector,
                '__simulation_weight': self.__simulation_weight,
                'channels': channels_pkl,
                'base_station': base_station_pkl}
        return pickle.dumps(data, protocol=4)

    def deserialize(self, data_pkl):
        data = pickle.loads(data_pkl)
        NuRadioReco.framework.base_station.BaseStation.deserialize(self, data['base_station'])
        self.__magnetic_field_vector = data['__magnetic_field_vector']
        self.__simulation_weight = data['__simulation_weight']
        for channel_pkl in data['channels']:
            channel = NuRadioReco.framework.sim_channel.SimChannel(0, 0, 0)
            channel.deserialize(channel_pkl)
            self.add_channel(channel)

