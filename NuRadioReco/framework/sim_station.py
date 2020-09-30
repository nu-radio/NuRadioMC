from __future__ import absolute_import, division, print_function
import NuRadioReco.framework.base_station
import NuRadioReco.framework.channel
import NuRadioReco.framework.sim_channel
import collections
try:
    import cPickle as pickle
except ImportError:
    import pickle
import logging
logger = logging.getLogger('SimStation')


class SimStation(NuRadioReco.framework.base_station.BaseStation):

    def __init__(self, station_id):
        NuRadioReco.framework.base_station.BaseStation.__init__(self, station_id)
        self.__magnetic_field_vector = None
        self.__simulation_weight = None
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
        adds a NuRadioReco.framework.sim_channel to the SimStation object
        """
        if not isinstance(channel, NuRadioReco.framework.sim_channel.SimChannel):
            raise AttributeError("channel needs to be of type NuRadioReco.framework.sim_channel")
        if(channel.get_unique_identifier() in self.__channels):
            raise AttributeError(f"channel with the unique identifier {channel.get_unique_identifier()} is already present in SimStation")
        self.__channels[channel.get_unique_identifier()] = channel

    def get_channel(self, unique_identifier):
        """
        returns channel identified by the triple (channel_id, shower_id, ray_tracing_id)
        """
        return self.__channels[unique_identifier]

    def get_channel_ids(self):
        """
        returns a list with the channel IDs of all simChannels of the simStation
        """
        channel_ids = []
        for unique_identifier in self.__channels.keys():
            if unique_identifier[0] not in channel_ids:
                channel_ids.append(unique_identifier[0])
        channel_ids.sort()
        return channel_ids

    def get_shower_ids(self):
        """
        returns a list with the shower IDs of all simChannels of the simStation
        """
        shower_ids = []
        for unique_identifier in self.__channels.keys():
            if unique_identifier[1] not in shower_ids:
                shower_ids.append(unique_identifier[1])
        shower_ids.sort()
        return shower_ids

    def get_ray_tracing_ids(self):
        """
        returns a list with the raytracing IDs of all simChannels of the simStation
        """
        ray_tracing_ids = []
        for unique_identifier in self.__channels.keys():
            if unique_identifier[2] not in ray_tracing_ids:
                ray_tracing_ids.append(unique_identifier[2])
        ray_tracing_ids.sort()
        return ray_tracing_ids

    def get_channels_by_channel_id(self, channel_id):
        """
        returns all simChannels that have the given channel_id
        """
        for channel in self.__channels.values():
            if channel.get_id() == channel_id:
                yield channel

    def get_channels_by_shower_id(self, shower_id):
        """
        returns all simChannels that have the given shower_id
        """
        for channel in self.__channels.values():
            if channel.get_shower_id() == shower_id:
                yield channel

    def get_channels_by_ray_tracing_id(self, ray_tracing_id):
        """
        returns all simChannels that have the given ray_tracing_id
        """
        for channel in self.__channels.values():
            if channel.get_ray_tracing_solution_id() == ray_tracing_id:
                yield channel

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
        if 'channels' in data.keys():
            for channel_pkl in data['channels']:
                channel = NuRadioReco.framework.sim_channel.SimChannel(0, 0, 0)
                channel.deserialize(channel_pkl)
                self.add_channel(channel)
