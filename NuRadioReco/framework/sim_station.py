from __future__ import absolute_import, division, print_function
import NuRadioReco.framework.base_station
import cPickle as pickle


class SimStation(NuRadioReco.framework.base_station.BaseStation):

    def __init__(self, station_id, sampling_rate=None, trace=None, position=None):
        NuRadioReco.framework.base_station.BaseStation.__init__(self, station_id)
        self.__channels = {}
        self.__position = position
        self.__magnetic_field_vector = None
        self.__simulation_weight = None
        if(trace is not None and sampling_rate is not None):
            self.set_trace(trace, sampling_rate)

    def get_channels(self):
        return self.__channels.values()

    def has_channels(self):
        return self.__channels is not {}

    def iter_channels(self, use_channels=None):
        for channel_id, channel in self.__channels.iteritems():
            if(use_channels is None):
                yield channel
            else:
                if channel_id in use_channels:
                    yield channel

    def get_channel(self, channel_id):
        return self.__channels[channel_id]

    def add_channel(self, channel):
        if(channel.get_id() not in self.__channels.keys()):
            self.__channels[channel.get_id()] = []
        self.__channels[channel.get_id()].append(channel)

    def get_position(self):
        return self.__position

    def get_magnetic_field_vector(self):
        return self.__magnetic_field_vector

    def set_magnetic_field_vector(self, magnetic_field_vector):
        self.__magnetic_field_vector = magnetic_field_vector

    def get_simulation_weight(self):
        return self.__simulation_weight

    def set_simulation_weight(self, simulation_weight):
        self.__simulation_weight = simulation_weight

    def serialize(self, mode):
        base_station_pkl = NuRadioReco.framework.base_station.BaseStation.serialize(self, mode)

        channels_pkl = []
        for channel1 in self.get_channels():
            for channel2 in channel1:
                channels_pkl.append(channel2.serialize(mode))

        data = {'__magnetic_field_vector': self.__magnetic_field_vector,
                '__simulation_weight': self.__simulation_weight,
                '__position': self.__position,
                'channels': channels_pkl,
                'base_station': base_station_pkl}
        return pickle.dumps(data, protocol=2)

    def deserialize(self, data_pkl):
        data = pickle.loads(data_pkl)
        NuRadioReco.framework.base_station.BaseStation.deserialize(self, data['base_station'])
        self.__magnetic_field_vector = data['__magnetic_field_vector']
        self.__simulation_weight = data['__simulation_weight']
        self.__position = data['__position']
        for channel_pkl in data['channels']:
            channel = NuRadioReco.framework.channel.Channel(0)
            channel.deserialize(channel_pkl)
            self.add_channel(channel)
