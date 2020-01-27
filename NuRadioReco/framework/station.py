from __future__ import absolute_import, division, print_function
import NuRadioReco.framework.base_station
import NuRadioReco.framework.sim_station
import NuRadioReco.framework.channel
from six import iteritems
import pickle
import logging
import collections
logger = logging.getLogger('Station')


class Station(NuRadioReco.framework.base_station.BaseStation):

    def __init__(self, station_id):
        NuRadioReco.framework.base_station.BaseStation.__init__(self, station_id)
        self.__channels = collections.OrderedDict()
        self.__reference_reconstruction = 'RD'
        self.__sim_station = None

    def set_sim_station(self, sim_station):
        self.__sim_station = sim_station

    def get_sim_station(self):
        return self.__sim_station

    def has_sim_station(self):
        return self.__sim_station is not None

    def iter_channels(self, use_channels=None):
        for channel_id, channel in iteritems(self.__channels):
            if(use_channels is None):
                yield channel
            else:
                if channel_id in use_channels:
                    yield channel

    def get_channel(self, channel_id):
        return self.__channels[channel_id]

    def get_number_of_channels(self):
        return len(self.__channels)

    def add_channel(self, channel):
        self.__channels[channel.get_id()] = channel

    def set_reference_reconstruction(self, reference):
        if reference not in ['RD', 'MC']:
            import sys
            logger.error("reference reconstructions other than RD and MC are not supported")
            sys.exit(-1)
        self.__reference_reconstruction = reference

    def get_reference_reconstruction(self):
        return self.__reference_reconstruction

    def get_reference_direction(self):
        if(self.__reference_reconstruction == 'RD'):
            return self.get_parameter('zenith'), self.get_parameter('azimuth')
        if(self.__reference_reconstruction == 'MC'):
            return self.get_sim_station().get_parameter('zenith'), self.get_sim_station().get_parameter('azimuth')

    def get_magnetic_field_vector(self, time=None):
        if(self.__reference_reconstruction == 'MC'):
            return self.get_sim_station().get_magnetic_field_vector()
        if(self.__reference_reconstruction == 'RD'):
            if time is not None:
                logger.warning("time dependent magnetic field model not yet implemented, returning static magnetic field for the ARIANNA site")
            from radiotools import helper
            return helper.get_magnetic_field_vector('arianna')

    def serialize(self, mode):
        base_station_pkl = NuRadioReco.framework.base_station.BaseStation.serialize(self, mode)
        channels_pkl = []
        for channel in self.iter_channels():
            channels_pkl.append(channel.serialize(mode))
        sim_station_pkl = None
        if(self.has_sim_station()):
            sim_station_pkl = self.get_sim_station().serialize(mode)

        data = {'__reference_reconstruction': self.__reference_reconstruction,
                'channels': channels_pkl,
                'base_station': base_station_pkl,
                'sim_station': sim_station_pkl}
        return pickle.dumps(data, protocol=4)

    def deserialize(self, data_pkl):
        data = pickle.loads(data_pkl)
        NuRadioReco.framework.base_station.BaseStation.deserialize(self, data['base_station'])
        if(data['sim_station'] is None):
            self.__sim_station = None
        else:
            self.__sim_station = NuRadioReco.framework.sim_station.SimStation(None, None, None)
            self.__sim_station.deserialize(data['sim_station'])
        for channel_pkl in data['channels']:
            channel = NuRadioReco.framework.channel.Channel(0)
            channel.deserialize(channel_pkl)
            self.add_channel(channel)

        self.__reference_reconstruction = data['__reference_reconstruction']
