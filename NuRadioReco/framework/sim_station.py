from __future__ import absolute_import, division, print_function
import NuRadioReco.framework.base_station
import NuRadioReco.framework.channel
import NuRadioReco.framework.sim_channel
import collections
import pickle
from NuRadioReco.utilities.io_utilities import _dumps
import logging
logger = logging.getLogger('NuRadioReco.SimStation')


class SimStation(NuRadioReco.framework.base_station.BaseStation):

    def __init__(self, station_id):
        NuRadioReco.framework.base_station.BaseStation.__init__(self, station_id)
        self.__magnetic_field_vector = None
        self.__simulation_weight = None
        self.__channels = collections.OrderedDict()
        self.__candidate = None

    def set_candidate(self, candidate_status):
            """
            Set the candidate for the simulation station. True means the station is a candidate for producing a trigger.

            Parameters
            ----------
            candidate_status : bool
                If the station is a candidate for producing a trigger.

            Returns
            -------
            None
            """
            if not isinstance(candidate_status, bool) and candidate_status is not None:
                raise ValueError("The candidate_status must be a bool or None.")
            self.__candidate = candidate_status

    def is_candidate(self):
        """
        Returns whether the station is a candidate for producing a trigger.

        Returns
        -------
        bool
            True if the station is a candidate for producing a trigger, False otherwise.
        """
        if self.__candidate is None:
            raise ValueError("The candidate status has not been set.")
        return self.__candidate

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

    def add_channel(self, channel, overwrite=False):
        """
        Add a `SimChannel` to the `SimStation`.

        Parameters
        ----------
        channel: `NuRadioReco.framework.sim_channel.SimChannel`
            Channel to be added.
        overwrite: bool (Default: False)
            If True, allow to overwrite a existing channel (i.e., a channel with the same unique identifier).
            If False, raise AttributeError if a channel with the same identifier is being added
        """
        if not isinstance(channel, NuRadioReco.framework.sim_channel.SimChannel):
            raise AttributeError("`Channel` needs to be of type `NuRadioReco.framework.sim_channel.SimChannel`")

        if not overwrite and channel.get_unique_identifier() in self.__channels:
            raise AttributeError(
                f"Channel with the unique identifier {channel.get_unique_identifier()} is already present in SimStation")

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
        return _dumps(data, protocol=4)

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

    def __add__(self, x):
        """
        adds a SimStation object to another SimStation object
        WARNING: Only channel and efield objects are added but no other meta information
        """
        if not isinstance(x, SimStation):
            raise AttributeError("Can only add SimStation to SimStation")
        if self.get_id() != x.get_id():
            raise AttributeError("Can only add SimStations with the same ID")
        for channel in x.iter_channels():
            if channel.get_unique_identifier() in self.__channels:
                raise AttributeError(f"Channel with ID {channel.get_unique_identifier()} already present in SimStation")
            self.add_channel(channel)
        efield_ids = self.get_electric_field_ids()
        for efield in x.get_electric_fields():
            if efield.get_unique_identifier() in efield_ids:
                raise AttributeError(f"Electric field with unique identifier {efield.get_unique_identifier()} already present in SimStation")
            self.add_electric_field(efield)
        return self
