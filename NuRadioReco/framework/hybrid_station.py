"""
This module implements the HybridStation class

This class is similar to `NuRadioReco.modules.framework.station.Station`, but is intended
for stations that feature non-radio channels, e.g. particle detectors.

"""

from NuRadioReco.framework.base_station import BaseStation
from NuRadioReco.framework.hybrid_channel import Channel
from NuRadioReco.framework.sim_station import SimStation
from NuRadioReco.utilities.io_utilities import _dumps

import pickle
from six import iteritems

class HybridStation(BaseStation):

    def __init__(self, station_id):
        super().__init__(station_id)
        self.__channels = {}

    def get_channel(self, channel_id):
        return self.__channels[channel_id]

    def get_trigger_channel(self, channel_id):
        """
        Returns the trigger channel of channel with id `channel_id`.

        If the trigger channel is not set, the channel itself is returned (i.e. this is equivalent to `get_channel`)

        Parameters
        ----------
        channel_id : int
            The id of the channel for which to get the trigger channel.

        Returns
        -------
        channel: `NuRadioReco.framework.hybrid_channel.Channel`
            The trigger channel of the channel with id `channel_id`.
        """
        channel = self.get_channel(channel_id)
        return channel.get_trigger_channel()

    def iter_channel_group(self, channel_group_id):
        found_channel_group = False
        for channel_id, channel in iteritems(self.__channels):
            if(channel.get_group_id() == channel_group_id):
                found_channel_group = True
                yield channel
        if found_channel_group == False:
            msg = f"channel group id {channel_group_id} is not present"
            raise ValueError(msg)

    def get_number_of_channels(self):
        return len(self.__channels)

    def get_channel_ids(self, return_group_ids=False):
        """
        Return all channel ids in the station

        Parameters
        ----------
        return_group_ids : bool, default: False
            If True, return a list of channel_group_ids
            instead of channel ids. Note that if no channel group ids
            are defined, these are the same as the channel ids

        Returns
        -------
        channel_ids : list
            List of all channel ids
        """
        if return_group_ids:
            channel_ids = set() # we use a set to avoid duplicates
            for channel in self.iter_channels():
                channel_ids.add(channel.get_group_id())
        else:
            channel_ids = self.__channels.keys()

        return list(channel_ids)

    def add_channel(self, channel, overwrite=False):
        """
        Adds a channel to the station. If a channel with the same id is already present, it is overwritten.

        Parameters
        ----------
        channel : `NuRadioReco.framework.hybrid_channel.Channel`
            The channel to add to the station.
        overwrite : bool, (Default: True)
            If True, allow to overwrite an existing channel (i.e., a channel with the same id).
            If False, raise AttributeError if a channel with the same id is being added.
        """
        if not isinstance(channel, Channel):
            raise AttributeError("`Channel` needs to be of type `NuRadioReco.framework.channel.Channel`")

        if not overwrite and channel.get_id() in self.__channels:
            raise AttributeError(
                f"Channel with the id {channel.get_id()} is already present in Station. "
                "If you want to add this channel nonetheless please pass `overwrite=True` as argument")

        self.__channels[channel.get_id()] = channel

    def has_channel(self, channel_id):
        return channel_id in self.__channels

    def remove_channel(self, channel_id):
        """
        Removes a channel from the station by deleting is from the channels dictionary. The `channel_id`
        should be the id of the Channel, but supplying the Channel object itself is also supported.

        Parameters
        ----------
        channel_id : int or NuRadioReco.framework.hybrid_channel.Channel
            The Channel (id) to remove from the Station.
        """
        if isinstance(channel_id, Channel):
            del self.__channels[channel_id.get_id()]
        else:
            del self.__channels[channel_id]

    def iter_channels(self, use_channels=None, sorted=False):
        """ Iterates over all channels of the station.

        If `use_channels` is not None, only the channels with the ids in `use_channels`
        are iterated over. If `sorted` is True, the channels are iterated over in
        ascending order of their ids.

        Parameters
        ----------
        use_channels : list of int, optional
            List of channel ids to iterate over. If None, all channels are iterated over.
        sorted : bool, optional
            If True, the channels are iterated over in ascending order of their ids.

        Yields
        ------
        NuRadioReco.framework.hybrid_channel.Channel
            The next channel in the iteration.
        """
        channel_ids = self.get_channel_ids()

        if use_channels is not None:
            channel_ids = [channel_id for channel_id in use_channels if channel_id in channel_ids]

        if sorted:
            channel_ids.sort()

        for channel_id in channel_ids:
            yield self.get_channel(channel_id)

    def serialize(self, mode):
        """
        Serialize the HybridStation

        Serializes the class instance as a pickled bytes for storage.

        """
        save_efield_traces = 'ElectricFields' in mode and mode['ElectricFields'] is True
        base_station_pkl = BaseStation.serialize(
            self, save_efield_traces=save_efield_traces)

        save_channel_trace = mode.get('Channels', False)
        channels_pkl = [channel.serialize(save_channel_trace) for channel in self.iter_channels()]

        data = {
                'channels': channels_pkl,
                'base_station': base_station_pkl,
                'sim_station': None}

        return _dumps(data, protocol=4)

    def deserialize(self, data_pkl):
        data = pickle.loads(data_pkl)
        BaseStation.deserialize(self, data['base_station'])

        if data['sim_station'] is None:
            self.__sim_station = None
        else:
            self.__sim_station = SimStation(None)
            self.__sim_station.deserialize(data['sim_station'])

        for channel_pkl in data['channels']:
            channel = Channel(0)
            channel.deserialize(channel_pkl)
            self.add_channel(channel)

