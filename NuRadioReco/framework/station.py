from __future__ import absolute_import, division, print_function
import NuRadioReco.framework.base_station
import NuRadioReco.framework.sim_station
import NuRadioReco.framework.channel
import NuRadioReco.framework.parameters
from six import iteritems
import pickle
from NuRadioReco.utilities.io_utilities import _dumps
import logging
import collections
logger = logging.getLogger('NuRadioReco.Station')


class Station(NuRadioReco.framework.base_station.BaseStation):

    def __init__(self, station_id):
        NuRadioReco.framework.base_station.BaseStation.__init__(self, station_id)
        self.add_parameter_type(NuRadioReco.framework.parameters.ARIANNAParameters)
        self.__channels = collections.OrderedDict()
        self.__reference_reconstruction = 'RD'
        self.__sim_station = None

    def set_sim_station(self, sim_station):
        """
        Sets the SimStation of the Station. If a SimStation is already present, it is overwritten.

        Parameters
        ----------
        sim_station : NuRadioReco.framework.sim_station.SimStation
            The SimStation to set as the SimStation of the Station.
        """
        self.__sim_station = sim_station

    def add_sim_station(self, sim_station):
        """
        Adds a SimStation to the Station. If a SimStation is already present,
        the new SimStation is merged to the existing one.

        Parameters
        ----------
        sim_station : NuRadioReco.framework.sim_station.SimStation
            The SimStation to add to the Station.
        """
        if self.__sim_station is None:
            self.__sim_station = sim_station
        else:
            self.__sim_station = self.__sim_station + sim_station

    def get_sim_station(self):
        """
        Returns the SimStation of the Station.

        Returns
        -------
        NuRadioReco.framework.sim_station.SimStation
            The SimStation of the Station.
        """
        return self.__sim_station

    def has_sim_station(self):
        """
        Returns whether the Station has a SimStation.

        Returns
        -------
        bool
            True if the Station has a SimStation, False otherwise.
        """
        return self.__sim_station is not None

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
        NuRadioReco.framework.channel.Channel
            The next channel in the iteration.
        """
        channel_ids = self.get_channel_ids()

        if use_channels is not None:
            channel_ids = [channel_id for channel_id in use_channels if channel_id in channel_ids]

        if sorted:
            channel_ids.sort()

        for channel_id in channel_ids:
            yield self.get_channel(channel_id)

    def iter_trigger_channels(self, use_channels=None):
        """ Iterates over all channels of the station and yields `channel.get_trigger_channel()` for each.

        If `use_channels` is not None, only the channels with the ids in `use_channels` are iterated over.

        Parameters
        ----------
        use_channels : list of int, optional
            List of channel ids to iterate over. If None, all channels are iterated over.

        Yields
        ------
        NuRadioReco.framework.channel.Channel
            The next (trigger) channel in the iteration.

        See Also
        --------
        NuRadioReco.framework.channel.Channel.get_trigger_channel
        NuRadioReco.framework.station.Station.iter_channels
        """

        for channel_id, channel in iteritems(self.__channels):
            if use_channels is None:
                yield channel.get_trigger_channel()
            else:
                if channel_id in use_channels:
                    yield channel.get_trigger_channel()

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
        channel: `NuRadioReco.framework.channel.Channel`
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
        channel : `NuRadioReco.framework.channel.Channel`
            The channel to add to the station.
        overwrite : bool, (Default: True)
            If True, allow to overwrite an existing channel (i.e., a channel with the same id).
            If False, raise AttributeError if a channel with the same id is being added.
        """
        if not isinstance(channel, NuRadioReco.framework.channel.Channel):
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
        channel_id : int or NuRadioReco.framework.channel.Channel
            The Channel (id) to remove from the Station.
        """
        if isinstance(channel_id, NuRadioReco.framework.channel.Channel):
            del self.__channels[channel_id.get_id()]
        else:
            del self.__channels[channel_id]

    def set_reference_reconstruction(self, reference):
        if reference not in ['RD', 'MC']:
            logger.error(f"Reference reconstructions other than 'RD' and 'MC' are not supported. Used value: '{reference}'")
            raise ValueError(f"Reference reconstructions other than 'RD' and 'MC' are not supported. Used value: '{reference}'")

        self.__reference_reconstruction = reference

    def get_reference_reconstruction(self):
        return self.__reference_reconstruction

    def get_reference_direction(self):
        if self.__reference_reconstruction == 'RD':
            return self.get_parameter('zenith'), self.get_parameter('azimuth')
        elif self.__reference_reconstruction == 'MC':
            return (
                self.get_sim_station().get_parameter('zenith'),
                self.get_sim_station().get_parameter('azimuth')
            )
        else:
            logger.error(f"Reference reconstruction not set / unknown: {self.__reference_reconstruction}")
            raise ValueError(f"Reference reconstruction not set / unknown: {self.__reference_reconstruction}")


    def get_magnetic_field_vector(self, time=None):
        if self.__reference_reconstruction == 'MC':
            return self.get_sim_station().get_magnetic_field_vector()
        elif self.__reference_reconstruction == 'RD':
            logger.error(
                "Magnetic field for `self.__reference_reconstruction == 'RD'` not implemented yet. "
                "Please use `radiotools.helper.get_magnetic_field_vector(site)` for the site you are interested in.")
            raise NotImplementedError(
                "Magnetic field for `self.__reference_reconstruction == 'RD'` not implemented yet. "
                "Please use `radiotools.helper.get_magnetic_field_vector(site)` for the site you are interested in."
            )
        else:
            logger.error(f"Reference reconstruction not set / unknown: {self.__reference_reconstruction}")
            raise ValueError(f"Reference reconstruction not set / unknown: {self.__reference_reconstruction}")

    def serialize(self, mode):
        save_efield_traces = 'ElectricFields' in mode and mode['ElectricFields'] is True
        base_station_pkl = NuRadioReco.framework.base_station.BaseStation.serialize(
            self, save_efield_traces=save_efield_traces)

        save_channel_trace = mode.get('Channels', False)
        channels_pkl = [channel.serialize(save_channel_trace) for channel in self.iter_channels()]

        sim_station_pkl = None
        if self.has_sim_station():
            sim_station_pkl = self.get_sim_station().serialize(
                save_channel_traces=mode.get('SimChannels', False),
                save_efield_traces=mode.get('SimElectricFields', False))

        data = {'__reference_reconstruction': self.__reference_reconstruction,
                'channels': channels_pkl,
                'base_station': base_station_pkl,
                'sim_station': sim_station_pkl}

        return _dumps(data, protocol=4)

    def deserialize(self, data_pkl):
        data = pickle.loads(data_pkl)
        NuRadioReco.framework.base_station.BaseStation.deserialize(self, data['base_station'])

        if data['sim_station'] is None:
            self.__sim_station = None
        else:
            self.__sim_station = NuRadioReco.framework.sim_station.SimStation(None)
            self.__sim_station.deserialize(data['sim_station'])

        for channel_pkl in data['channels']:
            channel = NuRadioReco.framework.channel.Channel(0)
            channel.deserialize(channel_pkl)
            self.add_channel(channel)

        self.__reference_reconstruction = data['__reference_reconstruction']
