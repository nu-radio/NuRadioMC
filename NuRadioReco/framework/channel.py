from __future__ import absolute_import, division, print_function
import NuRadioReco.framework.base_trace
import NuRadioReco.framework.parameters as parameters
import NuRadioReco.framework.parameter_serialization
import copy
try:
    import cPickle as pickle
except ImportError:
    import pickle
import logging
logger = logging.getLogger('NuRadioReco.Channel')


class Channel(NuRadioReco.framework.base_trace.BaseTrace):

    def __init__(self, channel_id, channel_group_id=None):
        """
        Parameters
        ----------
        channel_id: int
            the id of the channel
        channel_group_id: int (default None)
            optionally, several channels can belong to a "channel group". Use case is to identify
            the channels of a single dual or triple polarized antenna as common in air shower arrays.

        """
        NuRadioReco.framework.base_trace.BaseTrace.__init__(self)
        self._parameters = {}
        self._id = channel_id
        self._group_id = channel_group_id
        self._trigger_channel = None

    def set_trigger_channel(self, trigger_channel):
        """ Sets an extra trigger channel of this channel. """
        if not isinstance(trigger_channel, Channel):
            logger.error("trigger_channel needs to be of type NuRadioReco.framework.Channel")
            raise ValueError("trigger_channel needs to be of type NuRadioReco.framework.Channel")

        if trigger_channel.get_id() != self.get_id():
            msg = (f"channel id of trigger channel {trigger_channel.get_id()} is different "
                f"from the channel id {self.get_id()}")
            logger.error(msg)
            raise ValueError(msg)

        self._trigger_channel = trigger_channel

    def get_trigger_channel(self):
        """ Returns the trigger channel of this channel. If no trigger channel is set, this channel is returned. """
        if self._trigger_channel is None:
            return self

        return self._trigger_channel

    def has_extra_trigger_channel(self):
        """ Returns True if an extra trigger channel is set, i.e., if `self._trigger_channel` is not None. """
        return self._trigger_channel is not None

    def get_parameter(self, key):
        if not isinstance(key, parameters.channelParameters):
            logger.error("parameter key needs to be of type NuRadioReco.framework.parameters.channelParameters")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.channelParameters")
        return self._parameters[key]

    def get_parameters(self):
        return self._parameters

    def set_parameter(self, key, value):
        if not isinstance(key, parameters.channelParameters):
            logger.error("parameter key needs to be of type NuRadioReco.framework.parameters.channelParameters")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.channelParameters")
        self._parameters[key] = value

    def has_parameter(self, key):
        if not isinstance(key, parameters.channelParameters):
            logger.error("parameter key needs to be of type NuRadioReco.framework.parameters.channelParameters")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.channelParameters")
        return key in self._parameters

    def __setitem__(self, key, value):
        self.set_parameter(key, value)

    def __getitem__(self, key):
        return self.get_parameter(key)

    def get_id(self):
        return self._id

    def get_group_id(self):
        """
        channel group id
        If no group id is specified, the channel id is returned. This allows using modules that use the `group_id`
        feature also on detector setups that don't use this feature.
        """
        if self._group_id is None:
            return self._id
        else:
            return self._group_id

    def serialize(self, save_trace):
        if save_trace:
            base_trace_pkl = NuRadioReco.framework.base_trace.BaseTrace.serialize(self)
        else:
            base_trace_pkl = None

        if self._trigger_channel is not None:
            trigger_channel_pkl = self._trigger_channel.serialize(save_trace)
        else:
            trigger_channel_pkl = None

        data = {
            'parameters': NuRadioReco.framework.parameter_serialization.serialize(self._parameters),
            'id': self.get_id(),
            'group_id': self._group_id,
            'base_trace': base_trace_pkl,
            'trigger_channel_pkl': trigger_channel_pkl,
        }

        return pickle.dumps(data, protocol=4)

    def deserialize(self, data_pkl):
        data = pickle.loads(data_pkl)
        if data['base_trace'] is not None:
            NuRadioReco.framework.base_trace.BaseTrace.deserialize(self, data['base_trace'])

        self._parameters = NuRadioReco.framework.parameter_serialization.deserialize(data['parameters'], parameters.channelParameters)
        self._id = data['id']
        self._group_id = data.get('group_id')  # Attempts to load group_id, returns None if not found

        trigger_channel_pkl = data.get("trigger_channel_pkl", None)

        if trigger_channel_pkl is not None:
            trigger_channel = Channel(None)
            trigger_channel.deserialize(trigger_channel_pkl)
            self._trigger_channel = trigger_channel
        else:
            self._trigger_channel = None