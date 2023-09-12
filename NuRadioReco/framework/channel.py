from __future__ import absolute_import, division, print_function
import NuRadioReco.framework.base_trace
import NuRadioReco.framework.parameters as parameters
import NuRadioReco.framework.parameter_serialization
try:
    import cPickle as pickle
except ImportError:
    import pickle
import logging
logger = logging.getLogger('channel')


class Channel(NuRadioReco.framework.base_trace.BaseTrace):

    def __init__(self, channel_id, channel_group_id=None):
        """
        Parameters
        ----------
        channel_id: int
            the id of the channel
        channel_group_id: int (default None)
            optionally, severl channels can belong to a "channel group". Use case is to identify
            the channels of a single dual or triple polarized antenna as common in air shower arrays. 
        
        """
        NuRadioReco.framework.base_trace.BaseTrace.__init__(self)
        self._parameters = {}
        self._id = channel_id
        self._group_id = channel_group_id

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
        If no group id is specified, the channel id is returned. This allows using modules that use the `group_id` feature
        also on detector setups that don't use this feature. 
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
        data = {'parameters': NuRadioReco.framework.parameter_serialization.serialize(self._parameters),
                'id': self.get_id(),
                'group_id': self._group_id,
                'base_trace': base_trace_pkl}

        return pickle.dumps(data, protocol=4)

    def deserialize(self, data_pkl):
        data = pickle.loads(data_pkl)
        if(data['base_trace'] is not None):
            NuRadioReco.framework.base_trace.BaseTrace.deserialize(self, data['base_trace'])
        self._parameters = NuRadioReco.framework.parameter_serialization.deserialize(data['parameters'], parameters.channelParameters)
        self._id = data['id']
        self._group_id = data.get('group_id')  # Attempts to load group_id, returns None if not found
