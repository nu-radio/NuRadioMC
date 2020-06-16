from __future__ import absolute_import, division, print_function
import NuRadioReco.framework.base_trace
import NuRadioReco.framework.parameters as parameters
try:
    import cPickle as pickle
except ImportError:
    import pickle
import logging
logger = logging.getLogger('channel')


class SimChannel(NuRadioReco.framework.base_trace.BaseTrace):
    """
    Object to store simulated channels. 
    
    This class is the same as the regular channel trace but has apart from the channel id also 
    a shower and ray tracing solution id
    """

    def __init__(self, channel_id, shower_id, ray_tracing_id):
        """
        Initializes SimChannel object
        
        Parameters
        --------
        channel_id: int
            id of channel
        shower_id: int or None
            the id of the corresponding shower object
        ray_tracing_id: int or None
            the id of the corresponding ray tracing solution
        """
        NuRadioReco.framework.base_trace.BaseTrace.__init__(self)
        self._parameters = {}
        self._id = channel_id
        self._shower_id = shower_id
        self._ray_tracing_id = ray_tracing_id

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

    def get_shower_id(self):
        return self._shower_id

    def get_ray_tracing_solution_id(self):
        return self._ray_tracing_id

    def get_unique_identifier(self):
        """
        returns a unique identifier consisting of the tuple channel_id, shower_id and ray_tracing_id
        """
        return (self._id, self._shower_id, self._ray_tracing_id)

    def serialize(self, save_trace):
        if save_trace:
            base_trace_pkl = NuRadioReco.framework.base_trace.BaseTrace.serialize(self)
        else:
            base_trace_pkl = None
        data = {'parameters': NuRadioReco.framework.parameter_serialization.serialize(self._parameters),
                'unique_id': self.get_unique_identifier(),
                'base_trace': base_trace_pkl}

        return pickle.dumps(data, protocol=4)

    def deserialize(self, data_pkl):
        data = pickle.loads(data_pkl)
        if(data['base_trace'] is not None):
            NuRadioReco.framework.base_trace.BaseTrace.deserialize(self, data['base_trace'])
        self._parameters = NuRadioReco.framework.parameter_serialization.deserialize(data['parameters'], parameters.channelParameters)
        self._id = data['unique_id'][0]
        self._shower_id = data['unique_id'][1]
        self._ray_tracing_id = data['unique_id'][2]
