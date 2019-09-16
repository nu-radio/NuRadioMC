from __future__ import absolute_import, division, print_function
import NuRadioReco.framework.base_trace
import NuRadioReco.framework.parameters as parameters
try:
    import cPickle as pickle
except ImportError:
    import pickle
import logging
logger = logging.getLogger('electric_field')

class ElectricField(NuRadioReco.framework.base_trace.BaseTrace):

    def __init__(self, channel_ids, position=[0,0,0]):
        NuRadioReco.framework.base_trace.BaseTrace.__init__(self)
        self._channel_ids = channel_ids
        self._parameters = {}
        self._parameter_covariances = {}
        self._position = position

    def get_parameter(self, key):
        if not isinstance(key, parameters.electricFieldParameters):
            logger.error("parameter key needs to be of type NuRadioReco.framework.parameters.electricFieldParameters")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.electricFieldParameters")
        return self._parameters[key]

    def get_parameters(self):
        return self._parameters

    def set_parameter(self, key, value):
        if not isinstance(key, parameters.electricFieldParameters):
            logger.error("parameter key needs to be of type NuRadioReco.framework.parameters.electricFieldParameters")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.electricFieldParameters")
        self._parameters[key] = value

    def has_parameter(self, key):
        if not isinstance(key, parameters.electricFieldParameters):
            logger.error("parameter key needs to be of type NuRadioReco.framework.parameters.electricFieldParameters")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.electricFieldParameters")
        return key in self._parameters

    def has_parameter_error(self, key):
        if not isinstance(key, parameters.electricFieldParameters):
            logger.error("parameter key needs to be of type NuRadioReco.framework.parameters.electricFieldParameters")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.electricFieldParameters")
        return (key, key) in self._parameter_covariances

    def set_parameter_error(self, key, value):
        if not isinstance(key, parameters.electricFieldParameters):
            logger.error("parameter key needs to be of type NuRadioReco.framework.parameters.electricFieldParameters")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.electricFieldParameters")
        self._parameter_covariances[(key, key)] = value ** 2

    def get_parameter_error(self, key):
        if not isinstance(key, parameters.electricFieldParameters):
            logger.error("parameter key needs to be of type NuRadioReco.framework.parameters.electricFieldParameters")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.electricFieldParameters")
        return self._parameter_covariances[(key, key)] ** 0.5

    def __setitem__(self, key, value):
        self.set_parameter(key, value)

    def __getitem__(self, key):
        return self.get_parameter(key)

    def set_channel_ids(self, channel_ids):
        self._channel_ids = channel_ids

    def get_channel_ids(self):
        return self._channel_ids

    def has_channel_ids(self, channel_ids):
        for channel_id in channel_ids:
            if channel_id not in self._channel_ids:
                return False
        return True

    def get_position(self):
        """
        get position of the electric field relative to station position
        """
        return self._position

    def set_position(self, position):
        """
        set position of the electric field relative to station position
        """
        self._position = position

    def serialize(self, mode):
        if(mode == 'micro'):
            base_trace_pkl = None
        else:
            base_trace_pkl = NuRadioReco.framework.base_trace.BaseTrace.serialize(self)
        data = {'parameters': NuRadioReco.framework.parameter_serialization.serialize(self._parameters),
                'channel_ids': self._channel_ids,
                'position': self._position,
                'base_trace': base_trace_pkl}
        return pickle.dumps(data, protocol=4)

    def deserialize(self, data_pkl):
        data = pickle.loads(data_pkl)
        if(data['base_trace'] is not None):
            NuRadioReco.framework.base_trace.BaseTrace.deserialize(self, data['base_trace'])
        if 'position' in data:  #for backward compatibility
            self._position = data['position']
        self._parameters = NuRadioReco.framework.parameter_serialization.deserialize(data['parameters'], parameters.electricFieldParameters)
        self._channel_ids = data['channel_ids']
