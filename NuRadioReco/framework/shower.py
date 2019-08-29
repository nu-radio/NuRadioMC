from __future__ import absolute_import, division, print_function
from NuRadioReco.framework.parameters import showerTypes

try:
    import cPickle as pickle
except ImportError:
    import pickle
import logging
logger = logging.getLogger('Shower')


class Shower:

    def __init__(self, shower_type):
        self._parameters = {}

        # not None request for event deserialization
        if not isinstance(shower_type, showerTypes) and shower_type is not None:
            logger.error("Invalid shower type")
            raise ValueError("Invalid shower type")

        self._shower_type = shower_type

    def get_parameter(self, attribute):
        return self._parameters[attribute]

    def set_parameter(self, key, value):
        self._parameters[key] = value

    def has_parameter(self, key):
        return key in self._parameters

    def get_shower_type(self):
        return self._shower_type

    def serialize(self, mode):
        data = {'_parameters': self._parameters,
                '_shower_type': self._shower_type}

        return pickle.dumps(data, protocol=2)

    def deserialize(self, data_pkl):
        data = pickle.loads(data_pkl)
        self._parameters = data['_parameters']
        self._shower_type = data['_shower_type']
