from __future__ import absolute_import, division, print_function
import NuRadioReco.framework.parameters as parameters
import NuRadioReco.framework.parameter_serialization
import pickle
import logging
logger = logging.getLogger('Shower')


class BaseShower:

    def __init__(self, shower_id=0):
        self._id = shower_id
        self._parameters = {}

    def __setitem__(self, key, value):
        self.set_parameter(key, value)

    def __getitem__(self, key):
        return self.get_parameter(key)

    def get_id(self):
        return self._id

    def get_parameter(self, key):
        if not isinstance(key, parameters.showerParameters):
            logger.error("parameter key needs to be of type NuRadioReco.framework.parameters.showerParameters")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.showerParameters")
        return self._parameters[key]

    def set_parameter(self, key, value):
        if not isinstance(key, parameters.showerParameters):
            logger.error("parameter key needs to be of type NuRadioReco.framework.parameters.showerParameters")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.showerParameters")
        self._parameters[key] = value

    def has_parameter(self, key):
        if not isinstance(key, parameters.showerParameters):
            logger.error("parameter key needs to be of type NuRadioReco.framework.parameters.showerParameters")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.showerParameters")
        return key in self._parameters

    def serialize(self):
        data = {'_parameters': NuRadioReco.framework.parameter_serialization.serialize(self._parameters),
                '_id': self._id}
        return pickle.dumps(data, protocol=4)

    def deserialize(self, data_pkl):
        data = pickle.loads(data_pkl)
        if '_id' in data.keys():
            self._id = data['_id']
        else:
            self._id = None
        self._parameters = NuRadioReco.framework.parameter_serialization.deserialize(
            data['_parameters'],
            parameters.showerParameters
        )
