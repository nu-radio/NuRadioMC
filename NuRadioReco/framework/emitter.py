import NuRadioReco.framework.parameters as parameters
import NuRadioReco.framework.parameter_serialization
import pickle

import logging
logger = logging.getLogger('Emitter')


class Emitter:

    def __init__(self, emitter_id=0, station_ids=None):
        self._id = emitter_id
        self.__station_ids = station_ids
        self._parameters = {}

    def __setitem__(self, key, value):
        self.set_parameter(key, value)

    def __getitem__(self, key):
        return self.get_parameter(key)

    def get_id(self):
        return self._id

    def get_parameters(self):
        return self._parameters

    def get_station_ids(self):
        return self.__station_ids

    def has_station_ids(self, ids):
        for station_id in ids:
            if station_id not in self.__station_ids:
                return False
        return True

    def get_parameter(self, key):
        if not isinstance(key, parameters.emitterParameters):
            logger.error(f"parameter key needs to be of type NuRadioReco.framework.parameters.emitterParameters but is {type(key)}")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.emitterParameters")
        return self._parameters[key]

    def set_parameter(self, key, value):
        if not isinstance(key, parameters.emitterParameters):
            logger.error(f"parameter key needs to be of type NuRadioReco.framework.parameters.emitterParameters but is {type(key)}")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.emitterParameters")
        self._parameters[key] = value

    def has_parameter(self, key):
        if not isinstance(key, parameters.emitterParameters):
            logger.error(f"parameter key needs to be of type NuRadioReco.framework.parameters.emitterParameters but is {type(key)}")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.emitterParameters")
        return key in self._parameters

    def serialize(self):
        data = {'_parameters': NuRadioReco.framework.parameter_serialization.serialize(self._parameters),
                'station_ids': self.__station_ids,
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
            parameters.emitterParameters
        )
        self.__station_ids = data['station_ids']
