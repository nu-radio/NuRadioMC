import NuRadioReco.framework.parameters as parameters
import NuRadioReco.framework.parameter_storage
import pickle
from NuRadioReco.utilities.io_utilities import _dumps

import logging
logger = logging.getLogger('NuRadioReco.Emitter')


class Emitter(NuRadioReco.framework.parameter_storage.ParameterStorage):

    def __init__(self, emitter_id=0, station_ids=None):
        super().__init__(parameters.emitterParameters)
        self._id = emitter_id
        self.__station_ids = station_ids

    def get_id(self):
        return self._id

    def get_station_ids(self):
        return self.__station_ids

    def has_station_ids(self, ids):
        for station_id in ids:
            if station_id not in self.__station_ids:
                return False

        return True

    def serialize(self):
        data = NuRadioReco.framework.parameter_storage.ParameterStorage.serialize(self)
        data.update({
            'station_ids': self.__station_ids,
            '_id': self._id
        })

        return _dumps(data, protocol=4)

    def deserialize(self, data_pkl):
        data = pickle.loads(data_pkl)
        NuRadioReco.framework.parameter_storage.ParameterStorage.deserialize(self, data)
        self._id = data.get('_id', None)
        self.__station_ids = data['station_ids']
