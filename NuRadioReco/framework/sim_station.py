from __future__ import absolute_import, division, print_function
import NuRadioReco.framework.base_station
from six import iteritems
try:
    import cPickle as pickle
except ImportError:
    import pickle
import logging
logger = logging.getLogger('SimStation')


class SimStation(NuRadioReco.framework.base_station.BaseStation):

    def __init__(self, station_id, sampling_rate=None, trace=None, position=None):
        NuRadioReco.framework.base_station.BaseStation.__init__(self, station_id)
        self.__position = position
        self.__magnetic_field_vector = None
        self.__simulation_weight = None
        if(trace is not None and sampling_rate is not None):
            self.set_electric_fields(trace, sampling_rate)

    def get_position(self):
        return self.__position

    def get_magnetic_field_vector(self):
        return self.__magnetic_field_vector

    def set_magnetic_field_vector(self, magnetic_field_vector):
        self.__magnetic_field_vector = magnetic_field_vector

    def get_simulation_weight(self):
        return self.__simulation_weight

    def set_simulation_weight(self, simulation_weight):
        self.__simulation_weight = simulation_weight

    def serialize(self, mode):
        base_station_pkl = NuRadioReco.framework.base_station.BaseStation.serialize(self, mode)
        data = {'__magnetic_field_vector': self.__magnetic_field_vector,
                '__simulation_weight': self.__simulation_weight,
                '__position': self.__position,
                'base_station': base_station_pkl}
        return pickle.dumps(data, protocol=2)

    def deserialize(self, data_pkl):
        data = pickle.loads(data_pkl)
        NuRadioReco.framework.base_station.BaseStation.deserialize(self, data['base_station'])
        self.__magnetic_field_vector = data['__magnetic_field_vector']
        self.__simulation_weight = data['__simulation_weight']
        self.__position = data['__position']
