from __future__ import absolute_import, division, print_function
try:
    import cPickle as pickle
except ImportError:
    import pickle
import NuRadioReco.framework.station
from six import itervalues
import logging
logger = logging.getLogger('Event')


class Event:

    def __init__(self, run_number, event_id):
        self._parameters = {}
        self.__run_number = run_number
        self._id = event_id
        self.__stations = {}
        self.__event_time = 0

    def get_parameter(self, attribute):
        return self._parameters[attribute]

    def set_parameter(self, key, value):
        self._parameters[key] = value

    def get_id(self):
        return self._id

    def set_id(self, evt_id):
        self._id = evt_id

    def get_run_number(self):
        return self.__run_number

    def get_station(self, station_id):
        return self.__stations[station_id]

    def get_stations(self):
        for station in itervalues(self.__stations):
            yield station

    def set_station(self, station):
        self.__stations[station.get_id()] = station

    def serialize(self, mode):
        stations_pkl = []
        for station in self.get_stations():
            stations_pkl.append(station.serialize(mode))

        data = {'_parameters': self._parameters,
                '__run_number': self.__run_number,
                '_id': self._id,
                '__event_time': self.__event_time,
                'stations': stations_pkl}
        return pickle.dumps(data, protocol=2)

    def deserialize(self, data_pkl):
        data = pickle.loads(data_pkl)
        for station_pkl in data['stations']:
            station = NuRadioReco.framework.station.Station(0)
            station.deserialize(station_pkl)
            self.set_station(station)
        self._parameters = data['_parameters']
        self.__run_number = data['__run_number']
        self._id = data['_id']
        self.__event_time = data['__event_time']
