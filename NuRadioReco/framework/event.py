from __future__ import absolute_import, division, print_function
import pickle
import NuRadioReco.framework.station
import NuRadioReco.framework.radio_shower
import NuRadioreco.framework.hybrid_information
from six import itervalues
import logging
logger = logging.getLogger('Event')


class Event:

    def __init__(self, run_number, event_id):
        self._parameters = {}
        self.__run_number = run_number
        self._id = event_id
        self.__stations = {}
        self.__radio_showers = []
        self.__event_time = 0
        self.__hybrid_information = NuRadioReco.framework.hybrid_information.HybridInformation()

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

    def add_shower(self, shower):
        self.__radio_showers.append(shower)

    def get_showers(self, ids=None):
        for shower in self.__radio_showers:
            if ids is None:
                yield shower
            elif shower.has_station_ids(ids):
                yield shower

    def serialize(self, mode):
        stations_pkl = []
        for station in self.get_stations():
            stations_pkl.append(station.serialize(mode))

        showers_pkl = []
        for shower in self.get_showers():
            showers_pkl.append(shower.serialize(mode))

        data = {'_parameters': self._parameters,
                '__run_number': self.__run_number,
                '_id': self._id,
                '__event_time': self.__event_time,
                'stations': stations_pkl,
                'showers': showers_pkl}

        return pickle.dumps(data, protocol=2)

    def deserialize(self, data_pkl):
        data = pickle.loads(data_pkl)

        for station_pkl in data['stations']:
            station = NuRadioReco.framework.station.Station(0)
            station.deserialize(station_pkl)
            self.set_station(station)

        for shower_pkl in data['showers']:
            shower = NuRadioReco.framework.shower.Shower(None)
            shower.deserialize(shower_pkl)
            self.set_shower(shower)

        self._parameters = data['_parameters']
        self.__run_number = data['__run_number']
        self._id = data['_id']
        self.__event_time = data['__event_time']
