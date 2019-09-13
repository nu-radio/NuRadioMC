from __future__ import absolute_import, division, print_function
try:
    import cPickle as pickle
except ImportError:
    import pickle
import NuRadioReco.framework.station
from six import itervalues
import collections
import logging
logger = logging.getLogger('Event')


class Event:

    def __init__(self, run_number, event_id):
        self._parameters = {}
        self.__run_number = run_number
        self._id = event_id
        self.__stations = collections.OrderedDict()
        self.__event_time = 0
        self.__modules_event = []  # saves which modules were executed with what parameters on event level
        self.__modules_station = {}  # saves which modules were executed with what parameters on station level

    def register_module_event(self, instance, name, kwargs):
        """
        registers modules applied to this event
        """

        self.__modules_event.append([name, instance, kwargs])

    def register_module_station(self, station_id, instance, name, kwargs):
        """
        registers modules applied to this event
        """
        if(station_id not in self.__modules_station):
            self.__modules_station[station_id] = []
        iE = len(self.__modules_event)
        self.__modules_station[station_id].append([iE, name, instance, kwargs])

    def iter_modules(self, station_id=None):
        """
        returns an interator that loops over all modules. If a station id is provided it loops
        over all modules that are applied on event or station level (on this particular station). If no 
        station_id is provided, the loop is only over the event modules. 
        The order follows the sequence these modules were applied
        """
        iE = 0
        iS = 0
        while True:
            if(station_id in self.__modules_station and (len(self.__modules_station[station_id]) > iS) and self.__modules_station[station_id][iS][0] == iE):
                iS += 1
                yield self.__modules_station[station_id][iS - 1][1:]
            else:
                if(len(self.__modules_event) == iE):
                    break
                iE += 1
                yield self.__modules_event[iE - 1]

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

        modules_out_event = []
        for value in self.__modules_event:  # remove module instances (this will just blow up the file size)
            modules_out_event.append([value[0], None, value[2]])

        modules_out_station = {}
        for key in self.__modules_station:  # remove module instances (this will just blow up the file size)
            modules_out_station[key] = []
            for value in self.__modules_station[key]:
                modules_out_station[key].append([value[0], value[1], None, value[3]])

        data = {'_parameters': self._parameters,
                '__run_number': self.__run_number,
                '_id': self._id,
                '__event_time': self.__event_time,
                'stations': stations_pkl,
                '__modules_event': modules_out_event,
                '__modules_station': modules_out_station
                }
        return pickle.dumps(data, protocol=4)

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
        if("__modules_event" in data):
            self.__modules_event = data['__modules_event']
        if("__modules_station" in data):
            self.__modules_station = data['__modules_station']
