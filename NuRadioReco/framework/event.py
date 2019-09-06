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
        self.__modules = collections.OrderedDict()  # saves which modules were executed with what parameters
        
    def register_module(self, i, instance, name, kwargs):
        """
        registers modules applied to this event
        """
        self.__modules[i] = [name, instance, kwargs]
    
    def get_module_list(self):
        """
        returns list (actually a dictionary) of modules that have been executed on this station
        
        modules are stored in an ordered dictionary where the key is an integer specifying the order
        of module execution. This is needed because event and station modules can both be executed in arbitrary
        orders. 
        Each entry is a list of ['module name', 'module instance', 'dictionary of the kwargs of the run method']
        """
        return self.__modules
    
    def has_modules(self):
        """
        returns True if at least one module has been executed on event level so far for this event
        """
        return len(self.__modules) > 0

    def get_number_of_modules(self):
        """
        returns the numbers of modules executed on event level so far for this event
        """
        return len(self.__modules)

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
            
        modules_out = collections.OrderedDict()
        for key, value in self.__modules.items():  # remove module instances (this will just blow up the file size)
            modules_out[key] = [value[0], None, value[2]]


        data = {'_parameters': self._parameters,
                '__run_number': self.__run_number,
                '_id': self._id,
                '__event_time': self.__event_time,
                'stations': stations_pkl,
                '__modules': modules_out}
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
        if("__modules" in data):
            self.__modules = data['__modules']
