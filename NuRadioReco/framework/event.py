from __future__ import absolute_import, division, print_function
import pickle
import NuRadioReco.framework.station
import NuRadioReco.framework.radio_shower
import NuRadioReco.framework.hybrid_information
import NuRadioReco.framework.parameters as parameters
import NuRadioReco.utilities.version
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
        self.__radio_showers = collections.OrderedDict()
        self.__sim_showers = collections.OrderedDict()
        self.__event_time = 0
        self.__hybrid_information = NuRadioReco.framework.hybrid_information.HybridInformation()
        self.__modules_event = []  # saves which modules were executed with what parameters on event level
        self.__modules_station = {}  # saves which modules were executed with what parameters on station level

    def register_module_event(self, instance, name, kwargs):
        """
        registers modules applied to this event

        Parameters
        -----------
        instance: module instance
            the instance of the module that should be registered
        name: module name
            the name of the module
        kwargs:
            the key word arguments of the run method
        """

        self.__modules_event.append([name, instance, kwargs])

    def register_module_station(self, station_id, instance, name, kwargs):
        """
        registers modules applied to this event

        Parameters
        -----------
        station_id: int
            the station id
        instance: module instance
            the instance of the module that should be registered
        name: module name
            the name of the module
        kwargs:
            the key word arguments of the run method
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

    def get_parameter(self, key):
        if not isinstance(key, parameters.eventParameters):
            logger.error("parameter key needs to be of type NuRadioReco.framework.parameters.eventParameters")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.eventParameters")
        return self._parameters[key]

    def set_parameter(self, key, value):
        if not isinstance(key, parameters.eventParameters):
            logger.error("parameter key needs to be of type NuRadioReco.framework.parameters.eventParameters")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.eventParameters")
        self._parameters[key] = value

    def has_parameter(self, key):
        if not isinstance(key, parameters.eventParameters):
            logger.error("parameter key needs to be of type NuRadioReco.framework.parameters.eventParameters")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.eventParameters")
        return key in self._parameters

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
        """
        Adds a radio shower to the event

        Parameters
        ------------------------
        shower: RadioShower object
            The shower to be added to the event
        """
        if(shower.get_id() in self.__radio_showers):
            logger.error("shower with id {shower.get_id()} already exists. Shower id needs to be unique per event")
            raise AttributeError("shower with id {shower.get_id()} already exists. Shower id needs to be unique per event")
        self.__radio_showers[shower.get_id()] = shower

    def get_showers(self, ids=None):
        """
        Returns an iterator over the showers stored in the event

        Parameters
        ---------------------------
        ids: list of integers
            A list of station IDs. Only showers that are associated with
            all stations in the list are returned
        """
        for shower in self.__radio_showers.values():
            if ids is None:
                yield shower
            elif shower.has_station_ids(ids):
                yield shower

    def get_shower(self, shower_id):
        """
        returns a specific shower identified by its unique id
        """
        if(shower_id not in self.__radio_showers):
            raise AttributeError(f"shower with id {shower_id} not present")
        return self.__radio_showers[shower_id]

    def has_shower(self, shower_id=None):
        """
        Returns true if at least one shower is stored in the event

        If shower_id is given, it checks if this particular shower exists
        """
        if(shower_id is None):
            return shower_id in self.__radio_showers.keys()
        else:
            return len(self.__radio_showers) > 0

    def get_first_shower(self, ids=None):
        """
        Returns only the first shower stored in the event. Useful in cases
        when there is only one shower in the event.

        Parameters
        ---------------------------
        ids: list of integers
            A list of station IDs. The first shower that is associated with
            all stations in the list is returned
        """
        if len(self.__radio_showers) == 0:
            return None
        if ids is None:
            shower_ids = list(self.__radio_showers.keys())
            return self.__radio_showers[shower_ids[0]]
        for shower in self.__radio_showers:
            if shower.has_station_ids(ids):
                return shower
        return None

    def add_sim_shower(self, sim_shower):
        """
        Add a simulated shower to the event

        Parameters
        ------------------------
        sim_shower: RadioShower object
            The shower to be added to the event
        """
        if not isinstance(sim_shower, NuRadioReco.framework.radio_shower.RadioShower):
            raise AttributeError("sim_shower needs to be of type NuRadioReco.framework.radio_shower.RadioShower")
        if(sim_shower.get_id() in self.__sim_showers):
            logger.error(f"sim shower with id {sim_shower.get_id()} already exists. Shower id needs to be unique per event")
            raise AttributeError(f"sim shower with id {sim_shower.get_id()} already exists. Shower id needs to be unique per event")
        self.__sim_showers[sim_shower.get_id()] = sim_shower

    def get_sim_showers(self):
        """
        Get an iterator over all simulated showers in the event
        """
        for shower in self.__sim_showers.values():
            yield shower

    def get_sim_shower(self, shower_id):
        """
        returns a specific shower identified by its unique id
        """
        if(shower_id not in self.__sim_showers):
            raise AttributeError(f"sim shower with id {shower_id} not present")
        return self.__sim_showers[shower_id]

    def get_first_sim_shower(self, ids=None):
        """
        Returns only the first sim shower stored in the event. Useful in cases
        when there is only one shower in the event.

        Parameters
        ---------------------------
        ids: list of integers
            A list of station IDs. The first shower that is associated with
            all stations in the list is returned
        """
        if len(self.__sim_showers) == 0:
            return None
        if ids is None:
            shower_ids = list(self.__sim_showers.keys())
            return self.__sim_showers[shower_ids[0]]
        for shower in self.__sim_showers:
            if shower.has_station_ids(ids):
                return shower
        return None

    def has_sim_shower(self, shower_id=None):
        """
        Returns true if at least one simulated shower is stored in the event

        If shower_id is given, it checks if this particular shower exists
        """
        if(shower_id is None):
            return shower_id in self.__sim_showers.keys()
        else:
            return len(self.__sim_showers) > 0

    def get_hybrid_information(self):
        """
        Get information about hybrid detector data stored in the event.
        """
        return self.__hybrid_information

    def serialize(self, mode):
        stations_pkl = []
        try:
            commit_hash = NuRadioReco.utilities.version.get_NuRadioReco_commit_hash()
            self.set_parameter(parameters.eventParameters.hash_NuRadioReco, commit_hash)
        except:
            self.set_parameter(parameters.eventParameters.hash_NuRadioReco, None)
        try:
            commit_hash = NuRadioReco.utilities.version.get_NuRadioMC_commit_hash()
            self.set_parameter(parameters.eventParameters.hash_NuRadioMC, commit_hash)
        except:
            self.set_parameter(parameters.eventParameters.hash_NuRadioMC, None)

        for station in self.get_stations():
            stations_pkl.append(station.serialize(mode))

        showers_pkl = []
        for shower in self.get_showers():
            showers_pkl.append(shower.serialize())
        sim_showers_pkl = []
        for shower in self.get_sim_showers():
            sim_showers_pkl.append(shower.serialize())
        hybrid_info = self.__hybrid_information.serialize()
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
                'showers': showers_pkl,
                'sim_showers': sim_showers_pkl,
                'hybrid_info': hybrid_info,
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
        if 'showers' in data.keys():
            for shower_pkl in data['showers']:
                shower = NuRadioReco.framework.radio_shower.RadioShower(None)
                shower.deserialize(shower_pkl)
                self.add_shower(shower)
        if 'sim_showers' in data.keys():
            for shower_pkl in data['sim_showers']:
                shower = NuRadioReco.framework.radio_shower.RadioShower(None)
                shower.deserialize(shower_pkl)
                self.add_sim_shower(shower)
        self.__hybrid_information = NuRadioReco.framework.hybrid_information.HybridInformation()
        if 'hybrid_info' in data.keys():
            self.__hybrid_information.deserialize(data['hybrid_info'])
        self._parameters = data['_parameters']
        self.__run_number = data['__run_number']
        self._id = data['_id']
        self.__event_time = data['__event_time']
        if("__modules_event" in data):
            self.__modules_event = data['__modules_event']
        if("__modules_station" in data):
            self.__modules_station = data['__modules_station']
