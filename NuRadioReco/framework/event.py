from __future__ import absolute_import, division, print_function
import NuRadioReco.framework.station
import NuRadioReco.framework.radio_shower
import NuRadioReco.framework.emitter
import NuRadioReco.framework.sim_emitter
import NuRadioReco.framework.hybrid_information
import NuRadioReco.framework.particle
import NuRadioReco.framework.parameter_storage

from NuRadioReco.framework.parameters import (
    eventParameters as evp, channelParameters as chp, showerParameters as shp,
    particleParameters as pap, generatorAttributes as gta)

from NuRadioReco.utilities import io_utilities, version

import astropy.time
import datetime
from six import itervalues
import numpy as np
import collections
import pickle

import logging
logger = logging.getLogger('NuRadioReco.Event')


class Event(NuRadioReco.framework.parameter_storage.ParameterStorage):

    def __init__(self, run_number, event_id):
        super().__init__([evp, gta])

        self.__run_number = run_number
        self._id = event_id
        self.__stations = collections.OrderedDict()
        self.__radio_showers = collections.OrderedDict()
        self.__sim_showers = collections.OrderedDict()
        self.__sim_emitters = collections.OrderedDict()
        self.__event_time = None
        self.__particles = collections.OrderedDict() # stores a dictionary of simulated MC particles in an event
        self.__hybrid_information = NuRadioReco.framework.hybrid_information.HybridInformation()
        self.__modules_event = []  # saves which modules were executed with what parameters on event level
        # saves which modules were executed with what parameters on station level
        self.__modules_station = collections.defaultdict(list)

    def register_module_event(self, instance, name, kwargs):
        """
        registers modules applied to this event

        Parameters
        ----------
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
        ----------
        station_id: int
            the station id
        instance: module instance
            the instance of the module that should be registered
        name: module name
            the name of the module
        kwargs:
            the key word arguments of the run method
        """
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
            if (station_id in self.__modules_station and (len(self.__modules_station[station_id]) > iS)
                    and self.__modules_station[station_id][iS][0] == iE):
                iS += 1
                yield self.__modules_station[station_id][iS - 1][1:]
            else:
                if len(self.__modules_event) == iE:
                    break

                iE += 1
                yield self.__modules_event[iE - 1]

    def has_been_processed_by_module(self, module_name, station_id):
        """
        Checks if the event has been processed by a module with a specific name.

        Parameters
        ----------
        module_name: str
            The name of the module to check for.
        station_id: int
            The station id for which the module is run.

        Returns
        -------
        bool
        """
        for module in self.iter_modules(station_id):
            if module[0] == module_name:
                return True

        return False

    def get_generator_info(self, key):
        logger.warning("`get_generator_info` is deprecated. Use `get_parameter` instead.")
        return self.get_parameter(key)

    def set_generator_info(self, key, value):
        logger.warning("`set_generator_info` is deprecated. Use `set_parameter` instead.")
        self.set_parameter(key, value)

    def has_generator_info(self, key):
        logger.warning("`has_generator_info` is deprecated. Use `has_parameter` instead.")
        return self.has_parameter(key)

    def get_id(self):
        return self._id

    def set_id(self, evt_id):
        self._id = evt_id

    def get_run_number(self):
        return self.__run_number

    def get_waveforms(self, station_id=None, channel_id=None):
        """
        Returns the waveforms stored within the event.

        You can specify the station and channel id to get specific waveforms.
        If you do not specify anything you will get all waveforms.

        Parameters
        ----------
        station_id: int (Default: None)
            The station id of the station for which the waveforms should be returned.
            If `None`, the waveforms of all stations are returned.
        channel_id: int or list of ints (Default: None)
            The channel id(s) of the channel(s) for which the waveforms should be returned.
            If `None`, the waveforms of all channels are returned.

        Returns
        -------
        times: np.ndarray(nr_stations, nr_channels, nr_samples)
            A numpy array containing the times of the waveforms.
            The returned array is squeezed:
            (1, 10, 2048) -> (10, 2048) or (2, 1, 2048) -> (2, 2048).
        waveforms: np.ndarray(nr_stations, nr_channels, nr_samples)
            A numpy array containing the waveforms.
            The returned array is squeezed (see example for `times`).
        """
        times = []
        waveforms = []

        if isinstance(channel_id, int):
            channel_id = [channel_id]

        for station in self.get_stations():
            tmp_times = []
            tmp_waveforms = []
            if station_id is not None and station.get_id() != station_id:
                continue
            for channel in station.iter_channels(use_channels=channel_id, sorted=True):
                tmp_times.append(channel.get_times())
                tmp_waveforms.append(channel.get_trace())

            times.append(tmp_times)
            waveforms.append(tmp_waveforms)

        return np.squeeze(times), np.squeeze(waveforms)

    def get_station(self, station_id=None):
        """
        Returns the station for a given station id.

        Parameters
        ----------

        station_id: int
            Id of the station you want to get. If None and event has only one station
            return it, otherwise raise error. (Default: None)

        Returns
        -------

        station: NuRadioReco.framework.station
        """
        if station_id is None:
            if len(self.get_station_ids()) == 1:
                return self.__stations[self.get_station_ids()[0]]
            else:
                err = "Event has more than one station, you have to specify \"station_id\""
                logger.error(err)
                raise ValueError(err)

        return self.__stations[station_id]

    def set_event_time(self, time, format=None):
        """
        Set the (absolute) event time (will be stored as astropy.time.Time).

        Parameters
        ----------
        time: astropy.time.Time or datetime.datetime or float
            If "time" is a float, you have to specify its format.

        format: str (Default: None)
            Only used when "time" is a float. Format to interpret "time".
        """

        if isinstance(time, datetime.datetime):
            self.__event_time = astropy.time.Time(time)
        elif isinstance(time, astropy.time.Time):
            self.__event_time = time
        elif time is None:
            self.__event_time = None
        else:
            if format is None:
                logger.error("If you provide a float for the time, you have to specify the format.")
                raise ValueError("If you provide a float for the time, you have to specify the format.")
            self.__event_time = astropy.time.Time(time, format=format)

    def get_event_time(self):
        """
        Returns the event time (as astropy.time.Time object).

        If the event time is not set, an error is raised. The event time is often only used in simulations
        and typically the same a `station.get_station_time()`.

        Returns
        -------
        event_time : astropy.time.Time
            The event time.
        """
        if self.__event_time is None:
            logger.error("Event time is not set. You either have to set it or use `station.get_station_time()`")
            raise ValueError("Event time is not set. You either have to set it or use `station.get_station_time()`")

        return self.__event_time

    def get_stations(self):
        for station in itervalues(self.__stations):
            yield station

    def get_station_ids(self):
        return list(self.__stations.keys())

    def set_station(self, station):
        self.__stations[station.get_id()] = station

    def has_triggered(self, trigger_name=None):
        """
        Returns true if any station has been triggered.

        Parameters
        ----------
        trigger_name: string or None (default None)
            * if None: The function returns False if not trigger was set. If one or multiple triggers were set,
                       it returns True if any of those triggers triggered
            * if trigger name is set: return if the trigger with name 'trigger_name' has a trigger

        Returns
        -------

        has_triggered : bool
        """
        for station in self.get_stations():
            if station.has_triggered(trigger_name):
                return True

        # if it reaches this point, no station has a trigger
        return False

    def add_particle(self, particle):
        """
        Adds a MC particle to the event

        Parameters
        ----------
        particle : NuRadioReco.framework.particle.Particle
            The MC particle to be added to the event
        """
        if not isinstance(particle, NuRadioReco.framework.particle.Particle):
            logger.error("Requested to add non-Particle item to the list of particles. {particle} needs to be an instance of Particle.")
            raise TypeError("Requested to add non-Particle item to the list of particles. {particle}   needs to be an instance of Particle.")

        if particle.get_id() in self.__particles:
            logger.error("MC particle with id {particle.get_id()} already exists. Simulated particle id needs to be unique per event")
            raise AttributeError("MC particle with id {particle.get_id()} already exists. Simulated particle id needs to be unique per event")

        self.__particles[particle.get_id()] = particle

    def get_particles(self):
        """
        Returns an iterator over the MC particles stored in the event
        """
        for particle in self.__particles.values():
            yield particle

    def get_particle(self, particle_id):
        """
        returns a specific MC particle identified by its unique id
        """
        if particle_id not in self.__particles:
            raise AttributeError(f"MC particle with id {particle_id} not present")
        return self.__particles[particle_id]

    def get_primary(self):
        """
        returns a first MC particle
        """
        if len(self.__particles) == 0:
            return None

        return next(iter(self.__particles.values()))

    def get_parent(self, particle_or_shower):
        """
        returns the parent of a particle or a shower
        """
        if isinstance(particle_or_shower, NuRadioReco.framework.base_shower.BaseShower):
            par_id = particle_or_shower[shp.parent_id]
        elif isinstance(particle_or_shower, NuRadioReco.framework.particle.Particle):
            par_id = particle_or_shower[pap.parent_id]
        else:
            raise ValueError("particle_or_shower needs to be an instance of NuRadioReco.framework.base_shower.BaseShower or NuRadioReco.framework.particle.Particle")
        if par_id is None:
            logger.info("did not find parent for {particle_or_shower}")
            return None
        return self.get_particle(par_id)

    def has_particle(self, particle_id=None):
        """
        Returns true if at least one MC particle is stored in the event

        If particle_id is given, it checks if this particular MC particle exists
        """
        if particle_id is None:
            return len(self.__particles) > 0

        return particle_id in self.__particles.keys()

    def get_interaction_products(self, parent_particle, showers=True, particles=True):
        """
        Return all the daughter particles and showers generated in the interaction of the <parent_particle>

        Parameters
        ----------
        showers: bool
              Include simulated showers in the list
        showers: bool
            Include simulated particles in the list
        """

        parent_id = parent_particle.get_id()
        # iterate over sim_showers to look for parent id
        if showers is True:
            for shower in self.get_showers():
                if shower[shp.parent_id] == parent_id:
                    yield shower
        # iterate over secondary particles to look for parent id
        if particles is True:
            for particle in self.get_particles():
                if particle[pap.parent_id] == parent_id:
                    yield particle

    def add_shower(self, shower):
        """
        Adds a radio shower to the event

        Parameters
        ----------
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
        ----------
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
            return len(self.__radio_showers) > 0
        else:
            return shower_id in self.__radio_showers.keys()

    def get_first_shower(self, ids=None):
        """
        Returns only the first shower stored in the event. Useful in cases
        when there is only one shower in the event.

        Parameters
        ----------
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
        ----------
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

        Returns
        -------
        sim_showers: iterator
            An iterator over all simulated showers in the event
        """
        return self.__sim_showers.values()

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
        ----------
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

    def add_sim_emitter(self, sim_emitter):
        """
        Add a simulated emitter to the event

        Parameters
        ----------
        sim_emitter: SimEmitter object
            The emitter to be added to the event
        """
        if not isinstance(sim_emitter, NuRadioReco.framework.sim_emitter.SimEmitter):
            raise AttributeError(f"emitter needs to be of type NuRadioReco.framework.sim_emitter.SimEmitter but is of type {type(sim_emitter)}")
        if(sim_emitter.get_id() in self.__sim_emitters):
            logger.error(f"sim emitter with id {sim_emitter.get_id()} already exists. Emitter id needs to be unique per event")
            raise AttributeError(f"sim emitter with id {sim_emitter.get_id()} already exists. Emitter id needs to be unique per event")
        self.__sim_emitters[sim_emitter.get_id()] = sim_emitter

    def get_sim_emitters(self):
        """
        Get an iterator over all simulated emitters in the event
        """
        for emitter in self.__sim_emitters.values():
            yield emitter

    def get_sim_emitter(self, emitter_id):
        """
        returns a specific emitter identified by its unique id
        """
        if(emitter_id not in self.__sim_emitters):
            raise AttributeError(f"sim emitter with id {emitter_id} not present")
        return self.__sim_emitters[emitter_id]

    def get_first_sim_emitter(self, ids=None):
        """
        Returns only the first sim emitter stored in the event. Useful in cases
        when there is only one emitter in the event.

        Parameters
        ----------
        station_ids: list of integers
            A list of station IDs. The first emitter that is associated with
            all stations in the list is returned
        """
        if len(self.__sim_emitters) == 0:
            return None
        if ids is None:
            emitter_ids = list(self.__sim_emitters.keys())
            return self.__sim_emitters[emitter_ids[0]]
        for emitter in self.__sim_emitters:
            if emitter.has_station_ids(ids):
                return emitter
        return None

    def has_sim_emitter(self, emitter_id=None):
        """
        Returns true if at least one simulated emitter is stored in the event

        If emitter_id is given, it checks if this particular emitter exists
        """
        if(emitter_id is None):
            return emitter_id in self.__sim_emitters.keys()
        else:
            return len(self.__sim_emitters) > 0

    def get_hybrid_information(self):
        """
        Get information about hybrid detector data stored in the event.
        """
        return self.__hybrid_information

    def serialize(self, mode):
        stations_pkl = []
        try:
            commit_hash = version.get_NuRadioMC_commit_hash()
            self.set_parameter(evp.hash_NuRadioMC, commit_hash)
        except:
            logger.warning("Event is serialized without commit hash!")
            self.set_parameter(evp.hash_NuRadioMC, None)

        for station in self.get_stations():
            stations_pkl.append(station.serialize(mode))

        showers_pkl = [shower.serialize() for shower in self.get_showers()]
        sim_showers_pkl = [shower.serialize() for shower in self.get_sim_showers()]
        sim_emitters_pkl = [emitter.serialize() for emitter in self.get_sim_emitters()]
        particles_pkl = [particle.serialize() for particle in self.get_particles()]

        hybrid_info = self.__hybrid_information.serialize()

        modules_out_event = []
        for value in self.__modules_event:  # remove module instances (this will just blow up the file size)
            modules_out_event.append([value[0], None, value[2]])
            invalid_keys = [key for key,val in value[2].items() if isinstance(val, BaseException)]
            if len(invalid_keys):
                logger.warning(f"The following arguments to module {value[0]} could not be "
                               f"serialized and will not be stored: {invalid_keys}")

        modules_out_station = {}
        for key in self.__modules_station:  # remove module instances (this will just blow up the file size)
            modules_out_station[key] = []
            for value in self.__modules_station[key]:
                modules_out_station[key].append([value[0], value[1], None, value[3]])
                invalid_keys = [key for key,val in value[3].items() if isinstance(val, BaseException)]
                if len(invalid_keys):
                    logger.warning(f"The following arguments to module {value[0]} could not be "
                                   f"serialized and will not be stored: {invalid_keys}")

        data = NuRadioReco.framework.parameter_storage.ParameterStorage.serialize(self)

        event_time_dict = io_utilities._astropy_to_dict(self.__event_time)
        data.update({
            '__run_number': self.__run_number,
            '_id': self._id,
            '__event_time': event_time_dict,
            'stations': stations_pkl,
            'showers': showers_pkl,
            'sim_showers': sim_showers_pkl,
            'sim_emitters': sim_emitters_pkl,
            'particles': particles_pkl,
            'hybrid_info': hybrid_info,
            '__modules_event': modules_out_event,
            '__modules_station': modules_out_station
        })

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
        if 'sim_emitters' in data.keys():
            for emmitter_pkl in data['sim_emitters']:
                emitter = NuRadioReco.framework.sim_emitter.SimEmitter(None)
                emitter.deserialize(emmitter_pkl)
                self.add_sim_emitter(emitter)
        if 'particles' in data.keys():
            for particle_pkl in data['particles']:
                particle = NuRadioReco.framework.particle.Particle(None)
                particle.deserialize(particle_pkl)
                self.add_particle(particle)

        self.__hybrid_information = NuRadioReco.framework.hybrid_information.HybridInformation()
        if 'hybrid_info' in data.keys():
            self.__hybrid_information.deserialize(data['hybrid_info'])

        NuRadioReco.framework.parameter_storage.ParameterStorage.deserialize(self, data)

        self.__run_number = data['__run_number']
        self._id = data['_id']
        self.__event_time = io_utilities._time_object_to_astropy(data['__event_time'])

        # For backward compatibility, now generator_info are stored in `_parameters`.
        if 'generator_info' in data:
            for key in data['generator_info']:
                self.set_parameter(key, data['generator_info'][key])

        if "__modules_event" in data:
            self.__modules_event = data['__modules_event']
        if "__modules_station" in data:
            self.__modules_station = data['__modules_station']
