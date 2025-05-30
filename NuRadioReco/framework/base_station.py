from __future__ import absolute_import, division, print_function, unicode_literals
import NuRadioReco.framework.base_trace
import NuRadioReco.framework.trigger
import NuRadioReco.framework.electric_field
import NuRadioReco.framework.parameters as parameters
import NuRadioReco.framework.parameter_storage
from NuRadioReco.utilities import io_utilities

import datetime
import astropy.time
import logging
import collections
import pickle
from NuRadioReco.utilities.io_utilities import _dumps

logger = logging.getLogger('NuRadioReco.BaseStation')


class BaseStation(NuRadioReco.framework.parameter_storage.ParameterStorage):

    def __init__(self, station_id):
        super().__init__(
            [parameters.stationParameters, parameters.stationParametersRNOG,
             parameters.ARIANNAParameters])
        self._station_id = station_id
        self._station_time = None
        self._triggers = collections.OrderedDict()
        self._triggered = False
        self._electric_fields = []
        self._particle_type = ''

    def set_station_time(self, time, format=None):
        """
        Set the (absolute) time for the station (stored as astropy.time.Time).

        Parameters
        ----------
        time: astropy.time.Time or datetime.datetime or float
            If "time" is a float, you have to specify its format.

        format: str
            Only used when "time" is a float. Format to interpret "time". (Default: None)
        """

        if isinstance(time, datetime.datetime):
            self._station_time = astropy.time.Time(time)
        elif isinstance(time, astropy.time.Time):
            self._station_time = time
        elif time is None:
            self._station_time = None
        else:
            if format is None:
                logger.error("If you provide a float for the time, you have to specify the format.")
                raise ValueError("If you provide a float for the time, you have to specify the format.")
            self._station_time = astropy.time.Time(time, format=format)

    def get_station_time(self, format='isot'):
        """
        Returns the station time as an astropy.time.Time object

        The station time corresponds to the absolute time at which the event
        starts, i.e. all times in Channel, Trigger and ElectricField objects
        are measured relative to this time.

        Parameters
        ----------
        format: str
            Format in which the time object is displayed. (Default: isot)

        Returns
        -------

        station_time: astropy.time.Time
        """
        if self._station_time is None:
            return None

        self._station_time.format = format
        return self._station_time

    def get_id(self):
        return self._station_id

    def remove_triggers(self):
        """
        removes all triggers from the station
        """
        self._triggers = collections.OrderedDict()

    def get_trigger(self, name):
        """
        returns the trigger with the name 'name'

        Parameters
        ----------
        name: string
            the name of the trigger

        Returns
        -------
            trigger: Trigger
        """
        if name not in self._triggers:
            raise ValueError("trigger with name {} not present".format(name))
        return self._triggers[name]

    def get_primary_trigger(self):
        """
        Returns the primary trigger of the station. If no primary trigger exists, it returns None
        """
        trigger = None
        primary_trigger_count = 0
        # test if only one primary trigger exists
        for trig in self.get_triggers().values():
            if trig.is_primary():
                primary_trigger_count += 1
                trigger = trig

        if primary_trigger_count > 1:
            logger.error(
                'More than one primary trigger exists. Only one trigger can be the primary trigger. '
                'Please check your code.')
            raise ValueError

        return trigger

    def get_first_trigger(self):
        """
        Returns the first/earliest trigger. Returns None if no trigger fired.
        """
        if not self._triggered:
            return None

        min_trigger_time = float('inf')
        for trig in self._triggers.values():
            if trig.has_triggered() and trig.get_trigger_time() < min_trigger_time:
                min_trigger_time = trig.get_trigger_time()
                min_trig = trig

        return min_trig

    def has_trigger(self, trigger_name):
        """
        Checks if station has a trigger with a certain name.
        WARNING: This function does not check if the trigger has triggered.

        Parameters
        ----------
        trigger_name: string
            the name of the trigger

        Returns bool
        """
        return trigger_name in self._triggers

    def get_triggers(self):
        """
        Returns a dictionary of the triggers. key is the trigger name, value is a trigger object
        """
        return self._triggers

    def set_trigger(self, trigger):
        """
        sets a trigger for the station. If a trigger with the same name already exists, it will be overridden

        Parameters
        ----------
        trigger: Trigger
            the trigger object to set
        """
        if trigger.get_name() in self._triggers:
            logger.warning(
                f"Station has already a trigger with name {trigger.get_name()}. The previous trigger will be overridden!")

        self._triggers[trigger.get_name()] = trigger
        self._triggered = trigger.has_triggered() or self._triggered

    def has_triggered(self, trigger_name=None):
        """
        Checks if the station has triggered. If trigger_name is set, check if the trigger with that name has triggered.

        Parameters
        ----------
        trigger_name: string or None (default None)
            * if None: The function returns False if not trigger was set. If one or multiple triggers were set,
                       it returns True if any of those triggers triggered
            * if trigger name is set: return if the trigger with name 'trigger_name' has a trigger
        """
        if trigger_name is None:
            return self._triggered
        else:
            return self.get_trigger(trigger_name).has_triggered()

    def set_triggered(self, triggered=True):
        """
        Convenience function to set a simple trigger. The recommended interface is to set triggers through the
        set_trigger() interface.
        """
        if len(self._triggers) > 1:
            raise ValueError("more then one trigger were set. Request is ambiguous")
        trigger = NuRadioReco.framework.trigger.Trigger('default')
        trigger.set_triggered(triggered)
        self.set_trigger(trigger)

    def set_electric_fields(self, electric_fields):
        self._electric_fields = electric_fields

    def get_electric_fields(self):
        return self._electric_fields

    def get_electric_field_ids(self):
        """
        returns a sorted list with the electric field IDs of all simElectricFields of the simStation

        Returns
        -------
        efield_ids: list
        """
        efield_ids = []
        for efield in self._electric_fields:
            efield_ids.append(efield.get_unique_identifier())
        efield_ids.sort()
        return efield_ids

    def add_electric_field(self, electric_field):
        self._electric_fields.append(electric_field)

    def get_electric_fields_for_channels(self, channel_ids=None, ray_path_type=None):
        for e_field in self._electric_fields:
            channel_ids2 = channel_ids
            if channel_ids is None:
                channel_ids2 = e_field.get_channel_ids()
            if e_field.has_channel_ids(channel_ids2):
                if ray_path_type is None:
                    yield e_field
                elif ray_path_type == e_field.get_parameter(parameters.electricFieldParameters.ray_path_type):
                    yield e_field

    def is_neutrino(self):
        if self._particle_type == '':
            msg = "Stations particle type has not been set. Please call the module `eventTypeIdentifier.run(event, station, mode='forced', forced_event_type='neutrino'/'cosmic_ray')`."
            msg += " This flag is used to differentiate between signals that originate from air vs. signals that originate from within the ice which is needed to e.g. determine if refraction "
            msg += "into the ice needs to be considered."
            logger.error(msg)
            raise ValueError(msg)

        return self._particle_type == 'nu'

    def is_cosmic_ray(self):
        if self._particle_type == '':
            msg = "Stations particle type has not been set. Please call the module `eventTypeIdentifier.run(event, station, mode='forced', forced_event_type='neutrino'/'cosmic_ray')`."
            msg += " This flag is used to differentiate between signals that originate from air vs. signals that originate from within the ice which is needed to e.g. determine if refraction "
            msg += "into the ice needs to be considered."
            logger.error(msg)
            raise ValueError(msg)

        return self._particle_type == 'cr'

    def set_is_neutrino(self):
        """
        set station type to neutrino
        """
        self._particle_type = 'nu'

    def set_is_cosmic_ray(self):
        """
        set station type to cosmic rays (relevant e.g. for refraction into the snow)
        """
        self._particle_type = 'cr'

    def serialize(self, save_efield_traces):
        trigger_pkls = []
        for trigger in self._triggers.values():
            trigger_pkls.append(trigger.serialize())

        efield_pkls = []
        for efield in self.get_electric_fields():
            efield_pkls.append(efield.serialize(save_trace=save_efield_traces))

        station_time_dict = io_utilities._astropy_to_dict(self.get_station_time())

        data = NuRadioReco.framework.parameter_storage.ParameterStorage.serialize(self)
        data.update({
            '_station_id': self._station_id,
            '_station_time': station_time_dict,
            '_particle_type': self._particle_type,
            'triggers': trigger_pkls,
            '_triggered': self._triggered,
            'electric_fields': efield_pkls
        })

        return _dumps(data, protocol=4)

    def deserialize(self, data_pkl):
        data = pickle.loads(data_pkl)
        NuRadioReco.framework.parameter_storage.ParameterStorage.deserialize(self, data)

        if 'triggers' in data:
            self._triggers = NuRadioReco.framework.trigger.deserialize(data['triggers'])

        if 'triggers' in data:
            self._triggered = data['_triggered']

        for electric_field in data['electric_fields']:
            efield = NuRadioReco.framework.electric_field.ElectricField([])
            efield.deserialize(electric_field)
            self.add_electric_field(efield)

        # For backward compatibility, now ARIANNA parameters are stored in `_parameters`.
        if '_ARIANNA_parameters' in data:
            for key in data['_ARIANNA_parameters']:
                self.set_parameter(key, data['_ARIANNA_parameters'][key])

        self._station_id = data['_station_id']
        if data['_station_time'] is not None:
            station_time = io_utilities._time_object_to_astropy(data['_station_time'])
            self.set_station_time(station_time)

        self._particle_type = data['_particle_type']


    def __add__(self, x):
        """
        adds a BaseStation object to another BaseStation object
        WARNING: Only channel and efield objects are added but no other meta information

        Parameters
        ----------
        x: BaseStation
            the BaseStation object to add
        """
        if not isinstance(x, BaseStation):
            raise AttributeError("Can only add BaseStation to BaseStation")

        if self.get_id() != x.get_id():
            raise AttributeError("Can only add BaseStations with the same ID")

        for trigger in x.get_triggers().values():
            self.set_trigger(trigger)

        for efield in x.get_electric_fields():
            self.add_electric_field(efield)

        for key, value in x.get_parameters().items():
            self.set_parameter(key, value)

        return self


    ######## Deprecated functions ########

    def get_ARIANNA_parameter(self, key):
        logger.warning("`get_ARIANNA_parameter` is deprecated. Use `get_parameter` instead.")
        raise NotImplementedError("`get_ARIANNA_parameter` is deprecated. Use `get_parameter` instead.")

    def get_ARIANNA_parameters(self):
        logger.warning("`get_ARIANNA_parameters` is deprecated. Use `get_parameters` instead.")
        raise NotImplementedError("`get_ARIANNA_parameters` is deprecated. Use `get_parameters` instead.")

    def has_ARIANNA_parameter(self, key):
        logger.warning("`has_ARIANNA_parameter` is deprecated. Use `has_parameter` instead.")
        raise NotImplementedError("`has_ARIANNA_parameter` is deprecated. Use `has_parameter` instead.")

    def set_ARIANNA_parameter(self, key, value):
        logger.warning("`set_ARIANNA_parameter` is deprecated. Use `set_parameter` instead.")
        raise NotImplementedError("`set_ARIANNA_parameter` is deprecated. Use `set_parameter` instead.")
