from __future__ import absolute_import, division, print_function, unicode_literals
import NuRadioReco.framework.base_trace
import NuRadioReco.framework.trigger
import NuRadioReco.framework.electric_field
import NuRadioReco.framework.parameters as parameters
import datetime
import astropy.time
import NuRadioReco.framework.parameter_serialization

try:
    import cPickle as pickle
except ImportError:
    import pickle
import logging
import collections

logger = logging.getLogger('NuRadioReco.BaseStation')


class BaseStation():

    def __init__(self, station_id):
        self._parameters = {}
        self._ARIANNA_parameters = {}
        self._parameter_covariances = {}
        self._station_id = station_id
        self._station_time = None
        self._triggers = collections.OrderedDict()
        self._triggered = False
        self._electric_fields = []
        self._particle_type = ''

    def __setitem__(self, key, value):
        self.set_parameter(key, value)

    def __getitem__(self, key):
        return self.get_parameter(key)

    def get_parameter(self, key):
        if not isinstance(key, parameters.stationParameters):
            logger.error("parameter key needs to be of type NuRadioReco.framework.parameters.stationParameters")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.stationParameters")
        return self._parameters[key]

    def get_parameters(self):
        return self._parameters

    def has_parameter(self, key):
        if not isinstance(key, parameters.stationParameters):
            logger.error("parameter key needs to be of type NuRadioReco.framework.parameters.stationParameters")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.stationParameters")
        return key in self._parameters.keys()

    def set_parameter(self, key, value):
        if not isinstance(key, parameters.stationParameters):
            logger.error("parameter key needs to be of type NuRadioReco.framework.parameters.stationParameters")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.stationParameters")
        self._parameters[key] = value

    def set_parameter_error(self, key, value):
        if not isinstance(key, parameters.stationParameters):
            logger.error("parameter key needs to be of type NuRadioReco.framework.parameters.stationParameters")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.stationParameters")
        self._parameter_covariances[(key, key)] = value ** 2

    def get_parameter_error(self, key):
        if not isinstance(key, parameters.stationParameters):
            logger.error("parameter key needs to be of type NuRadioReco.framework.parameters.stationParameters")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.stationParameters")
        return self._parameter_covariances[(key, key)] ** 0.5

    def remove_parameter(self, key):
        if not isinstance(key, parameters.stationParameters):
            logger.error("parameter key needs to be of type NuRadioReco.framework.parameters.stationParameters")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.stationParameters")
        self._parameters.pop(key, None)

    def set_station_time(self, time, format=None):
        """
        Set the (absolute) time for the station (stored as astropy.time.Time).
        Not related to the event._event_time.

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
            self._station_time = astropy.time.Time(time, format=format)

    def get_station_time(self, format='isot'):
        """
        Returns a astropy.time.Time object

        Parameters
        ----------

        format: str
            Format in which the time object is displayed. (Default: isot)

        Returns
        -------

        _station_time: astropy.time.Time
        """
        if self._station_time is None:
            return None

        self._station_time.format = format
        return self._station_time

    def get_station_time_dict(self):
        """ Return the station time as dict {value, format}. Used for reading and writing """
        if self._station_time is None:
            return None
        else:
            return {'value': self._station_time.value, 'format': self._station_time.format}

    def get_id(self):
        return self._station_id

    def remove_triggers(self):
        self._triggers = collections.OrderedDict()

    def get_trigger(self, name):
        if name not in self._triggers:
            raise ValueError("trigger with name {} not present".format(name))
        return self._triggers[name]

    def get_first_trigger(self):
        """
        Returns the first trigger. Returns None if no trigger is present.
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

    # provide interface to ARIANNA specific parameters
    def get_ARIANNA_parameter(self, key):
        if not isinstance(key, parameters.ARIANNAParameters):
            logger.error("parameter key needs to be of type NuRadioReco.framework.parameters.ARIANNAParameters")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.ARIANNAParameters")
        return self._ARIANNA_parameters[key]

    def get_ARIANNA_parameters(self):
        return self._ARIANNA_parameters

    def has_ARIANNA_parameter(self, key):
        if not isinstance(key, parameters.ARIANNAParameters):
            logger.error("parameter key needs to be of type NuRadioReco.framework.parameters.ARIANNAParameters")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.ARIANNAParameters")
        return key in self._ARIANNA_parameters.keys()

    def set_ARIANNA_parameter(self, key, value):
        if not isinstance(key, parameters.ARIANNAParameters):
            logger.error("parameter key needs to be of type NuRadioReco.framework.parameters.ARIANNAParameters")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.ARIANNAParameters")
        self._ARIANNA_parameters[key] = value

    def serialize(self, save_efield_traces):
        trigger_pkls = []
        for trigger in self._triggers.values():
            trigger_pkls.append(trigger.serialize())

        efield_pkls = []
        for efield in self.get_electric_fields():
            efield_pkls.append(efield.serialize(save_trace=save_efield_traces))

        station_time_dict = self.get_station_time_dict()

        data = {'_parameters': NuRadioReco.framework.parameter_serialization.serialize(self._parameters),
                '_parameter_covariances': NuRadioReco.framework.parameter_serialization.serialize_covariances(self._parameter_covariances),
                '_ARIANNA_parameters': self._ARIANNA_parameters,
                '_station_id': self._station_id,
                '_station_time': station_time_dict,
                '_particle_type': self._particle_type,
                'triggers': trigger_pkls,
                '_triggered': self._triggered,
                'electric_fields': efield_pkls}

        return pickle.dumps(data, protocol=4)

    def deserialize(self, data_pkl):
        data = pickle.loads(data_pkl)

        if 'triggers' in data:
            self._triggers = NuRadioReco.framework.trigger.deserialize(data['triggers'])

        if 'triggers' in data:
            self._triggered = data['_triggered']

        for electric_field in data['electric_fields']:
            efield = NuRadioReco.framework.electric_field.ElectricField([])
            efield.deserialize(electric_field)
            self.add_electric_field(efield)

        self._parameters = NuRadioReco.framework.parameter_serialization.deserialize(data['_parameters'],
                                                                                     parameters.stationParameters)

        self._parameter_covariances = NuRadioReco.framework.parameter_serialization.deserialize_covariances(
            data['_parameter_covariances'], parameters.stationParameters)

        if '_ARIANNA_parameters' in data:
            self._ARIANNA_parameters = data['_ARIANNA_parameters']

        self._station_id = data['_station_id']
        if data['_station_time'] is not None:
            if isinstance(data['_station_time'], dict):
                station_time = astropy.time.Time(data['_station_time']['value'], format=data['_station_time']['format'])
                self.set_station_time(station_time)
            # For backward compatibility, we also keep supporting station times stored as astropy.time objects
            else:
                self.set_station_time(data['_station_time'])

        self._particle_type = data['_particle_type']
