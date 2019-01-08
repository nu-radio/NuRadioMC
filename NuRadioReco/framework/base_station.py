from __future__ import absolute_import, division, print_function, unicode_literals
import NuRadioReco.framework.base_trace
import NuRadioReco.framework.trigger
import NuRadioReco.framework.parameters as parameters
try:
    import cPickle as pickle
except ImportError:
    import pickle
import logging
logger = logging.getLogger('BaseStation')


class BaseStation(NuRadioReco.framework.base_trace.BaseTrace):

    def __init__(self, station_id):
        NuRadioReco.framework.base_trace.BaseTrace.__init__(self)
        self._parameters = {}
        self._parameter_covariances = {}
        self._station_id = station_id
        self._station_time = None
        self._triggers = {}
        self._triggered = False
        self._is_neutrino = True

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

    def set_station_time(self, time):
        self._station_time = time

    def get_station_time(self):
        return self._station_time

#     def get_trace(self):
#         return self._time_trace
#
#     def set_trace(self, trace, sampling_rate):
#         self._time_trace = trace
#         self._sampling_rate = sampling_rate
#
#     def get_sampling_rate(self):
#         return self._sampling_rate
#
#     def get_times(self):
#         return np.arange(0, len(self._time_trace) / self._sampling_rate, 1. / self._sampling_rate)

    def get_id(self):
        return self._station_id

    def get_trigger(self, name):
        if(name not in self._triggers):
            raise ValueError("trigger with name {} not present".format(name))
        return self._triggers[name]
    
    def get_triggers(self):
        """
        returns a dictionary of the triggers. key is the trigger name, value is a trigger object
        """
        return self._triggers

    def set_trigger(self, trigger):
        if(trigger.get_name() in self._triggers):
            logger.warning(
                "station has already a trigger with name {}. The previous trigger will be overridden!".format(trigger.get_name()))
        self._triggers[trigger.get_name()] = trigger
        self._triggered = trigger.has_triggered() or self._triggered

    def has_triggered(self, trigger_name=None):
        """
        convenience function. 
        
        Parameters
        ---------- 
        trigger_name: string or None (default None)
            * if None: The function returns False if not trigger was set. If one or multiple triggers were set,
                       it returns True if any of those triggers triggered
            * if trigger name is set: return if the trigger with name 'trigger_name' has a trigger
        """
        if(trigger_name is None):
            return self._triggered
        else:
            return self.get_trigger(trigger_name).has_triggered()

    def set_triggered(self, triggered=True):
        """
        convenience function to set a simple trigger. The recommended interface is to set triggers through the 
        set_trigger() interface.
        """
        if(len(self._triggers) > 1):
            raise ValueError("more then one trigger were set. Request is ambiguous")
        trigger = NuRadioReco.framework.trigger.Trigger('default')
        trigger.set_triggered(triggered)
        self.set_trigger(trigger)

    def is_neutrino():
        return self._is_neutrino
        
    def is_cosmic_ray():
        return not self._is_neutrino
        
    def set_is_neutrino(is_neutrino):
        self._is_neutrino = is_neutrino


    def serialize(self, mode):
        if(mode == 'micro'):
            base_trace_pkl = None
        else:
            base_trace_pkl = NuRadioReco.framework.base_trace.BaseTrace.serialize(self)
        trigger_pkls = []
        for trigger in self._triggers.values():
            trigger_pkls.append(trigger.serialize())
        data = {'_parameters': self._parameters,
                '_parameter_covariances': self._parameter_covariances,
                '_station_id': self._station_id,
                '_station_time': self._station_time,
                '_is_neutrino': self._is_neutrino,
                'triggers': trigger_pkls,
                '_triggered': self._triggered,
                'base_trace': base_trace_pkl}
        return pickle.dumps(data, protocol=2)

    def deserialize(self, data_pkl):
        data = pickle.loads(data_pkl)
        if(data['base_trace'] is not None):
            NuRadioReco.framework.base_trace.BaseTrace.deserialize(self, data['base_trace'])
        if ('triggers' in data):
            self._triggers = NuRadioReco.framework.trigger.deserialize(data['triggers'])
        if ('triggers' in data):
            self._triggered = data['_triggered']
        self._parameters = data['_parameters']
        self._parameter_covariances = data['_parameter_covariances']
        self._station_id = data['_station_id']
        self._station_time = data['_station_time']
        self._is_neutrino = data['_is_neutrino']
