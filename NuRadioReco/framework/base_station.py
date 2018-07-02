from __future__ import absolute_import, division, print_function, unicode_literals
import NuRadioReco.framework.base_trace
import NuRadioReco.framework.trigger
import cPickle as pickle
import logging
logger = logging.getLogger('BaseStation')


class BaseStation(NuRadioReco.framework.base_trace.BaseTrace):

    def __init__(self, station_id):
        NuRadioReco.framework.base_trace.BaseTrace.__init__(self)
        self._parameters = {}
        self._parameter_covariances = {}
        self._station_id = station_id
        self._station_time = None
        self._trigger = NuRadioReco.framework.trigger.Trigger()
        self._parameter_names = ['cr_energy',
                                 'zenith',
                                 'azimuth',
                                 'signal_time',
                                 'triggered',
                                 'signal_energy_fluence_mag',
                                 'noise_energy_fluence_mag',
                                 'maximum_amplitude']

    def __setitem__(self, key, value):
        self.set_parameter(key, value)

    def __getitem__(self, key):
        return self.get_parameter(key)

    def get_parameter(self, attribute):
        return self._parameters[attribute]

    def get_parameters(self):
        return self._parameters

    def has_parameter(self, attribute):
        return attribute in self._parameters.keys()

    def set_parameter(self, key, value):
#         if key not in self._parameter_names:
#             logger.error("parameter {} does not exist in list of parameters".format(key))
#             import sys
#             sys.exit(-1)
        self._parameters[key] = value

    def set_parameter_error(self, key, value):
        self._parameter_covariances[(key, key)] = value ** 2

    def get_parameter_error(self, key):
        return self._parameter_covariances[(key, key)] ** 0.5

    def remove_parameter(self, key):
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

    def get_trigger(self):
        return self._trigger

    def set_trigger(self, trigger):
        self._trigger = trigger

    def has_triggered(self):
        return self._trigger.has_triggered()

    def set_triggered(self, triggered=True):
        self._trigger.set_triggered(triggered)

#     def get_frequencies(self):
#         return np.fft.rfftfreq(len(self._time_trace), d=(1. / self._sampling_rate))
#
#     def get_frequency_spectrum(self):
#         return np.fft.rfft(self._time_trace, axis=0, norm="ortho")

    def serialize(self, mode):
        if(mode == 'micro'):
            base_trace_pkl = None
        else:
            base_trace_pkl = NuRadioReco.framework.base_trace.BaseTrace.serialize(self)
        trigger_pkl = self._trigger.serialize()
        data = {'_parameters': self._parameters,
                '_parameter_covariances': self._parameter_covariances,
                '_station_id': self._station_id,
                '_station_time': self._station_time,
                'trigger': trigger_pkl,
                'base_trace': base_trace_pkl}
        return pickle.dumps(data, protocol=2)

    def deserialize(self, data_pkl):
        data = pickle.loads(data_pkl)
        if(data['base_trace'] is not None):
            NuRadioReco.framework.base_trace.BaseTrace.deserialize(self, data['base_trace'])
        self._trigger.deserialize(data['trigger'])
        self._parameters = data['_parameters']
        self._parameter_covariances = data['_parameter_covariances']
        self._station_id = data['_station_id']
        self._station_time = data['_station_time']
