from __future__ import absolute_import, division, print_function
import NuRadioReco.framework.base_trace
import cPickle as pickle
import logging
logger = logging.getLogger('channel')


class Channel(NuRadioReco.framework.base_trace.BaseTrace):

    def __init__(self, channel_id):
        NuRadioReco.framework.base_trace.BaseTrace.__init__(self)
        self._parameters = {}
        self._id = channel_id
        self.__electric_field = None

    def get_parameter(self, attribute):
        return self._parameters[attribute]

    def set_parameter(self, key, value):
        self._parameters[key] = value

    def has_parameter(self, key):
        return key in self._parameters

    def __setitem__(self, key, value):
        self.set_parameter(key, value)

    def __getitem__(self, key):
        return self.get_parameter(key)

    def get_id(self):
        return self._id

    def set_electric_field(self, trace):
        if not (isinstance(trace, NuRadioReco.framework.base_trace.BaseTrace)):
            logger.error("electric field is not of instance of framework.base_trace.BaseTrace, but is of name {}".format(trace.__class__.__name__))
            raise NotImplemented
        self.__electric_field = trace

    def get_electric_field(self):
        return self.__electric_field

    def serialize(self, mode):
        if(mode == 'micro' or mode == 'mini'):
            base_trace_pkl = None
        else:
            base_trace_pkl = NuRadioReco.framework.base_trace.BaseTrace.serialize(self)
        data = {'parameters': self._parameters,
                'id': self.get_id(),
                'base_trace': base_trace_pkl}
        if(self.__electric_field is not None):
            if(not(mode == 'micro' or mode == 'mini')):
                data['electric_field'] = self.__electric_field.serialize()
        return pickle.dumps(data, protocol=2)

    def deserialize(self, data_pkl):
        data = pickle.loads(data_pkl)
        if(data['base_trace'] is not None):
            NuRadioReco.framework.base_trace.BaseTrace.deserialize(self, data['base_trace'])
        self._parameters = data['parameters']
        self._id = data['id']
        if 'electric_field' in data.keys():
            self.__electric_field = NuRadioReco.framework.base_trace.BaseTrace.deserialize(self, data['electric_field'])
