from __future__ import absolute_import, division, print_function
import NuRadioReco.framework.parameters as parameters
import NuRadioReco.framework.parameter_serialization
import pickle
import collections
import logging
logger = logging.getLogger('Particle')


class Particle:

    def __init__(self, particle_id=0):
        self._id = particle_id
        self._parameters = {}
    
    def __setitem__(self, key, value):
        self.set_parameter(key, value)

    def __getitem__(self, key):
        return self.get_parameter(key)

    def get_id(self):
        return self._id

    def get_parameter(self, key):
        if not isinstance(key, parameters.particleParameters):
            logger.error("parameter key needs to be of type NuRadioReco.framework.parameters.particleParameters")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.particleParameters")
        return self._parameters[key]

    def set_parameter(self, key, value):
        if not isinstance(key, parameters.particleParameters):
            logger.error("parameter key needs to be of type NuRadioReco.framework.parameters.particleParameters")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.particleParameters")
        self._parameters[key] = value

    def has_parameter(self, key):
        if not isinstance(key, parameters.particleParameters):
            logger.error("parameter key needs to be of type NuRadioReco.framework.parameters.particleParameters")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.particleParameters")
        return key in self._parameters
    
    def as_hdf5_dict(self):
        hdf5_dict = collections.OrderedDict()
        hdf5_dict['azimuths'] = self.get_parameter(parameters.particleParameters.azimuth)
        hdf5_dict['energies'] = self.get_parameter(parameters.particleParameters.energy)
        hdf5_dict['event_group_ids'] = self.get_id()
        hdf5_dict['flavors'] = self.get_parameter(parameters.particleParameters.flavor)
        hdf5_dict['inelasticity'] = self.get_parameter(parameters.particleParameters.inelasticity)
        hdf5_dict['interaction_type'] = self.get_parameter(parameters.particleParameters.interaction_type)
        hdf5_dict['n_interaction'] = self.get_parameter(parameters.particleParameters.n_interaction)
        hdf5_dict['vertex_times'] = self.get_parameter(parameters.particleParameters.vertex_time)
        hdf5_dict['weights'] = self.get_parameter(parameters.particleParameters.weight)
        hdf5_dict['xx'] = self.get_parameter(parameters.particleParameters.vertex[0])
        hdf5_dict['yy'] = self.get_parameter(parameters.particleParameters.vertex[1])
        hdf5_dict['zeniths'] = self.get_parameter(parameters.particleParameters.zenith)
        hdf5_dict['zz'] = self.get_parameter(parameters.particleParameters.vertex[2])
        return(hdf5_dict)

    def serialize(self):
        data = {'_parameters': NuRadioReco.framework.parameter_serialization.serialize(self._parameters),
                '_id': self._id}
        return pickle.dumps(data, protocol=4)

    def deserialize(self, data_pkl):
        data = pickle.loads(data_pkl)
        if '_id' in data.keys():
            self._id = data['_id']
        else:
            self._id = None
        self._parameters = NuRadioReco.framework.parameter_serialization.deserialize(
            data['_parameters'],
            parameters.particleParameters
        )
