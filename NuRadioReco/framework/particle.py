from __future__ import absolute_import, division, print_function
from NuRadioReco.framework.parameters import particleParameters as paP
import NuRadioReco.framework.parameter_storage
import pickle
import collections
import math
import logging
logger = logging.getLogger('NuRadioReco.Particle')


class Particle(NuRadioReco.framework.parameter_storage.ParameterStorage):

    def __init__(self, particle_index):
        super().__init__(paP)
        # "_id" is not the PDG code but a hierarchical index
        # (PDG code is stored in _parameters["flavor"])
        self._id = particle_index

    def get_id(self):
        """ Returns hierarchical index """
        return self._id

    def __str__(self):
        msg = (
            "Particle ({}): "
            "Flavor: {: 3}, lgE = {:.1f}, cos(theta) = {:.2f}".format(
                hex(id(self)),
                self.get_parameter(paP.flavor),
                math.log10(self.get_parameter(paP.energy)),
                math.cos(self.get_parameter(paP.zenith)))
        )

        return msg

    def as_hdf5_dict(self):
        hdf5_dict = collections.OrderedDict()

        key_pairs = [
            (paP.azimuth, 'azimuths'), (paP.energy, 'energies'), (paP.flavor, 'flavors'),
            (paP.inelasticity, 'inelasticity'), (paP.interaction_type, 'interaction_type'),
            (paP.n_interaction, 'n_interaction'), (paP.vertex_time, 'vertex_times'),
            (paP.weight, 'weights'), (paP.vertex[0], 'xx'), (paP.vertex[1], 'yy'),
            (paP.zenith, 'zeniths'), (paP.vertex[2], 'zz')
        ]

        for key, name in key_pairs:
            hdf5_dict[name] = self.get_parameter(key)

        hdf5_dict['event_group_ids'] = self.get_id()
        return hdf5_dict

    def serialize(self):
        data = NuRadioReco.framework.parameter_storage.ParameterStorage.serialize(self)
        data['_id'] = self._id
        return pickle.dumps(data, protocol=4)

    def deserialize(self, data_pkl):
        data = pickle.loads(data_pkl)
        NuRadioReco.framework.parameter_storage.ParameterStorage.deserialize(self, data)
        self._id = data.get('_id', None)
