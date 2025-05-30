from __future__ import absolute_import, division, print_function
from NuRadioReco.framework.parameters import particleParameters as partp
import NuRadioReco.framework.parameter_storage
import pickle
from NuRadioReco.utilities.io_utilities import _dumps
import collections
import math
import logging
logger = logging.getLogger('NuRadioReco.Particle')


class Particle(NuRadioReco.framework.parameter_storage.ParameterStorage):

    def __init__(self, particle_index):
        super().__init__(partp)
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
                self.get_parameter(partp.flavor),
                math.log10(self.get_parameter(partp.energy)),
                math.cos(self.get_parameter(partp.zenith)))
        )

        return msg

    def as_hdf5_dict(self):
        hdf5_dict = collections.OrderedDict()

        key_pairs = [
            (partp.azimuth, 'azimuths'), (partp.energy, 'energies'), (partp.flavor, 'flavors'),
            (partp.inelasticity, 'inelasticity'), (partp.interaction_type, 'interaction_type'),
            (partp.n_interaction, 'n_interaction'), (partp.vertex_time, 'vertex_times'),
            (partp.weight, 'weights'), (partp.vertex[0], 'xx'), (partp.vertex[1], 'yy'),
            (partp.zenith, 'zeniths'), (partp.vertex[2], 'zz')
        ]

        for key, name in key_pairs:
            hdf5_dict[name] = self.get_parameter(key)

        hdf5_dict['event_group_ids'] = self.get_id()
        return hdf5_dict

    def serialize(self):
        data = NuRadioReco.framework.parameter_storage.ParameterStorage.serialize(self)
        data['_id'] = self._id
        return _dumps(data, protocol=4)

    def deserialize(self, data_pkl):
        data = pickle.loads(data_pkl)
        NuRadioReco.framework.parameter_storage.ParameterStorage.deserialize(self, data)
        self._id = data.get('_id', None)
