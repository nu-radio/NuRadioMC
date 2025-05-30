from __future__ import absolute_import, division, print_function
import NuRadioReco.framework.base_trace
import NuRadioReco.framework.channel

import pickle
from NuRadioReco.utilities.io_utilities import _dumps
import logging
logger = logging.getLogger('NuRadioReco.SimChannel')


class SimChannel(NuRadioReco.framework.channel.Channel):
    """
    Object to store simulated channels.

    This class is the same as the regular channel trace but has apart from the channel id (and possible group id) also
    a shower and ray tracing solution id
    """

    def __init__(self, channel_id, shower_id, ray_tracing_id, channel_group_id=None):
        """
        Initializes SimChannel object

        Parameters
        ----------
        channel_id: int
            id of channel
        shower_id: int or None
            the id of the corresponding shower object
        ray_tracing_id: int or None
            the id of the corresponding ray tracing solution
        channel_group_id: int (default: None)
            optionally, several channels can belong to a "channel group". Use case is to identify
            the channels of a single dual or triple polarized antenna as common in air shower arrays.
        """
        NuRadioReco.framework.channel.Channel.__init__(self, channel_id, channel_group_id=channel_group_id)
        self._shower_id = shower_id
        self._ray_tracing_id = ray_tracing_id

    def get_shower_id(self):
        return self._shower_id

    def get_ray_tracing_solution_id(self):
        return self._ray_tracing_id

    def get_unique_identifier(self):
        """
        returns a unique identifier consisting of the tuple channel_id, shower_id and ray_tracing_id
        """
        return (self._id, self._shower_id, self._ray_tracing_id)

    def serialize(self, save_trace):
        channel_pkl = NuRadioReco.framework.channel.Channel.serialize(self, save_trace)
        data = {
            'shower_id': self.get_shower_id(),
            'ray_tracing_id': self.get_ray_tracing_solution_id(),
            'channel': channel_pkl
        }

        return _dumps(data, protocol=4)

    def deserialize(self, data_pkl):
        data = pickle.loads(data_pkl)
        NuRadioReco.framework.channel.Channel.deserialize(self, data['channel'])

        self._shower_id = data['shower_id']
        self._ray_tracing_id = data['ray_tracing_id']
