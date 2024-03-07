import NuRadioReco.framework.parameters as parameters
import NuRadioReco.framework.parameter_serialization
import pickle

import logging
logger = logging.getLogger('SimEmitter')


class SimEmitter(NuRadioReco.framework.emitter.Emitter):
    def __init__(self, emitter_id=0, station_ids=None):
        NuRadioReco.framework.emitter.Emitter.__init__(self, emitter_id, station_ids)