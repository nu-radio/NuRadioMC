from NuRadioReco import framework as fwk

import NuRadioReco.framework.parameters as parameters
import NuRadioReco.framework.parameter_serialization
import pickle

import logging
logger = logging.getLogger('NuRadioReco.SimEmitter')


class SimEmitter(fwk.Emitter):
    def __init__(self, emitter_id=0, station_ids=None):
        fwk.Emitter.__init__(self, emitter_id, station_ids)