from NuRadioReco.framework.parameters import channelParametersRNOG as chp
from NuRadioReco.modules.base.module import register_run
import logging

class channelPiCalculator:

    def __init__(self, fraction = 0.5, log_level=logging.NOTSET):
        self.logger = logging.getLogger('NuRadioReco.RNO_G.channelPiCalculator')

    def begin(self):
        pass

    def end(self):
        pass

    @register_run()
    def run(self, event, station, det=None):
        for ch in station.iter_channels():
            trace = ch.get_trace()
            ch.set_parameter(chp.rnog_pi, 3.14159265)

