from NuRadioReco.framework.parameters import channelParametersRNOG as chp
from NuRadioReco.modules.base.module import register_run
from scipy import signal
import numpy as np
import logging

class channelImpulsivityCalculator:

    def __init__(self, fraction = 0.5, log_level=logging.NOTSET):
        self.fraction = fraction
        self.logger = logging.getLogger('NuRadioReco.RNO_G.channelImpulsivityCalculator')

    def begin(self):
        pass

    def end(self):
        pass

    def _calc(self, trace):
        env_pwr = np.square(np.abs(signal.hilbert(trace)))
        ind_pwr_max = np.argmax(env_pwr)
        inds = np.arange(0, len(env_pwr), 1)
        closeness = abs(inds - ind_pwr_max)
        sorter = np.argsort(closeness)

        env_pwr_sorted = env_pwr[sorter]

        env_pwr_xvals = np.linspace(0.0, 1.0, len(env_pwr_sorted))
        mask = env_pwr_xvals > self.fraction

        env_pwr_sorted[mask] = 0.0
        env_pwr_pdf = env_pwr_sorted / np.sum(env_pwr_sorted)

        peak_over_rms_power = env_pwr_pdf[0] * len(env_pwr_sorted) * self.fraction
        return peak_over_rms_power

    @register_run()
    def run(self, event, station, det=None):
        for ch in station.iter_channels():
            trace = ch.get_trace()
            ch.set_parameter(chp.power_impulsivity, self._calc(trace))

