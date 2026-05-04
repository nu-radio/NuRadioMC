from NuRadioReco.modules.phasedarray.beamformedPowerIntegrationTrigger import \
    BeamformedPowerIntegrationTrigger

import warnings
import logging
logger = logging.getLogger('NuRadioReco.phasedArray.triggerSimulator')


class triggerSimulator(BeamformedPowerIntegrationTrigger):

    def __init__(self, *args, **kwargs):

        warnings.warn("`NuRadioReco.modules.phasedarray.triggerSimulator.triggerSimulator()` is deprecated. "
                      "Please use `NuRadioReco.modules.phasedarray.beamformedPowerIntegrationTrigger.BeamformedPowerIntegrationTrigger()` instead.", DeprecationWarning)
        logger.warning("`NuRadioReco.modules.phasedarray.triggerSimulator.triggerSimulator()` is deprecated. "
                      "Please use `NuRadioReco.modules.phasedarray.beamformedPowerIntegrationTrigger.BeamformedPowerIntegrationTrigger()` instead.")
        super().__init__(*args, **kwargs)
