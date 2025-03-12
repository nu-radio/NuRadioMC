from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units, signal_processing
import logging


class channelAddCableDelay:
    """
    Adds the cable delay to channels.
    """

    def __init__(self):
        self.logger = logging.getLogger("NuRadioReco.channelApplyCableDelay")

    def begin(self):
        pass

    @register_run()
    def run(self, evt, station, det, mode='add'):
        """
        Adds cable delays to channels

        Parameters
        ----------
        evt : `NuRadioReco.framework.event.Event`
        station : `NuRadioReco.framework.station.Station`
        det : Detector
        mode : str (default: "add")
            options: 'add' or 'subtract'.
        """
        if mode not in ['add', 'subtract']:
            raise ValueError(f"Unknown mode '{mode}' for channelAddCableDelay. "
                            "Valid options are 'add' or 'subtract'.")

        sim_to_data = mode == "add"
        signal_processing.add_cable_delay(station, det, sim_to_data, trigger=False, logger=self.logger)