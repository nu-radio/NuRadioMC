from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units
import logging


class channelAddCableDelay:
    """
    Adds the cable delay to channels
    """

    def __init__(self):
        self.logger = logging.getLogger("NuRadioReco.channelApplyCableDelay")

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
        for channel in station.iter_channels():
            cable_delay = det.get_cable_delay(station.get_id(), channel.get_id())
            self.logger.debug(f"Channel {channel.get_id()}: {mode} {cable_delay / units.ns:.2f} ns")

            if mode == 'add':
                channel.add_trace_start_time(cable_delay)
            elif mode == 'subtract':
                channel.add_trace_start_time(-1 * cable_delay)
            else:
                raise ValueError(f"Unknown mode '{mode}' for channelAddCableDelay. "
                                 "Valid options are 'add' or 'subtract'.")
