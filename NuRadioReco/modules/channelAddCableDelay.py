from NuRadioReco.modules.base.module import register_run
from NuRadioReco.detector import detector
from NuRadioReco.utilities import units
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
        add_cable_delay(station, det, sim_to_data, trigger=False, logger=self.logger)


def add_cable_delay(station, det, sim_to_data=None, trigger=False, logger=None):
    """
    Add or subtract cable delay by modifying the ``trace_start_time``.

    Parameters
    ----------
    station: Station
        The station to add the cable delay to.

    det: Detector
        The detector description

    trigger: bool
        If True, take the time delay from the trigger channel response.
        Only possible if ``det`` is of type `rnog_detector.Detector`. (Default: False)

    logger: logging.Logger, default=None
        If set, use ``logger.debug(..)`` to log the cable delay.
    """
    assert sim_to_data is not None, "``sim_to_data`` is None, please specify."

    add_or_subtract = 1 if sim_to_data else -1
    msg = "Add" if sim_to_data else "Subtract"

    if trigger and not isinstance(det, detector.rnog_detector.Detector):
        raise ValueError("Simulating extra trigger channels is only possible with the `rnog_detector.Detector` class.")

    for channel in station.iter_channels():

        if trigger:
            if not channel.has_extra_trigger_channel():
                continue

            channel = channel.get_trigger_channel()
            cable_delay = det.get_cable_delay(station.get_id(), channel.get_id(), trigger=True)

        else:
            # Only the RNOG detector has the argument `trigger`. Default is false
            cable_delay = det.get_cable_delay(station.get_id(), channel.get_id())

        if logger is not None:
            logger.debug(f"{msg} {cable_delay / units.ns:.2f}ns "
                        f"of cable delay to channel {channel.get_id()}")

        channel.add_trace_start_time(add_or_subtract * cable_delay)