from NuRadioReco.framework.parameters import channelParametersRNOG as chpRNOG
from NuRadioReco.framework.event import Event


def has_glitch(event_or_station):
    """
    Returns true if any channel in any station has a "glitch".

    Requires the RNO_G.channelGlitchDetector module to have ran on the event.

    Parameters
    ----------
    event_or_station : Event or Station
        The event or station to check for glitches. If an event is given, the first station is used
        (if multiple stations exist in the event an error is raised).

    Returns
    -------
    has_glitch : bool
        True if any channel has a glitch, False otherwise.

    Also see
    --------
    NuRadioReco.modules.RNO_G.channelGlitchDetector
    """
    if isinstance(event_or_station, Event):
        # This will throw an error if the event has more than one station
        station = event_or_station.get_station()
    else:
        station = event_or_station

    for channel in station.iter_channels():
        if channel.has_parameter(chpRNOG.glitch) and channel.get_parameter(chpRNOG.glitch):
            return True

    return False