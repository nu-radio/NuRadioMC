import numpy as np
from NuRadioReco.framework.event import Event
import collections

def get_averaged_channel_parameter(event_or_station, key, channels_to_include=None):
    """
    Return average across all channels to include for a channel parameter "param"
    """
    if isinstance(event_or_station, Event):
        # This will throw an error if the event has more than one station
        station = event_or_station.get_station()
    else:
        station = event_or_station

    params = None
    for channel in station.iter_channels(channels_to_include):
        if not channel.has_parameter(key):
            raise KeyError(f"Channel {channel.get_id()} has no parameter {key}.")

        param = channel.get_parameter(key)

        if isinstance(param, dict):
            if params is None:
                params = collections.defaultdict(list)

            for k, v in param.items():
                params[k].append(v)
        elif isinstance(param, (float, int)):
            if params is None:
                params = []

            params.append(param)
        else:
            raise ValueError(f"Unknown type ({type(param)}) for parameter {key}")

    if isinstance(params, dict):
        params = {k: np.average(v) for k, v in params.items()}
    else:
        params = np.average(params)

    return params

