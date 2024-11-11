"""
This module implements the HybridStation class

This class is similar to `NuRadioReco.modules.framework.station.Station`, but is intended
for stations that feature non-radio channels, e.g. particle detectors.

For now it is just a Station by another name...

"""

from NuRadioReco.framework.station import Station


class HybridStation(Station):

    def __init__(self, station_id):
        super().__init__(station_id)
