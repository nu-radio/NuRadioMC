import logging
import numpy as np
import datetime
from NuRadioReco.utilities import units
import NuRadioReco.detector.detector as detector

# Logging level
from NuRadioReco.modules.base import module
logger = module.setup_logger(level=logging.INFO)

# read in detector positions (this is a dummy detector)
det = detector.Detector()

station_id = 52

det.update(datetime.datetime(2018, 1, 1))

n_channels = det.get_number_of_channels(station_id)

for channel_id in range(n_channels):
    position = det.get_relative_position(station_id, channel_id)
    antenna_orientation = det.get_antenna_orientation(station_id, channel_id)
    print("position of channel {:d} is {:.2f} m, {:.2f} m, {:.2f} ".format(channel_id, *position / units.m))
    print("antenna boresight direction ({:.0f}, {:.0f}), antenna rotation ({:.0f}, {:.0f})".format(*antenna_orientation / units.deg))

