from NuRadioReco.utilities import units
import numpy as np
import time
import logging
logger = logging.getLogger('triggerSimulator')


class triggerSimulator:
    """
    Calculates the trigger for a phased array

    explain the module in more detail here
    """

    def __init__(self):
        self.__t = 0
        self.begin()

    def begin(self, samples_before_trigger=50):
        """
        initialize some general settings here
        """
        self.__samples_before_trigger = samples_before_trigger

    def run(self, evt, station, det,
            triggered_channels=[0, 1, 2, 3]):
        """
        simulates phased array trigger for each event

        describe the run method in more detail here (if neccessary).
        I left triggered_channels in as an optional parameter, don't know if you
        need it.

        Parameters
        ----------
        triggered_channels: array of ints
            channels ids that are triggered on
        """

        sampling_rate = station.get_channel(0).get_sampling_rate()

        phased_trace = None

        for channel in station.iter_channels():  # loop over all channels (i.e. antennas) of the station
            channel_id = channel.get_id()
            if channel_id not in triggered_channels:  # skip all channels that do not participate in the trigger decision
                continue

            trace = channel.get_trace()  # get the time trace (i.e. an array of amplitudes)
            times = channel.get_times()  # get the corresponding time bins

            antenna_position = det.get_relative_position(station.get_id(), channel.get_id())  # ask the detector to get the antenna position, the function returns an array of [x, y,z] coordinates

            # phase up the signal, assuming no time delays between the channels
            if(phased_trace is None):
                phased_trace = trace
            else:
                phased_trace += trace

        if(np.max(np.abs(phased_trace)) > 10 * units.mV):  # define a simple threshold trigger
            station.set_triggered(True)

    def end(self):
        pass
