import numpy as np
from NuRadioReco.utilities import units
import NuRadioReco.utilities.diodeSimulator

def get_window_around_maximum(station, diode, triggered_channels=None, ratio = 0.01, edge=20*units.ns):

    if diode == None:
        diode_passband = (None, 200*units.MHz)
        diode = NuRadioReco.utilities.diodeSimulator.diodeSimulator(diode_passband)

    left_times = []
    right_times = []

    if (triggered_channels == None):
    	triggered_channels = [channel._id for channel in station.iter_channels()]

    for channel in station.iter_channels():  # loop over all channels (i.e. antennas) of the station

        if channel.get_id() not in triggered_channels:
            continue

        times = channel.get_times()

        trace = diode.tunnel_diode(channel)
        trace_max = np.max(np.max(trace))
        argmax = np.argmax(trace)
        trace_min = np.min(np.min(trace))
        argmin = np.argmin(trace)

        if (argmin == len(trace)-1):
            argmin = len(trace)-2
        if (argmin == argmax):
            if (argmin + 5 < len(trace)):
                argmax = argmin + 5
            else:
                argmax = len(trace) - 1
        if (argmin == 0):
            argmin = 1

        left_bin = argmin - np.argmin( np.abs(trace[0:argmin][::-1]+ratio*trace_max) )
        left_times.append(times[left_bin])

        right_bin = argmax + np.argmin(np.abs(trace[argmax:None]-ratio*trace_max))
        right_times.append(times[right_bin])

    left_time = np.min(left_times) - edge
    right_time = np.max(right_times) + edge

    return (left_time,right_time)
