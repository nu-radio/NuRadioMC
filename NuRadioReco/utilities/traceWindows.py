import numpy as np
from NuRadioReco.utilities import units
import NuRadioReco.utilities.diodeSimulator

def get_window_around_maximum(station,
                              diode=None,
                              triggered_channels=None,
                              ratio = 0.01,
                              edge=20*units.ns):
    """
    This function filters the signal using a diode model and calculates
    the times around the filtered maximum where the signal is the ratio
    parameter times the maximum. Then, it creates a time window by substracting
    and adding the edge parameter to these times. This function is useful
    for reducing the probability of noise-triggered events while using the
    envelope phased array, although it can also be applied to the standard
    phased array.

    Parameters
    ----------
    station: NuRadioReco station
        Station containing the information on the detector
    diode: diodeSimulator or None
        Diode model to be applied. If None, a diode with an output 200 MHz low-pass
        filter is chosen
    triggered_channels: array of integers
        Channels used for the trigger, and also for creating the window
    ratio: float
        Cut parameter
    edge: float
        Times to be sustracted from the points defined by the ratio cut to
        create the window

    Returns:
    (left_time, right_time): (float, float) tuple
        Tuple containing the edges of the time window
    """

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

        left_bin = argmin - np.argmin(np.abs(trace[0:argmin][::-1]+ratio*trace_max))
        left_times.append(times[left_bin])

        right_bin = argmax + np.argmin(np.abs(trace[argmax:None]-ratio*trace_max))
        right_times.append(times[right_bin])

    left_time = np.min(left_times) - edge
    right_time = np.max(right_times) + edge

    return (left_time,right_time)
