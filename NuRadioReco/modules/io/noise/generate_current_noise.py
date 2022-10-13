from re import S
import numpy as np
import matplotlib.pyplot as plt
from NuRadioReco.utilities import fft
from NuRadioReco.utilities import units
from NuRadioReco.framework import base_trace
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
import NuRadioReco.modules.channelGenericNoiseAdder
from NuRadioReco.detector import detector
import datetime

noise_adder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
hardware_response = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()

def empty_event(run_number=1, event_number=1, station=11):
    '''Fucntion creating empty event'''
    evt = NuRadioReco.framework.event.Event(run_number, event_number)
    station = NuRadioReco.framework.station.Station(station)
    # Loop over all requested channels in data
    for chan in range(24):
        channel = NuRadioReco.framework.channel.Channel(chan)
        channel.set_trace(np.zeros(2048), 3.2*units.GHz)
        station.add_channel(channel)
    evt.set_station(station)
    return evt

def get_sim_noise():
    '''Function to get simulated noise'''
    evt = empty_event()
    det = detector.Detector(json_filename="RNO_G/RNO_season_2021.json")
    det.update(datetime.datetime.now())
    noise_adder.run(evt, evt.get_station(11), det)
    hardware_response.run(evt, evt.get_station(11), det, sim_to_data=True)
    return evt


def generate_current_noise(number_of_events, trace_length):
    '''Function that generates noise used in current simulations'''
    events = np.zeros(shape=(number_of_events,trace_length))
    for i in range (number_of_events):
        evt = get_sim_noise()
        channel = evt.get_station(11).get_channel(13)
        events[i] = channel.get_trace()[0:trace_length]

    return events


if __name__ == '__main__':
    events = generate_current_noise(10000,512)
    np.save("current_noise", events)
     