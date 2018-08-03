import numpy as np
import matplotlib.pyplot as plt
import logging
from NuRadioReco.utilities import units
import NuRadioReco.framework.channel
import NuRadioReco.framework.station
import NuRadioReco.framework.event
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelGenericNoiseAdder

import NuRadioReco.modules.ARA.triggerSimulator
import NuRadioReco.modules.ARIANNA.triggerSimulator

from NuRadioReco.detector import detector
det = detector.Detector(json_filename='example_data/dummy_detector.json')

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("TriggerComparison")

# Use numpy array with band-pass limited pulse [30-1000] MHz, samples at 10 GHz
data = np.load("example_data/Test_data_8.npy")

test_pulse = data[2,:]
# Normalize test pulse to 1
test_pulse /= np.max(np.abs(test_pulse))

channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
channelGenericNoiseAdder.begin(debug=False)

channelResampler = NuRadioReco.modules.channelResampler.channelResampler()

triggerSimulator_ARA = NuRadioReco.modules.ARA.triggerSimulator.triggerSimulator()
triggerSimulator_ARA.begin()
triggerSimulator_ARIANNA = NuRadioReco.modules.ARIANNA.triggerSimulator.triggerSimulator()
triggerSimulator_ARIANNA.begin()

event_ARA = NuRadioReco.framework.event.Event(1,1)
event_ARIANNA = NuRadioReco.framework.event.Event(1,1)

station_ARA = NuRadioReco.framework.station.Station(101)
channel_ARA = NuRadioReco.framework.channel.Channel(0)

station_ARIANNA = NuRadioReco.framework.station.Station(101)
channel_ARIANNA = NuRadioReco.framework.channel.Channel(0)

def Calc_SNR(trace):
    max = np.max(np.abs(trace))/(17.6*units.mV)
    return max

n_scaling = 50
result_ARA = np.zeros((n_scaling,2))
result_ARIANNA = np.zeros((n_scaling,2))

i = 0
n_iter = 20

for scaling in np.linspace(10*units.mV,200*units.mV,n_scaling):
    test_pulse_sc = test_pulse * scaling

    n_trigger_ARA = 0
    n_trigger_ARIANNA = 0

    SNR_ARA = 0
    SNR_ARIANNA = 0

    max = []
    for n in xrange(n_iter):

        channel_ARA.set_trace(test_pulse_sc,10*units.GHz)
        station_ARA.add_channel(channel_ARA)


        channel_ARIANNA.set_trace(test_pulse_sc,10*units.GHz)
        station_ARIANNA.add_channel(channel_ARIANNA)


        channelGenericNoiseAdder.run(event_ARA,station_ARA,
                            det,amplitude=20*units.mV,min_freq=30*units.MHz,max_freq=1000*units.MHz,type='white')
        channelGenericNoiseAdder.run(event_ARIANNA,station_ARIANNA,
                            det,amplitude=20*units.mV,min_freq=30*units.MHz,max_freq=1000*units.MHz,type='white')

        channelResampler.run(event_ARA, station_ARA, det, sampling_rate=1 * units.GHz)
        channelResampler.run(event_ARIANNA, station_ARIANNA, det, sampling_rate=1 * units.GHz)

        triggerSimulator_ARA.run(event_ARA, station_ARA, det, power_threshold=6.5,
                                        coinc_window = 110 * units.ns,
                                        number_concidences =  1,
                                        triggered_channels = [0, 1, 2, 3, 4, 5, 6, 7])

        triggerSimulator_ARIANNA.run(event_ARIANNA, station_ARIANNA, det,threshold_high=53 * units.mV,
                threshold_low=-53 * units.mV,
                high_low_window=20 * units.ns,
                coinc_window=32 * units.ns,
                number_concidences=1,
                triggered_channels=[0, 1, 2, 3])


        SNR_ARA +=  Calc_SNR(station_ARA.get_channels()[0].get_trace())
        SNR_ARIANNA += Calc_SNR(station_ARIANNA.get_channels()[0].get_trace())

        max.append(np.max(np.abs(station_ARA.get_channels()[0].get_trace())))
#         channel_ARA.clear_trace()

        if station_ARA.has_triggered() != False:
            n_trigger_ARA  += 1.
        if station_ARIANNA.has_triggered()!= False:
            n_trigger_ARIANNA  += 1.


    result_ARA[i,0] = SNR_ARA / n_iter
    result_ARA[i,1] = n_trigger_ARA / n_iter

    result_ARIANNA[i,0] = SNR_ARIANNA / n_iter
    result_ARIANNA[i,1] = n_trigger_ARIANNA / n_iter
    i += 1




plt.figure()
plt.plot(result_ARA[:,0],result_ARA[:,1], linestyle='None', marker ='o',label="ARA, power_threshold 6.5")
plt.plot(result_ARIANNA[:,0],result_ARIANNA[:,1], linestyle='None', marker ='s',label="ARIANNA, 3 sigma")
plt.ylabel("Trigger efficiency on one antenna")
plt.xlabel("SNR: max(ampl_signal) / RMS(ampl_noise)")
plt.legend()
plt.show()