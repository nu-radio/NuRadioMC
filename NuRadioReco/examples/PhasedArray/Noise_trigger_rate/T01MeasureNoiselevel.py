import sys
sys.path.append('/home/danielsmith/icecube_gen2/NuRadioReco')

import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.channelBandPassFilter
from NuRadioReco.utilities import units
import numpy as np
from scipy import constants
import argparse
import NuRadioReco.modules.phasedarray.triggerSimulator
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.framework.station
import NuRadioReco.framework.channel
from NuRadioReco.utilities import fft
from NuRadioReco.detector import detector
from matplotlib import pyplot as plt
from astropy.time import Time
import time
import copy

det = detector.Detector(json_filename="../Effective_volume/8antennas_100m_0.5GHz.json")
det.update(Time.now())

channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()

n_channels = 8

main_low_angle = np.deg2rad(-59.55)
main_high_angle = np.deg2rad(59.55)
channels = []

if(n_channels == 4):
    upsampling_factor = 2
    default_angles = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 11))
    channels = np.arange(4, 8)
elif(n_channels == 8):
    upsampling_factor = 4
    default_angles = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 21))
    channels = None
else:
    print("wrong n_channels!")
    exit()

station_id = 101
thresholds = np.arange(3.5, 5.5, 0.1)
Vrms = 1

n_samples = det.get_number_of_samples(station_id, 1)  # we assume that all channels have the same parameters

sampling_rate = det.get_sampling_frequency(station_id, 1)
print(f"sampling rate = {sampling_rate/units.MHz}MHz, {n_samples} samples")

dt = 1 / sampling_rate
ff = np.fft.rfftfreq(n_samples, dt)

filt1 = channelBandPassFilter.get_filter(ff, 0, 0, None, passband=[0, 240 * units.MHz], filter_type="cheby1", order=9, rp=.1)
filt2 = channelBandPassFilter.get_filter(ff, 0, 0, None, passband=[80 * units.MHz, 230 * units.MHz], filter_type="cheby1", order=4, rp=.1)
filt = filt1 * filt2

# calculate bandwith
max_freq = ff[-1]
min_freq = 0
fff = np.linspace(min_freq, max_freq, 10000)
filt1_highres = channelBandPassFilter.get_filter(fff, 0, 0, None, passband=[0, 240 * units.MHz], filter_type="cheby1", order=9, rp=.1)
filt2_highres = channelBandPassFilter.get_filter(fff, 0, 0, None, passband=[80 * units.MHz, 230 * units.MHz], filter_type="cheby1", order=4, rp=.1)
filt_highres = filt1_highres * filt2_highres
bandwidth = np.trapz(np.abs(filt_highres) ** 2, fff)

Vrms_ratio = np.sqrt(bandwidth / (max_freq - min_freq))
amplitude = Vrms / Vrms_ratio

channel_ids = np.arange(8) #n_channels)

pattern = f"pa_trigger_rate_{n_channels:d}channels_{upsampling_factor}xupsampiling"

triggerSimulator = NuRadioReco.modules.phasedarray.triggerSimulator.triggerSimulator()
thresholdSimulator = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()

phase = True

def loop(zipped):

    threshold = float(zipped[0])
    seed = int(float(zipped[1] * np.power(2.0, 32)))

    station = NuRadioReco.framework.station.Station(station_id)
    evt = NuRadioReco.framework.event.Event(0, 0)

    channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
    channelGenericNoiseAdder.begin(seed = seed)

    for channel_id in channel_ids:
        # Noise rms is amplified to greater than Vrms so that, after filtering, its the Vrms we expect
        spectrum = channelGenericNoiseAdder.bandlimited_noise(min_freq, max_freq, n_samples, sampling_rate, amplitude,
                                                               type="rayleigh", time_domain=False)
        trace = fft.freq2time(spectrum * filt, sampling_rate)        

        channel = NuRadioReco.framework.channel.Channel(channel_id)
        channel.set_trace(trace, sampling_rate)
        station.add_channel(channel)

    if(phase):
        triggered = triggerSimulator.run(evt, station, det, 
                                         Vrms, #self._Vrms,
                                         threshold, 
                                         triggered_channels=channels,
                                         phasing_angles=default_angles,
                                         ref_index=1.75,
                                         trigger_name='primary_phasing',
                                         trigger_adc=False,
                                         adc_output='voltage',
                                         nyquist_zone=None,
                                         bandwidth_edge=20 * units.MHz, 
                                         upsampling_factor=upsampling_factor, 
                                         window=int(16 / (sampling_rate*upsampling_factor) ), 
                                         step = int(8  / (sampling_rate*upsampling_factor) ))
    else:
        thresholdSimulator.run(evt, station, det,
                               threshold=threshold,
                               triggered_channels=[4],  # run trigger on all channels
                               number_concidences=1,
                               trigger_name='simple_threshold')

        triggered = station.get_trigger('simple_threshold').has_triggered()
        
    return triggered

from multiprocessing import Pool as ThreadPool
from itertools import product

pool = ThreadPool(20)

for threshold in thresholds:
    n_triggers = 0
    i = 0
    t00 = time.time()
    t0 = time.time()

    while i < 10000: #True:

        print("Events:", i, " Delta t=", time.time() - t00, " N_triggers =", n_triggers)
        i += 1000
        t00 = time.time()

        #results = loop(0)

        results = pool.map(loop, zip(threshold * np.ones(1000), np.random.uniform(0.0, 1.0, 1000)))

        n_triggers += np.sum(results)
        rate = 1. * n_triggers / (i * n_samples * dt)

        '''
        if(n_triggers >= 10):
            rate = 1. * n_triggers / (i * n_samples * dt)
            with open(f"{pattern}.txt", "a") as fout:
                fout.write(f"{threshold}\t{n_triggers}\t{i*n_samples*dt}\t{rate}\n")
                fout.close()
            print(f"threshold = {threshold:.3f}: n_triggers = {n_triggers} -> rate = {rate/units.Hz:.0f} Hz, {(time.time() -  t0)/i*1000:.1f}ms per event -> {(time.time() -  t0)/n_triggers*100/60:.1f}min for 100 triggered events")
            break
        '''

    rate = 1. * n_triggers / (i * n_samples * dt)
    with open(f"{pattern}_halfwindow.txt", "a") as fout:
        fout.write(f"{threshold}\t{n_triggers}\t{i*n_samples*dt}\t{rate}\n")
        fout.close()
    print(f"threshold = {threshold:.3f}: n_triggers = {n_triggers} -> rate = {rate/units.Hz:.0f} Hz, {(time.time() -  t0)/i*1000:.1f}ms per event -> {(time.time() -  t0)/n_triggers*100/60:.1f}min for 100 triggered events")

