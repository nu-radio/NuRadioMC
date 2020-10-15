import argparse
import time
import copy
from astropy.time import Time
from scipy import constants
from itertools import product
from matplotlib import pyplot as plt
import numpy as np
from multiprocessing import Pool as ThreadPool
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.phasedarray.triggerSimulator
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.framework.station
import NuRadioReco.framework.channel
from NuRadioReco.utilities import units
from NuRadioReco.utilities import fft
from NuRadioReco.detector import detector

det = detector.Detector(json_filename="../Effective_volume/8antennas_100m_0.5GHz.json")
det.update(Time.now())

channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()

n_channels = 1

main_low_angle = np.deg2rad(-59.55)
main_high_angle = np.deg2rad(59.55)
channels = []

if(n_channels == 4):
    upsampling_factor = 2
    default_angles = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 11))
    channels = np.arange(4, 8)
    phase = True
elif(n_channels == 8):
    upsampling_factor = 4
    default_angles = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 21))
    channels = None
    phase = True
elif(n_channels == 1):
    upsampling_factor = 1
    default_angles = []
    channels = [4]
    phase = False
else:
    print("wrong n_channels!")
    exit()

thresholds = (np.arange(1.5, 3.5, 0.1))
channel_ids = np.arange(8)
Vrms = 1
station_id = 1
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

pattern = f"pa_trigger_rate_{n_channels:d}channels_{upsampling_factor}xupsampiling"

triggerSimulator = NuRadioReco.modules.phasedarray.triggerSimulator.triggerSimulator()
thresholdSimulator = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()

def loop(zipped):

    threshold = float(zipped[0])
    seed = int(zipped[1])

    station = NuRadioReco.framework.station.Station(station_id)
    evt = NuRadioReco.framework.event.Event(0, 0)

    channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
    channelGenericNoiseAdder.begin(seed = seed)

    dt = 1.0 / sampling_rate

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
                                         Vrms,
                                         threshold * Vrms,
                                         triggered_channels=channels,
                                         phasing_angles=default_angles,
                                         ref_index=1.75,
                                         trigger_name='primary_phasing',
                                         trigger_adc=False,
                                         adc_output='voltage',
                                         trigger_filter=None,
                                         upsampling_factor=upsampling_factor,
                                         window=int(16 * sampling_rate * upsampling_factor),
                                         step = int(8 * sampling_rate * upsampling_factor))
    else:
        # code for a fake single antenna, integrated antenna

        original_trace = np.array(station.get_channel(4).get_trace())

        squared_mean, num_frames = triggerSimulator.powerSum(coh_sum=original_trace,
                                                             window=int(16 * sampling_rate),
                                                             step=int(8 * sampling_rate),
                                                             adc_output=f'voltage')

        squared_mean_threshold = 1.0 * np.power(threshold, 2.0)

        triggered = (True in (squared_mean > squared_mean_threshold))

    return triggered

pool = ThreadPool(20)

for threshold in thresholds:
    n_triggers = 0
    i = 0
    t00 = time.time()
    t0 = time.time()

    while i < 10000:
        n_pool = 1000

        print("Events:", i, " Delta t=", time.time() - t00, " N_triggers =", n_triggers)
        i += n_pool
        t00 = time.time()

        results = pool.map(loop, zip(threshold * np.ones(n_pool), i + np.arange(n_pool)))

        n_triggers += np.sum(results)
        rate = 1. * n_triggers / (i * n_samples * dt)

    rate = 1. * n_triggers / (i * n_samples * dt)
    with open(f"{pattern}.txt", "a") as fout:
        fout.write(f"{threshold}\t{n_triggers}\t{i*n_samples*dt}\t{rate}\n")
        fout.close()
    print(f"threshold = {threshold:.3f}: n_triggers = {n_triggers} -> rate = {rate/units.Hz:.0f} Hz, {(time.time() -  t0)/i*1000:.1f}ms per event -> {(time.time() -  t0)/n_triggers*100/60:.1f}min for 100 triggered events")
