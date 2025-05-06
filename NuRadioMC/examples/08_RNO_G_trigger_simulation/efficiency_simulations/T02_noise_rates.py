import argparse
import time
import numpy as np
from astropy.time import Time
from multiprocessing import Pool as ThreadPool
from NuRadioMC.simulation import simulation
import os
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.framework.station
import NuRadioReco.framework.channel
from NuRadioReco.utilities import units
import copy
import datetime
from NuRadioReco.utilities import fft
from NuRadioReco.detector import detector
import matplotlib.pyplot as plt
import logging
from NuRadioReco.modules.RNO_G import hardwareResponseIncorporator, triggerBoardResponse
from NuRadioReco.detector.RNO_G import rnog_detector
import datetime as dt
import matplotlib.pyplot as plt

from NuRadioReco.modules.trigger.highLowThreshold import triggerSimulator as highLowTrigger
from NuRadioReco.modules.phasedarray.beamformedPowerIntegrationTrigger import triggerSimulator as powerTrigger
from NuRadioReco.modules.phasedarray.digitalBeamformedEnvelopeTrigger import triggerSimulator as envelopeTrigger

from scipy import constants

rnogHardwareResponse = hardwareResponseIncorporator.hardwareResponseIncorporator()
rnogHardwareResponse.begin()
rnogADCResponse = triggerBoardResponse.triggerBoardResponse(log_level=logging.ERROR)
rnogADCResponse.begin(clock_offset=0.0, adc_output="counts")




parser = argparse.ArgumentParser(description='calculates noise trigger rate for phased array')
parser.add_argument('--ntries', type=int, help='number noise traces to which a trigger is applied for each threshold', default=10000)
parser.add_argument('--ncpus', type=int, help='number of parallel jobs that can be run', default=8)
parser.add_argument('--detectordescription', type=str, help='path to file containing the detector description', default='../RNO_single_station_only_PA.json')
parser.add_argument('--station_id', type=int, help='station_id', default=0)

parser.add_argument('--threshold', type=float, help='threshold to test. If -1, runs a default sweep', default=-1.0)
parser.add_argument('--trigger', type=str, help='trigger type', default="power")
parser.add_argument('--upsampling_factor', type=int, help='upsampling factor', default=4)
parser.add_argument('--window', type=int, help='power integration window in units of unsamping_factor*1 samples', default=24)
parser.add_argument('--step', type=int, help='power trigger step in units of upsampling_factor*1 samples', default=4)
parser.add_argument('--beam_number', type=int, help='', default=12)
args = parser.parse_args()

# Get the detector info
det_file = args.detectordescription

det_defaults = {
    "trigger_adc_sampling_frequency": 0.472,
    "trigger_adc_nbits": 8,
    "trigger_adc_noise_count": 5,
    "trigger_adc_min_voltage": -1,
    "trigger_adc_max_voltage": 1,
}

if args.station_id!=0:
    station_id=args.station_id
    print("Getting measured, database, detector description")
    det = rnog_detector.Detector(
        detector_file=None, log_level=logging.INFO,
        always_query_entire_description=False, select_stations=station_id,
        over_write_handset_values=det_defaults)

    det.update(dt.datetime(2023, 8, 3))
    measured_response=True
else:
    print("Using place holder station description")
    station_id=11
    det = detector.Detector(source='json',json_filename=det_file)
    det.update(Time.now())
    measured_response=False


# Get the trigger info
channels = [0, 1, 2, 3]
main_low_angle = np.deg2rad(-60)
main_high_angle = np.deg2rad(60)
phasing_angles = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), args.beam_number))

if args.trigger == "power":
    trigger = powerTrigger()
    trig_kwargs = triggerBoardResponse.powerTriggerKwargs
    trig_kwargs["window"] = args.window
    trig_kwargs["step"] = args.step
    trig_kwargs["upsampling_factor"] = args.upsampling_factor
    trig_kwargs["phasing_angles"] = phasing_angles
    threshold_scan = np.arange(4, 13, .25)
    pattern = f"station{args.station_id}_{args.trigger}_{args.upsampling_factor}x_{args.window}win_{args.step}step"
    print(pattern)

elif args.trigger == "envelope":
    trigger = envelopeTrigger()
    trig_kwargs = triggerBoardResponse.envelopeTriggerKwargs
    trig_kwargs["upsampling_factor"] = args.upsampling_factor
    trig_kwargs["phasing_angles"] = phasing_angles
    threshold_scan = np.arange(3, 10, .25)
    pattern = f"{args.trigger}_{args.upsampling_factor}x"
    print(pattern)

elif args.trigger == "highlow":
    trigger = highLowTrigger()
    trig_kwargs=triggerBoardResponse.highLowTriggerKwargs
    threshold_scan = np.arange(1, 5, .25)
    pattern = f"{args.trigger}"
    print(pattern)

else:
    raise ValueError("Provide valid trigger type: power, envelope, highlow")

# Check if single threshold of perform a scan
if(args.threshold == -1.0):
    thresholds = threshold_scan
else:
    thresholds = np.array([float(args.thresholds)])

# Misc command line args
ntrials = args.ntries
ncpus = args.ncpus


# Get signal chain responses and calculate the noise rms values
n_samples = 1024
sampling_rate = .472*units.GHz
print(f"sampling rate = {sampling_rate/units.MHz}MHz, {n_samples} samples")

dt = 1 / sampling_rate
ff = np.fft.rfftfreq(n_samples, dt)

max_freq = ff[-1]
min_freq = 0
fff = np.linspace(min_freq, max_freq, 10000)
four_filters={}
four_filters_highres={}
noise_temp=300
Vrms_ratio = {}
amplitude = {}
Vrms = 1
per_channel_vrms=[]

for i in channels:
    if measured_response:
        four_filters[i]=det.get_signal_chain_response(station_id=station_id, channel_id=i, trigger=True)(ff)
        four_filters_highres[i]=det.get_signal_chain_response(station_id=station_id, channel_id=i, trigger=True)(fff)

    else:
        four_filters[i]=rnogHardwareResponse.get_filter(ff,11,0,det,sim_to_data=True,is_trigger=True)
        four_filters_highres[i]=rnogHardwareResponse.get_filter(fff,11,0,det,sim_to_data=True,is_trigger=True)

    integrated_channel_response = np.trapz(np.abs(four_filters_highres[i]) ** 2, fff)
    rel_channel_response=np.trapz(np.abs(four_filters_highres[i]) ** 2, fff)
    Vrms_ratio[i] = np.sqrt(rel_channel_response / (max_freq - min_freq))    
    chan_vrms=(noise_temp * 50 * constants.k * integrated_channel_response / units.Hz) ** 0.5
    per_channel_vrms.append(chan_vrms)
    amplitude[i] = chan_vrms / Vrms_ratio[i]

def loop(zipped):
    threshold = float(zipped[0])
    seed = int(zipped[1])

    # Generate dummy station with a fake event to be filled with noise traces.
    station = NuRadioReco.framework.station.Station(station_id)
    evt = NuRadioReco.framework.event.Event(0, 0)

    channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
    channelGenericNoiseAdder.begin(seed=seed)

    # Fill traces with noise.
    for channel_id in channels:
        spectrum = channelGenericNoiseAdder.bandlimited_noise(min_freq, max_freq, n_samples, sampling_rate, amplitude[channel_id],
                                                              type="rayleigh", time_domain=False)

        trace=fft.freq2time(spectrum*four_filters[channel_id], sampling_rate)
        channel = NuRadioReco.framework.channel.Channel(channel_id)
        channel.set_trace(trace, sampling_rate)
        channel.set_trigger_channel(copy.deepcopy(channel))
        station.add_channel(channel)

    # Gain normaliztion and digitization.
    chan_rms, gains = rnogADCResponse.run(evt, station, det, vrms=per_channel_vrms, trigger_channels=channels, apply_adc_gain=True, digitize_trace=True)
    
    # Calculate thresholds.
    power_vrms=0
    threshold_high={}
    threshold_low={}
    for i in channels:
        power_vrms+=chan_rms[i]**2
        threshold_high[i]=np.rint(chan_rms[i]*threshold)
        threshold_low[i]=-np.rint(chan_rms[i]*threshold)
    power_vrms=power_vrms
    voltage_rms=np.sqrt(power_vrms)

    if args.trigger=="power":
        trigger_threshold=np.rint(power_vrms*threshold)
    elif args.trigger=="envelope":
        trigger_threshold=np.rint(voltage_rms*threshold)
    else: trigger_threshold=None

    # Run trigger.
    has_triggered=trigger.run(
            evt,
            station,
            det,
            Vrms=None,
            threshold=trigger_threshold,
            threshold_high=threshold_high,
            threshold_low=threshold_high,
            trigger_channels=channels,
            trigger_name=f"{args.trigger}",
            **trig_kwargs
        )
    
    return has_triggered


if not os.path.exists("data/noise"):
    os.mkdir("data/noise")
if os.path.exists(f"data/noise/{pattern}.txt"):
    os.remove(f"data/noise/{pattern}.txt")

pool = ThreadPool(ncpus)

for threshold in thresholds:
    n_triggers = 0
    i = 0
    t00 = time.time()
    t0 = time.time()
    print(f"running threshold @ {threshold:.2f}")
    while i < ntrials:
        n_pool = int(float(ntrials) / 10.0)

        print("Events:", i, " Delta t=", time.time() - t00, " N_triggers =", n_triggers)
        i += n_pool
        t00 = time.time()

        results = pool.map(loop, zip(threshold * np.ones(n_pool), np.random.get_state()[1][0] + i + np.arange(n_pool)))

        n_triggers += np.sum(results)
        rate = 1. * n_triggers / (i * n_samples * dt)

    rate = 1. * n_triggers / (i * n_samples * dt)

    with open(f"data/noise/{pattern}.txt", "a") as fout:
        fout.write(f"{threshold}\t{n_triggers}\t{i*n_samples*dt}\t{rate}\n")
        fout.close()
    print(f"threshold = {threshold:.3f}: n_triggers = {n_triggers} -> rate = {rate/units.Hz:.0f} Hz, {(time.time() -  t0)/i*1000:.1f}ms per event -> {(time.time() -  t0)/n_triggers*100/60:.1f}min for 100 triggered events\n")
