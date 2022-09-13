import numpy as np
import helper_cr_eff as hcr
import json
import os
import time
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.channelGalacticNoiseAdder
import NuRadioReco.modules.trigger.envelopeTrigger as envelopeTrigger
import NuRadioReco.modules.trigger.highLowThreshold as highLowThreshold
import NuRadioReco.modules.trigger.powerIntegration as powerIntegration
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.eventTypeIdentifier
import NuRadioReco.utilities.fft
from NuRadioReco.detector.generic_detector import GenericDetector
from NuRadioReco.utilities import units
import argparse
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

'''
This script calculates the trigger rate for different thresholds over a given number of iterations.
The resolution is calculated in the create_config_file.

For the galactic noise, the sky maps from PyGDSM are used. You can install it with 
pip install git+https://github.com/telegraphic/pygdsm .

The sampling rate has a huge influence on the threshold, because the trace has more time to exceed the threshold
for a sampling rate of 1GHz, 1955034 iterations yields a resolution of 0.5 Hz
Due to computational efficiency (galactic noise adder is slow), one amplitude is reused with 10 random phases

For one config file I used something on the cluster like:
for number in $(seq 0 1 110); do echo $number; qsub Cluster_ntr_2.sh --number $number; sleep 0.2; done;

for several config files I used:
FILES=output_threshold_calculation/*
for file in $FILES
do
echo "Processing $file"
  for number in $(seq 0 1 110)
  do echo $number
  qsub calculate_threshold_multi_job --config_file $file --number $number
  sleep 0.2
  done
done
'''

parser = argparse.ArgumentParser(description='Noise Trigger Rate')
parser.add_argument('--config_file', type=str, nargs='?',
                    help='input filename from which the calculation starts.')
parser.add_argument('--number', type=int, nargs='?', default=1,
                    help='Assigns a number to the scrip. '
                         'Check in config file for number_of_jobs '
                         'and sumbit the job with a for loop over the range of number_of_jobs')
parser.add_argument('--n_thresholds', type=int, nargs='?', default=15,
                    help='number of thresholds that will be tested.')
parser.add_argument('--output_path', type=str, nargs='?', default=os.path.dirname(__file__),
                    help='Path to save output, most likely the path to the cr_efficiency_analysis directory')

args = parser.parse_args()

with open(args.config_file, 'r') as fp:
    cfg = json.load(fp)

n_iterations_total = cfg['n_iterations_total']
trigger_thresholds = (np.arange(cfg['threshold_start'],
                                cfg['threshold_start'] + (args.n_thresholds * cfg['threshold_step']),
                                cfg['threshold_step']))

logger.info("Processing trigger thresholds {}".format(trigger_thresholds))

det = GenericDetector(json_filename=cfg['detector_file'], default_station=cfg['default_station'])
station_ids = det.get_station_ids()
channel_ids = det.get_channel_ids(station_ids[0])

event, station, channel = hcr.create_empty_event(
    det, cfg['trace_samples'], cfg['station_time'], cfg['station_time_random'], cfg['sampling_rate'])

eventTypeIdentifier = NuRadioReco.modules.eventTypeIdentifier.eventTypeIdentifier()

channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
channelGenericNoiseAdder.begin()

channelGalacticNoiseAdder = NuRadioReco.modules.channelGalacticNoiseAdder.channelGalacticNoiseAdder()
channelGalacticNoiseAdder.begin(
    n_side=cfg['galactic_noise_n_side'],
    interpolation_frequencies=np.arange(cfg['galactic_noise_interpolation_frequencies_start'],
                                        cfg['galactic_noise_interpolation_frequencies_stop'],
                                        cfg['galactic_noise_interpolation_frequencies_step']))

hardwareResponseIncorporator = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()

channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelBandPassFilter.begin()

t = time.time()  # absolute time of system
sampling_rate = station.get_channel(channel_ids[0]).get_sampling_rate()
dt = 1. / sampling_rate

time = channel.get_times()
channel_trace_start_time = time[0]
channel_trace_final_time = time[len(time) - 1]
channel_trace_time_interval = channel_trace_final_time - channel_trace_start_time

trigger_status = []
trigger_rate = []
trigger_efficiency = []

for n_it in range(n_iterations_total):
    station = event.get_station(cfg['default_station'])
    if cfg['station_time_random']:
        station = hcr.set_random_station_time(station, cfg['station_time'])

    eventTypeIdentifier.run(event, station, "forced", 'cosmic_ray')

    channel = hcr.create_empty_channel_trace(station, cfg['trace_samples'], sampling_rate)

    channelGenericNoiseAdder.run(event, station, det,
                                 amplitude=cfg['Vrms_thermal_noise'],
                                 min_freq=cfg['T_noise_min_freq'],
                                 max_freq=cfg['T_noise_max_freq'], type='rayleigh', bandwidth=None)

    channelGalacticNoiseAdder.run(event, station, det)

    if cfg['hardware_response']:
        hardwareResponseIncorporator.run(event, station, det, sim_to_data=True)

    for i_phase in range(cfg['n_random_phase']):
        channel = hcr.add_random_phase(station, sampling_rate)
        channelBandPassFilter.run(event, station, det, passband=cfg['passband_trigger'],
                                  filter_type='butter', order=cfg['order_trigger'])

        trace = station.get_channel(station.get_channel_ids()[0]).get_trace()
        trigger_status_all_thresholds = []
        for threshold in trigger_thresholds:
            if cfg['trigger_name'] == 'high_low':
                triggered_samples = highLowThreshold.get_high_low_triggers(trace, threshold, -threshold,
                                                                           cfg['coinc_window'], dt=dt)
                if True in triggered_samples:
                    has_triggered = bool(1)
                else:
                    has_triggered = bool(0)

            if cfg['trigger_name'] == 'envelope':
                triggered_samples = envelopeTrigger.get_envelope_triggers(trace, threshold)
                if True in triggered_samples:
                    has_triggered = bool(1)
                else:
                    has_triggered = bool(0)

            if cfg['trigger_name'] == 'power_integration':
                trace_1 = station.get_channel(1).get_trace()
                triggered_samples, int_power = powerIntegration.get_power_int_triggers(trace, threshold, cfg['int_window'], dt=dt, full_output=True)
                if True in triggered_samples:
                    has_triggered = bool(1)
                else:
                    has_triggered = bool(0)
                print('threshold', threshold)
                print('int_power', np.max(int_power))
                print(has_triggered)

            trigger_status_all_thresholds.append(has_triggered)
        trigger_status.append(trigger_status_all_thresholds)

trigger_status = np.array(trigger_status)
triggered_trigger = np.sum(trigger_status, axis=0)
trigger_efficiency = triggered_trigger / len(trigger_status)
trigger_rate = (1 / channel_trace_time_interval) * trigger_efficiency

logger.info("Triggered true per trigger thresholds {}/{}".format(triggered_trigger, len(trigger_status)))

dic = {'thresholds': trigger_thresholds, 'efficiency': trigger_efficiency, 'trigger_rate': trigger_rate,
        'T_noise': cfg['T_noise'], 'Vrms_thermal_noise': cfg['Vrms_thermal_noise'],
       'n_iterations_total': n_iterations_total, 'iterations_per_job': cfg['iterations_per_job'],
       'number_of_jobs': cfg['number_of_jobs'], 'target_single_trigger_rate': cfg['target_single_trigger_rate'],
       'target_global_trigger_rate': cfg['target_global_trigger_rate'], 'resolution': cfg['resolution'],
       'trigger_name': cfg['trigger_name'], 'passband_trigger': cfg['passband_trigger'],
       'total_number_triggered_channels': cfg['total_number_triggered_channels'],
       'number_coincidences': cfg['number_coincidences'], 'triggered_channels': cfg['triggered_channels'],
       'coinc_window': cfg['coinc_window'], 'order_trigger': cfg['order_trigger'],
       'detector_file': cfg['detector_file'], 'default_station': cfg['default_station'],
       'trace_samples': cfg['trace_samples'], 'sampling_rate': cfg['sampling_rate'],
       'trace_length': cfg['trace_length'], 'T_noise_min_freq': cfg['T_noise_min_freq'],
       'T_noise_max_freq': cfg['T_noise_max_freq'], 'galactic_noise_n_side': cfg['galactic_noise_n_side'],
       'galactic_noise_interpolation_frequencies_start': cfg['galactic_noise_interpolation_frequencies_start'],
       'galactic_noise_interpolation_frequencies_stop': cfg['galactic_noise_interpolation_frequencies_stop'],
       'galactic_noise_interpolation_frequencies_step': cfg['galactic_noise_interpolation_frequencies_step'],
       'station_time': cfg['station_time'], 'station_time_random': cfg['station_time_random'],
       'hardware_response': cfg['hardware_response'], 'n_random_phase': cfg['n_random_phase'],
       'threshold_start': cfg['threshold_start'], 'threshold_step': cfg['threshold_step']}

os.makedirs(os.path.join(args.output_path, 'output_threshold_calculation'), exist_ok=True)

output_file = 'output_threshold_calculation/{}_trigger_{:.0f}Hz_{}of{}_i{}_{}.json'.format(
    cfg['trigger_name'], cfg['target_global_trigger_rate'] / units.Hz,
    cfg['number_coincidences'], cfg['total_number_triggered_channels'], len(trigger_status), args.number)

abs_path_output_file = os.path.normpath(os.path.join(args.output_path, output_file))
with open(abs_path_output_file, 'w') as outfile:
    json.dump(dic, outfile, cls=hcr.NumpyEncoder, indent=4, sort_keys=True)
