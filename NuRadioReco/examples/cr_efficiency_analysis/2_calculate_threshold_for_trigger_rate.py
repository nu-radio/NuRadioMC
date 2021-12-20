import numpy as np
import helper_cr_eff as hcr
import time
import json
import os
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.channelGalacticNoiseAdder
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.trigger.envelopeTrigger as envelopeTrigger
import NuRadioReco.modules.trigger.highLowThreshold as highLowThreshold
import NuRadioReco.modules.trigger.powerIntegration as powerIntegration
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
import NuRadioReco.modules.eventTypeIdentifier
from NuRadioReco.detector.generic_detector import GenericDetector
from NuRadioReco.utilities import units
import argparse
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

'''
This script calculates the trigger threshold for a given global trigger rate.
The trigger rate for a single antenna is calculated in the crate_config_file.py
Afterwards the threshold is increased incrementally until the target trigger rate is achieved.
This script is slower than 2_I_calculate_trigger_rate_for_threshold, where the trigger rate
is calculated for a given array of thresholds.

For the galactic noise, the sky maps from PyGDSM are used. You can install it with 
pip install git+https://github.com/telegraphic/pygdsm .

The sampling rate has a huge influence on the threshold, because the trace has more time to exceed the threshold
for a sampling rate of 1GHz, 1955034 iterations yields a resolution of 0.5 Hz
Due to computational efficiency (galactic noise adder is slow), one amplitude is reused with 10 random phases
'''

parser = argparse.ArgumentParser(description='Noise Trigger Rate')
parser.add_argument('--config_file', type=str, nargs='?',
                    help='input filename from which the calculation starts.')
parser.add_argument('--output_path', type=str, nargs='?', default=os.path.dirname(__file__),
                    help='Path to save output, most likely the path to the cr_efficiency_analysis directory')

args = parser.parse_args()

with open(args.config_file, 'r') as fp:
    cfg = json.load(fp)

det = GenericDetector(json_filename=cfg['detector_file'], default_station=cfg['default_station'])
station_ids = det.get_station_ids()
channel_ids = det.get_channel_ids(station_ids[0])

logger.info("Apply {} trigger with passband {} "
            .format(cfg['trigger_name'], np.array(cfg['passband_trigger']) / units.MHz))

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

time = station.get_channel(channel_ids[0]).get_times()
channel_trace_start_time = time[0]
channel_trace_final_time = time[len(time) - 1]
channel_trace_time_interval = channel_trace_final_time - channel_trace_start_time

trigger_status = []
triggered_trigger = []
trigger_rate = []
trigger_efficiency = []
thresholds = []
iterations = []
channel_rms = []
channel_sigma = []

n_thres = 0
sum_trigger = cfg['number_of_allowed_trigger'] + 1
while sum_trigger > cfg['number_of_allowed_trigger']:
    # with each iteration the threshold increases one step
    threshold = cfg['threshold_start'] + (n_thres * cfg['threshold_step'])
    thresholds.append(threshold)
    logger.info("Processing threshold {:.2e} V".format(threshold))
    trigger_status_per_all_it = []

    for n_it in range(cfg['n_iterations_total']):
        station = event.get_station(cfg['default_station'])
        eventTypeIdentifier.run(event, station, "forced", 'cosmic_ray')

        # here an empty channel trace is created
        channel = hcr.create_empty_channel_trace(station, cfg['trace_samples'], sampling_rate)

        # thermal and galactic noise is added
        channelGenericNoiseAdder.run(event, station, det, amplitude=cfg['Vrms_thermal_noise'],
                                     min_freq=cfg['T_noise_min_freq'], max_freq=cfg['T_noise_max_freq'],
                                     type='rayleigh')
        channelGalacticNoiseAdder.run(event, station, det)

        # includes the amplifier response, if set true at the beginning
        if cfg['hardware_response']:
            hardwareResponseIncorporator.run(event, station, det, sim_to_data=True)

        channelBandPassFilter.run(event, station, det, passband=cfg['passband_trigger'],
                                      filter_type='butter', order=cfg['order_trigger'])

        # This loop changes the phase of a trace with rand_phase, this is because the GalacticNoiseAdder
        # needs some time and one amplitude is good enough for several traces.
        # The current number of iteration can be calculated with i_phase + n_it*n_random_phase
        for i_phase in range(cfg['n_random_phase']):
            trigger_status_one_it = []
            channel = hcr.add_random_phase(station, sampling_rate)

            trace = station.get_channel(station.get_channel_ids()[0]).get_trace()

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
                triggered_samples = powerIntegration.get_power_int_triggers(trace, threshold, cfg['int_window'], dt=dt, full_output=False)
                if True in triggered_samples:
                    has_triggered = bool(1)
                else:
                    has_triggered = bool(0)

            # trigger status for all iteration
            trigger_status_per_all_it.append(has_triggered)
            sum_trigger = np.sum(trigger_status_per_all_it)

    # here it is checked, how many of the triggers in n_iteration are triggered true.
    # If it is more than number of allowed trigger, the threshold is increased with n_thres.
    if np.sum(trigger_status_per_all_it) > cfg['number_of_allowed_trigger']:
        number_of_trigger = np.sum(trigger_status_per_all_it)
        trigger_efficiency_per_tt = np.sum(trigger_status_per_all_it) / len(trigger_status_per_all_it)
        trigger_rate_per_tt = (1 / channel_trace_time_interval) * trigger_efficiency_per_tt

        trigger_rate.append(trigger_rate_per_tt)
        trigger_efficiency.append(trigger_efficiency_per_tt)
        estimated_global_rate = hcr.get_global_trigger_rate(trigger_rate_per_tt, cfg['total_number_triggered_channels'],
                                                            cfg['number_coincidences'], cfg['coinc_window'])
        logger.info("current trigger rate {:.2e} Hz at threshold {:.2e} V"
                    .format(trigger_rate_per_tt / units.Hz, threshold))
        logger.info("target trigger rate single:{:.2e} Hz, global:{:.2e} Hz "
                    .format(cfg['target_single_trigger_rate'] / units.Hz, cfg['target_global_trigger_rate'] / units.Hz))
        logger.info("continue".format(cfg['trigger_name']))
        n_thres += 1

    elif n_it == (cfg['n_iterations_total'] - 1):
        number_of_trigger = np.sum(trigger_status_per_all_it)
        trigger_efficiency_per_tt = np.sum(trigger_status_per_all_it) / len(trigger_status_per_all_it)
        trigger_rate_per_tt = (1 / channel_trace_time_interval) * trigger_efficiency_per_tt

        trigger_rate.append(trigger_rate_per_tt)
        trigger_efficiency.append(trigger_efficiency_per_tt)
        estimated_global_rate = hcr.get_global_trigger_rate(trigger_rate_per_tt, cfg['total_number_triggered_channels'],
                                                            cfg['number_coincidences'], cfg['coinc_window'])
        logger.info("checked {} thresholds".format(n_thres))
        logger.info("current trigger rate {:.2e} Hz at threshold {:.2e} V"
                    .format(trigger_rate_per_tt / units.Hz, threshold))
        logger.info("resolution for single trigger rate {:.2e} Hz"
                    .format(cfg['resolution'] / units.Hz))
        logger.info("estimated global trigger rate {:.2e} Hz"
                    .format(estimated_global_rate / units.Hz))
        logger.info("target trigger rate single:{:.2e} Hz, global:{:.2e} Hz"
                    .format(cfg['target_single_trigger_rate'] / units.Hz, cfg['target_global_trigger_rate'] / units.Hz))

        dic = {'thresholds': thresholds,
               'efficiency': trigger_efficiency,
               'trigger_rate': trigger_rate,
               'estimated_global_trigger_rate': estimated_global_rate,
               'final_threshold': thresholds[-1],
               'T_noise': cfg['T_noise'],
               'Vrms_thermal_noise': cfg['Vrms_thermal_noise'],
               'n_iterations_total': cfg['n_iterations_total'],
               'iterations_per_job': cfg['iterations_per_job'],
               'number_of_jobs': cfg['number_of_jobs'],
               'target_single_trigger_rate': cfg['target_single_trigger_rate'],
               'target_global_trigger_rate': cfg['target_global_trigger_rate'],
               'resolution': cfg['resolution'],
               'trigger_name': cfg['trigger_name'],
               'passband_trigger': cfg['passband_trigger'],
               'total_number_triggered_channels': cfg['total_number_triggered_channels'],
               'number_coincidences': cfg['number_coincidences'],
               'triggered_channels': cfg['triggered_channels'],
               'coinc_window': cfg['coinc_window'],
               'order_trigger': cfg['order_trigger'],
               'detector_file': cfg['detector_file'],
               'default_station': cfg['default_station'],
               'trace_samples': cfg['trace_samples'],
               'sampling_rate': cfg['sampling_rate'],
               'trace_length': cfg['trace_length'],
               'T_noise_min_freq': cfg['T_noise_min_freq'],
               'T_noise_max_freq ': cfg['T_noise_max_freq'],
               'galactic_noise_n_side': cfg['galactic_noise_n_side'],
               'galactic_noise_interpolation_frequencies_start': cfg['galactic_noise_interpolation_frequencies_start'],
               'galactic_noise_interpolation_frequencies_stop': cfg['galactic_noise_interpolation_frequencies_stop'],
               'galactic_noise_interpolation_frequencies_step': cfg['galactic_noise_interpolation_frequencies_step'],
               'station_time': cfg['station_time'],
               'station_time_random': cfg['station_time_random'],
               'hardware_response': cfg['hardware_response'],
               'n_random_phase': cfg['n_random_phase'],
               'threshold_start': cfg['threshold_start'],
               'threshold_step': cfg['threshold_step']
               }

        os.makedirs(os.path.join(args.output_path, 'config/air_shower'), exist_ok=True)

        output_file = 'config/air_shower/final_config_{}_trigger_{:.2e}_{}of{}_{:.0f}Hz.json'.format(
            cfg['trigger_name'], thresholds[-1], cfg['target_global_trigger_rate']/ units.Hz,
            cfg['number_coincidences'], cfg['total_number_triggered_channels'])
        abs_path_output_file = os.path.normpath(os.path.join(args.output_path, output_file))

        with open(abs_path_output_file, 'w') as outfile:
            json.dump(dic, outfile, cls=hcr.NumpyEncoder, indent=4, sort_keys=True)
