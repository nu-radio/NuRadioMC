import numpy as np
import helper_cr_eff as hcr
import matplotlib.pyplot as plt
import os
from NuRadioReco.utilities import units
import argparse
import json
import glob
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

'''
Use the script calculate trigger_rate_for_threshold.py to create the files.
This script reads all files in output_treshold_calculations and writes one new dict which then contains the
information of all files. This is necessary if you want to split the total number of iterations in different job. 
'''

parser = argparse.ArgumentParser(description='Noise Trigger Rate')
parser.add_argument('--directory', type=str, nargs='?',
                    default='output_threshold_calculation/', help='directory with output files')
parser.add_argument('--condition', type=str, nargs='?',
                    default='envelope_trigger_0Hz_3of4', help='string which should be in dict name')
args = parser.parse_args()

logger.info(f"Checking {args.directory} for condition: {args.condition}")

file_list = []
# get non corrupted files from threshold calculations with specified passband
for file in glob.glob(args.directory + '*' + args.condition + '*.json'):
    if os.path.isfile(file) and os.path.getsize(file) > 0:
        file_list.append(file)


n_files = len(file_list)

logger.info(f"Using files {file_list}")

# open one file to check the number of tested thresholds
with open(file_list[0], 'r') as fp:
    cfg = json.load(fp)

thresholds = cfg['thresholds']

trigger_efficiency = np.zeros((n_files, len(thresholds)))
trigger_rate = np.zeros((n_files, len(thresholds)))

for i_file, filename in enumerate(file_list):
    with open(filename, 'r') as fp:
        data = json.load(fp)
    trigger_efficiency[i_file] = data['efficiency']
    trigger_rate[i_file] = data['trigger_rate']

trigger_efficiency_all = np.sum(trigger_efficiency, axis=0) / n_files
trigger_rate_all = np.sum(trigger_rate, axis=0) / n_files
iterations = cfg['n_iterations_total'] * cfg['n_random_phase'] * n_files
estimated_global_rate = hcr.get_global_trigger_rate(trigger_rate_all, cfg['total_number_triggered_channels'],
                                                            cfg['number_coincidences'], cfg['coinc_window'])

dic = {}
dic = cfg.copy()
dic['iteration'] = iterations
dic['efficiency'] = trigger_efficiency_all
dic['trigger_rate'] = trigger_rate_all
dic['estimated_global_trigger_rate'] = estimated_global_rate

final_threshold = thresholds[np.argmin(trigger_rate_all - cfg['target_single_trigger_rate'])]

dic['final_threshold'] = final_threshold

os.makedirs('config/air_shower', exist_ok=True)

if cfg['hardware_response']:
    output_file = 'config/air_shower/final_config_{}_trigger_{:.0f}Hz_{}of{}_{:.2f}mV.json'.format(
        cfg['trigger_name'], cfg['target_global_trigger_rate']/units.Hz,
        cfg['number_coincidences'], cfg['total_number_triggered_channels'],
        final_threshold/units.millivolt)
else:
    output_file = 'config/air_shower/final_config_{}_trigger_{:.0f}Hz_{}of{}_{:.2e}V.json'.format(
        cfg['trigger_name'], cfg['target_global_trigger_rate']/units.Hz,
        cfg['number_coincidences'], cfg['total_number_triggered_channels'], final_threshold)

with open(output_file, 'w') as outfile:
    json.dump(dic, outfile, cls=hcr.NumpyEncoder, indent=4, sort_keys=True)
