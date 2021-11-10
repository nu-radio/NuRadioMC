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
parser.add_argument('--directory', type=str, nargs='?', default='output_threshold_calculation/', help='directory with output files')
parser.add_argument('--condition', type=str, nargs='?', default='i200', help='string which should be in dict name')
args = parser.parse_args()

logger.info(f"Checking {args.directory} for string {args.condition}")

file_list = []  # get non corrupted files from threshold calculations with specified passband
i = 0
for file in glob.glob('{}*.json'.format(args.directory)):
    if os.path.isfile(file) and str(args.condition) in file:
        i = i+1
        if os.path.getsize(file) > 0:
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

dic = {}
for key in cfg:
    dic[key] = cfg[key]
dic['iteration'] = iterations
dic['efficiency'] = trigger_efficiency_all
dic['trigger_rate'] = trigger_rate_all

nearest = hcr.find_nearest(trigger_rate_all, cfg['target_single_trigger_rate'])
index = np.where(trigger_rate_all == nearest)
final_threshold = thresholds[index[0][0]]

dic['final_threshold'] = final_threshold

os.makedirs('config/air_shower', exist_ok=True)

output_file = 'config/air_shower/final_config_{}_trigger_{}of{}_{:.2e}.json'.format(
    cfg['trigger_name'], cfg['number_coincidences'], cfg['total_number_triggered_channels'], final_threshold)

with open(output_file, 'w') as outfile:
    json.dump(dic, outfile, cls=hcr.NumpyEncoder, indent=4, sort_keys=True)
