import numpy as np
import helper_cr_eff as hcr
import matplotlib.pyplot as plt
import os
from NuRadioReco.utilities import units
import argparse
import json
import glob

'''This script reads all dicts for different job files and writes one new dict which then contains the 
information of all files.
The last part creates a plot of the final dict. If you just want to plot the dict, comment the first part.
'''
parser = argparse.ArgumentParser(description='Noise Trigger Rate')
parser.add_argument('passband_low', type=int, nargs='?', default=80, help='passband low to check in MHz')
parser.add_argument('passband_high', type=int, nargs='?', default=180, help='passband high to check in MHz')
parser.add_argument('directory', type=str, nargs='?', default='/Users/lilly/Software/NuRadioMC/NuRadioReco/examples/cr_efficiency_analysis/output_threshold_calculation/', help='directory with output files')
parser.add_argument('create_plot', type=bool, nargs='?', default=True, help='decide if you want to plot the thresholds')

args = parser.parse_args()
passband_low = args.passband_low * units.MHz
passband_high = args.passband_high * units.MHz

file_list = []  # get non corrupted files from threshold calculations with specified passband
i = 0
for file in glob.glob('{}*.json'.format(args.directory)):
    if os.path.isfile(file) and str(int(passband_low/units.MHz)) + '_' + str(int(passband_high/units.MHz)) in file:
        i = i+1
        if os.path.getsize(file) > 0:
            file_list.append(file)

n_files = len(file_list)

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

print(cfg['target_single_trigger_rate'])
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

os.makedirs('config/', exist_ok=True)

output_file = 'config/final_config_{}_trigger_{:.2e}_pb_{:.0f}_{:.0f}_n{}.json'.format(
    cfg['trigger_name'], final_threshold, cfg['passband_trigger'][0] / units.MHz,
                                         cfg['passband_trigger'][1] / units.MHz,
                                         cfg['n_iterations_total'] * cfg['n_random_phase'])

with open(output_file, 'w') as outfile:
    json.dump(dic, outfile, cls=hcr.NumpyEncoder, indent=4, sort_keys=True)

if args.create_plot:
    with open(output_file, 'r') as fp:
        result = json.load(fp)
    from scipy.interpolate import interp1d

    thresholds = np.array(result['thresholds'])
    trigger_rate = np.array(result['trigger_rate'])

    if result['hardware_response']:
        x = thresholds / units.millivolt
        y = trigger_rate / units.Hz
        f1 = interp1d(x, y, 'cubic')

        xnew = np.linspace(thresholds[0] / units.millivolt,
                           thresholds[-1] / units.millivolt)
        xlabel = 'Threshold [mV]'
        target_trigger_rate = result['target_single_trigger_rate'] / units.millivolt

    else:
        x = thresholds / units.microvolt
        y = trigger_rate / units.Hz
        f1 = interp1d(x, y, 'cubic')

        xnew = np.linspace(thresholds[0] / units.microvolt,
                           thresholds[-1] / units.microvolt)
        xlabel = r'Threshold [$\mu$V]'
        target_trigger_rate = result['target_single_trigger_rate'] / units.microvolt



    ynew = f1(xnew)
    nearest = hcr.find_nearest(ynew, target_trigger_rate)
    index = np.where(ynew == nearest)
    thresh = xnew[index[0][0]]

    plt.plot(x, y, marker='x', label='Noise trigger rate',
             linestyle='none')
    plt.plot(xnew, f1(xnew), '--', label='interp1d f({:.2e}) = {:.2e} Hz'.format(thresh, f1(thresh)))
    plt.title('{}, passband = {} MHz, iterations = {:.1e}'.format(result['trigger_name'], np.array(result['passband_trigger']) / units.megahertz,
                                                                  iterations))
    plt.xlabel(xlabel, fontsize=18)
    plt.hlines(target_trigger_rate, x[0], x[-1], label='Target trigger rate one antenna {:.2e} Hz'.format(result['target_single_trigger_rate']/units.Hz))
    plt.ylabel('Trigger rate [Hz]', fontsize=18)
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        'results/fig_ntr_{}_passband_{:.0f}_{:.0f}.png'.format(result['trigger_name'], result['passband_trigger'][0] / units.megahertz,
                                                               result['passband_trigger'][1] / units.megahertz))
    plt.show()
    plt.close()