import numpy as np
import helper_cr_eff as hcr
import matplotlib.pyplot as plt
import os
from NuRadioReco.utilities import units
import argparse
import json
import glob

'''This scripts plots the trigger rate for different thresholds 
as given by the final config files for the air shower calculations.

'''
parser = argparse.ArgumentParser(description='Noise Trigger Rate')
parser.add_argument('config_file', type=str, nargs='?', default='config/air_shower/final_config_envelope_trigger_3of4_2.35e-02_n1240.json', help='directory with output files')
parser.add_argument('save_plot', type=bool, nargs='?', default=True, help='decide if you want to save the plot')

args = parser.parse_args()

with open(args.config_file, 'r') as fp:
    cfg = json.load(fp)
from scipy.interpolate import interp1d

thresholds = np.array(cfg['thresholds'])
trigger_rate = np.array(cfg['trigger_rate'])

if cfg['hardware_response']:
    x = thresholds / units.millivolt
    y = trigger_rate / units.Hz
    f1 = interp1d(x, y, 'cubic')

    xnew = np.linspace(thresholds[0] / units.millivolt,
                       thresholds[-1] / units.millivolt)
    xlabel = 'Threshold [mV]'
    target_trigger_rate = cfg['target_single_trigger_rate'] / units.Hz

else:
    x = thresholds / units.microvolt
    y = trigger_rate / units.Hz
    f1 = interp1d(x, y, 'cubic')

    xnew = np.linspace(thresholds[0] / units.microvolt,
                       thresholds[-1] / units.microvolt)
    xlabel = r'Threshold [$\mu$V]'
    target_trigger_rate = cfg['target_single_trigger_rate'] / units.Hz



ynew = f1(xnew)
nearest = hcr.find_nearest(ynew, target_trigger_rate)
index = np.where(ynew == nearest)
thresh = xnew[index[0][0]]

plt.plot(x, y, marker='x', label='Noise trigger rate',
         linestyle='none')
plt.plot(xnew, f1(xnew), '--', label='interp1d f({:.2e}) = {:.2e} Hz'.format(thresh, f1(thresh)))
plt.title('{} trigger, {}/{}, trigger rate {} Hz'.format(cfg['trigger_name'],
                                                         cfg['number_coincidences'],
                                                         cfg['total_number_triggered_channels'],
                                                         cfg['target_global_trigger_rate']/units.Hz))
plt.xlabel(xlabel, fontsize=18)
plt.hlines(target_trigger_rate, x[0], x[-1], label='Target trigger rate one antenna {:.2e} Hz'.format(
    cfg['target_single_trigger_rate']/units.Hz))
plt.ylabel('Trigger rate [Hz]', fontsize=18)
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=16)
plt.yscale('log')
plt.ylim(target_trigger_rate*1e-2)
plt.legend()
plt.tight_layout()
if args.save_plot:
    plt.savefig(
        'results/fig_ntr_{}_{}of{}_{:.0f}Hz_threshold_{:.2e}.png'.format(cfg['trigger_name'],
                                                                         cfg['number_coincidences'],
                                                                         cfg['total_number_triggered_channels'],
                                                                         cfg['target_global_trigger_rate']/units.Hz,
                                                                         cfg['final_threshold']))
plt.show()
plt.close()