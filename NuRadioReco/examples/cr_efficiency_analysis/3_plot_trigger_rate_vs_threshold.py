import numpy as np
import helper_cr_eff as hcr
import matplotlib.pyplot as plt
import os
from NuRadioReco.utilities import units
import argparse
import json

'''This scripts plots the trigger rate for different thresholds 
as given by the final config files for the air shower calculations.
'''

parser = argparse.ArgumentParser(description='Noise Trigger Rate')
parser.add_argument('--config_directory', type=str, nargs='?', default='config/air_shower_envelope_trigger_1Hz_2of3_22.46mV/',
                    help='directory with final config files')
parser.add_argument('--save_plot', type=bool, nargs='?', default=True, help='decide if you want to save the plot')

args = parser.parse_args()

for config_file in os.listdir(args.config_directory):
    with open(os.path.join(args.config_directory, config_file), 'r') as fp:
        cfg = json.load(fp)
    from scipy.interpolate import interp1d

    thresholds = np.array(cfg['thresholds'])

    #  choose between a single antenna and global trigger rate
    trigger_rate = np.array(cfg['estimated_global_trigger_rate'])


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
    thresh = xnew[np.argmin(ynew - target_trigger_rate)]


    plt.plot(x, y, marker='x', label='Noise trigger rate',
             linestyle='none')
    plt.plot(xnew, f1(xnew), '--', label='interp1d f({:.2e}) = {:.2e} Hz'.format(thresh, f1(thresh)))
    plt.title('{} trigger, {}/{}, trigger rate {} Hz'.format(cfg['trigger_name'],
                                                             cfg['number_coincidences'],
                                                             cfg['total_number_triggered_channels'],
                                                             cfg['target_global_trigger_rate'] / units.Hz))
    plt.xlabel(xlabel, fontsize=18)
    plt.hlines(target_trigger_rate, x[0], x[-1], label='Target trigger rate one antenna {:.2e} Hz'.format(
        cfg['target_single_trigger_rate'] / units.Hz))
    plt.ylabel('Trigger rate [Hz]', fontsize=18)
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.yscale('log')
    plt.ylim(target_trigger_rate * 1e-2)
    plt.legend()
    plt.tight_layout()

    if args.save_plot:
        os.makedirs('results/plots', exist_ok=True)
        if cfg['hardware_response']:
            plt.savefig(
                'results/plots/fig_ntr_{}_{:.0f}Hz_{}of{}_threshold_{:.2f}mV.png'.format(
                    cfg['trigger_name'],
                    cfg['target_global_trigger_rate'] / units.Hz,
                    cfg['number_coincidences'],
                    cfg['total_number_triggered_channels'],
                    cfg['final_threshold'] / units.millivolt))
        else:
            plt.savefig(
                'results/plots/fig_ntr_{}_{:.0f}Hz_{}of{}_threshold_{:.2e}V.png'.format(
                    cfg['trigger_name'],
                    cfg['target_global_trigger_rate'] / units.Hz,
                    cfg['number_coincidences'],
                    cfg['total_number_triggered_channels'],
                    cfg['final_threshold']))
    plt.show()
    plt.close()
