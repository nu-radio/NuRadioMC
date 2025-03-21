import numpy as np
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units
import argparse
import os



parser = argparse.ArgumentParser()
parser.add_argument('--hardware_number', type=int, default = 7)# default is for station 21
args = parser.parse_args()
hardware_number = args.hardware_number


coax_names = []
# station 21: hardware number 7, station 22: hardware number 6, station 11: hardware number 5
# if you need the surface s21 data, ask Dan or Zack.
for file in os.listdir("data/station_{}".format(hardware_number)):
    if file.endswith("LOG.csv"):
        file = file.replace("S21_SRX{}_".format(hardware_number), '')
        coax_names.append(file.replace("_LOG.csv", ''))


## dict ({hardwarenumber: {cable number: number of jumper cables}}) with number of jumper cables (taken from: https://radio.uchicago.edu/wiki/index.php/File:ChannelMapping_v1.xlsx)

jumper_cables = {'5': {'1':0,'2': 0, '3': 0 , '4': 0 , '5': 0 , '6': 0 , '7': 0 , '8': 0 , '9':0  },
'6': {'1': 1, '2': 0, '3':2  , '4': 2 , '5': 0 , '6':2  , '7': 1 , '8': 0 , '9': 1 },
'7': {'1': 1, '2': 0, '3': 1 , '4':1  , '5':0  , '6':1  , '7':1  , '8':0  , '9':1  }
}

jumper_cable_delay = 3.97*0.89 #3.97 ns/m #cables of 35 inch/0.89 m, https://www.timesmicrowave.com/DataSheets/CableProducts/LMR-240.pdf

fig1 = plt.figure(figsize=(9, 30))
fig2 = plt.figure(figsize=(9, 30))
for i_coax, coax_name in enumerate(coax_names):
    log_mag_data = np.genfromtxt(
        'data/station_{}/S21_SRX{}_{}_LOG.csv'.format(hardware_number, hardware_number, coax_name),
        delimiter=',',
        skip_header=17,
        skip_footer=1
    )
    mag_freqs = log_mag_data[:, 0] * units.Hz
    log_magnitude = log_mag_data[:, 1]
    ax1_1 = fig1.add_subplot(len(coax_names), 1, i_coax + 1)
    ax1_1.grid()
    ax1_1.plot(
        mag_freqs / units.MHz,
        np.power(10., -log_magnitude)
    )
    phase_data = np.genfromtxt(
        'data/station_{}/S21_SRX{}_{}_PHASE.csv'.format(hardware_number, hardware_number, coax_name),
        delimiter=',',
        skip_header=17,
        skip_footer=1
    )
    phase_freqs = phase_data[:, 0] * units.Hz
    phase = np.unwrap(phase_data[:, 1] * units.deg)
    group_delay = -.5 * np.diff(phase) / np.diff(phase_freqs) / np.pi
    # phase = (phase_data[:, 1] * units.deg)
    freq_mask = phase_freqs > 25 * units.MHz
    ax2_1 = fig2.add_subplot(len(coax_names), 2, 2 * i_coax + 1)
    ax2_1.grid()
    ax2_1.plot(
        phase_freqs[freq_mask] / units.MHz,
        phase[freq_mask]
    )
    line_fit = np.polyfit(
        phase_freqs[freq_mask] * 2. * np.pi,
        - phase[freq_mask],
        1
    )
    ax2_1.set_xlabel('f [MHz]')
    ax2_1.set_ylabel('phase [rad]')
    ax2_2 = fig2.add_subplot(len(coax_names), 2, 2 * i_coax + 2)
    ax2_2.grid()
    ax2_2.plot(
        phase_freqs[freq_mask][:-1] / units.MHz,
        group_delay[freq_mask[:-1]] / units.ns
    )
    ax2_2.set_title(coax_name)
    print('Coax {}: {:.2f}ns + {} * {} = {}'.format(coax_name, line_fit[0], jumper_cables[str(hardware_number)][str(coax_name)], jumper_cable_delay, line_fit[0] + jumper_cables[str(hardware_number)][str(coax_name)]*jumper_cable_delay))
    ax2_2.axhline(
        line_fit[0],
        color='r',
        linestyle=':'
    )
    ax2_2.set_xlabel('f [MHz]')
    ax2_2.set_ylabel(r'$\Delta t$ [ns]')

fig1.tight_layout()
fig1.savefig('coax_magnitudes.png')

fig2.tight_layout()
fig2.savefig('coax_phases.png')
