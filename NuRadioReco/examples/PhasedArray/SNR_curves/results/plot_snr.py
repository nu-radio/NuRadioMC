import matplotlib.pyplot as plt
import numpy as np
import json

angles = np.linspace(-10.,10.,41)

angle_labels = ['{:.1f} degrees'.format(angle) for angle in angles]
chosen_angles = angle_labels[6:-6:4]
#chosen_angles = [angle_labels[18], angle_labels[33]]
#chosen_angles = [angle_labels[0], angle_labels[20], angle_labels[-1]]

filenames = ["snr_{:.1f}deg.json".format(-angle+35.) for angle in angles]

dict_list = []

for filename in filenames:
    with open(filename) as input:
        snr = json.load(input)
        dict_list.append(snr)

snr_halves = []

for angle_label, snr in zip(angle_labels, dict_list):

    SNRs = np.array( snr['SNRs'] )
    triggered = np.array(snr['triggered'])
    total_evts = float(snr['total_events'])
    efficiency = triggered/total_evts

    snr_half = SNRs[ efficiency > 0.5 ][0]
    snr_halves.append(snr_half)

    if angle_label in chosen_angles:
        plt.plot(SNRs, efficiency, label = angle_label)

plt.xlabel('SNR')
plt.xlim((0.5,5))
plt.ylim((0,1.01))
plt.legend()
plt.ylabel('Efficiency')
plt.savefig('snr_curves_more.pdf', format='pdf')
plt.show()

plt.plot(angles, snr_halves)
plt.xlabel(r'Vertex elevation angle [deg]')
plt.ylabel('SNR at 0.5 efficiency')
plt.ylim((1,3))
plt.xlim((-8,8))
plt.savefig('snr_half_eff.pdf', format='pdf')
plt.show()
