import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.optimize import curve_fit
import argparse

parser = argparse.ArgumentParser(description='Run NuRadioMC simulation')
parser.add_argument('station_id', type=int, help='station_id', default=0)

args = parser.parse_args()

angles = np.linspace(-10., 10., 41)
angle_labels = ['{:.1f}degrees'.format(angle) for angle in angles]
chosen_angles=angle_labels[32:8:-2] 
         
def fit_func(x,x0,k):
    return 1/(1+np.exp(-k*(x-x0)))

filenames = [f"data/output/station{args.station_id}/boresight/{angle + 36.6:.1f}.json" for angle in angles]

print(filenames)
dict_list = []

for filename in filenames:
    with open(filename) as input:
        snr = json.load(input)
        dict_list.append(snr)

snr_halves = []


for angle_label, snr in zip(angle_labels, dict_list):

    SNRs = np.array(snr['SNRs'])
    triggered = np.array(snr['triggered'])
    total_evts = float(snr['total_events'])
    efficiency = triggered / total_evts

    snr_half = np.interp(0.5,efficiency,SNRs[:])

    snr_halves.append(snr_half)

    if angle_label in chosen_angles:
        plt.plot(SNRs, efficiency, label=angle_label)

        test_snr=np.linspace(0,12,500)
        fit=curve_fit(fit_func,SNRs,efficiency,p0=[4,10])
        eff_fit=fit_func(test_snr,fit[0][0],fit[0][1])

        print(angle_label,snr_half)


print(np.mean(snr_halves))
plt.title('Trig Eff. to Nu')
plt.xlabel('SNR')
plt.xlim((0.5, 8))
plt.ylim((0, 1.01))
plt.legend()
plt.ylabel('Efficiency')
plt.savefig('snr_curves.png')
plt.show()

plt.plot(angles, snr_halves)
plt.xlabel(r'View Angle [deg]')
plt.ylabel('SNR at 0.5 efficiency')
plt.ylim((1, 5))
plt.xlim((-8, 8))
plt.savefig('snr_half_eff.png')
plt.show()
