import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.optimize import curve_fit


angles = np.linspace(-60., 30., 61)
angle_labels = ['{:.1f}degrees'.format(angle) for angle in angles]
      
def fit_func(x,x0,k):
    return 1/(1+np.exp(-k*(x-x0)))
# chosen_angles = [angle_labels[18], angle_labels[33]]
# chosen_angles = [angle_labels[0], angle_labels[20], angle_labels[-1]]


filenames = ["output/vertical/{:.1f}.json".format(angle + 66.6) for angle in angles]
print(angles)
#filenames = ["data/1.75bettersig/{:.1f}.json".format(angle + 39.6) for angle in angles]


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

plt.figure()
beams=np.arcsin(np.linspace(np.sin(-60*np.pi/180),np.sin(60*np.pi/180),12))*180/np.pi
plt.plot(angles, snr_halves)
plt.xlabel(r'Elev. Angle [deg]')
plt.ylabel('SNR at 0.5 efficiency')
plt.ylim((0, 5))
plt.xlim([-60,30])
for i in range(12):
    plt.axvline(beams[i],linestyle="dashed",alpha=.3)
plt.savefig('vert_snr_half_eff.png')
plt.show()
