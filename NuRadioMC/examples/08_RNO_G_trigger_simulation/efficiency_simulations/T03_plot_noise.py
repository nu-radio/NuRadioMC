import numpy as np
import scipy
import copy
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from radiotools import plthelpers as php
from NuRadioReco.utilities import units
import argparse

single_rates = [3000*units.Hz,2000*units.Hz,1000*units.Hz,100.0 * units.Hz, 10.0 * units.Hz, 1.0 * units.Hz]

max_entry = 50

parser = argparse.ArgumentParser(description='plot noise trigger rates')
parser.add_argument('--station_id', type=int, help='station_id', default=0)
parser.add_argument('--trigger', type=str, help='trigger type', default="power")
parser.add_argument('--upsampling_factor', type=int, help='upsampling factor', default=4)
parser.add_argument('--window', type=int, help='power integration window in units of unsamping_factor*1 samples', default=24)
parser.add_argument('--step', type=int, help='power trigger step in units of upsampling_factor*1 samples', default=4)
args = parser.parse_args()

if args.trigger == "power":
    pattern = f"data/noise/station{args.station_id}_{args.trigger}_{args.upsampling_factor}x_{args.window}win_{args.step}step"
    print(pattern)

elif args.trigger == "envelope":
    pattern = f"data/noise/station{args.station_id}_{args.trigger}_{args.upsampling_factor}x"
    print(pattern)

elif args.trigger == "highlow":
    pattern = f"data/noise/station{args.station_id}_{args.trigger}"
    print(pattern)


title = pattern
thresholds, n_triggers, ts, rates = np.loadtxt(str(pattern) + ".txt", unpack=True)

thresholds = np.round(thresholds, 5)

t_avg = []
rate_avg_err = []
rate_avg = []
for iThres, thres in enumerate(thresholds):
    mask = thresholds == thres
    t = np.sum(ts[mask])
    n_trig = np.sum(n_triggers[mask])
    rate = n_trig / (t)
    rate_error = n_trig ** 0.5 / (t)

    t_avg.append(thres)
    rate_avg_err.append(rate_error)
    rate_avg.append(rate)

t_avg = np.array(t_avg)
rate_avg = np.array(rate_avg)
rate_avg_err = np.array(rate_avg_err)

t_avg_og = copy.deepcopy(np.array(t_avg))
rate_avg_err_og = copy.deepcopy(np.array(rate_avg_err))
rate_avg_og = copy.deepcopy(np.array(rate_avg))

entries = np.arange(len(t_avg))

# Masks entries which plateau in trigger rate. 100's kHz (these might be flipped)
t_avg = np.array(t_avg)[rate_avg < 4e-4]
rate_avg_err = np.array(rate_avg_err)[rate_avg < 4e-4]
rate_avg = np.array(rate_avg)[rate_avg < 4e-4]

# Masks entries which don't trigger at all. (these might be flipped)
t_avg = np.array(t_avg)[rate_avg > 1e-7]
rate_avg_err = np.array(rate_avg_err)[rate_avg >1e-7]
rate_avg = np.array(rate_avg)[rate_avg >1e-7]

print("t_avg: ", t_avg)
print("rate_avg: ", rate_avg)

f_intp = interp1d(t_avg, rate_avg, fill_value='extrapolate')

# Linear exponential fit or quadratic exponential fit.
f3 = np.poly1d(np.polyfit(t_avg, np.log10(rate_avg), deg=1, w=1.0 / np.log10(rate_avg_err) ** 2))
f4 = np.poly1d(np.polyfit(t_avg, np.log10(rate_avg), deg=2, w=1.0 / np.log10(rate_avg_err) ** 2))
print(f4)

def f1(x):
    return 10 ** f3(x)

def f2(x):
    return 10 ** f4(x)

xxx = np.linspace(0.0, 50.0, 1000)
yyy = f1(xxx)
yyy2 = f2(xxx)
fig, ax = plt.subplots(1, 1)
ax.errorbar(t_avg_og, rate_avg_og / units.Hz, fmt='o', yerr=rate_avg_err_og / units.Hz, markersize=4)
#ax.plot(xxx, yyy / units.Hz, php.get_color_linestyle(0))
ax.plot(xxx, yyy2 / units.Hz, php.get_color_linestyle(1))


def obj(x, t):
    # return f1(x) - t
    return f2(x) - t

try:
    for rate in single_rates:
        t = scipy.optimize.brentq(obj, 2.0, 20.0, args=rate)
        print(f"  {rate/units.Hz:.0f} Hz -> {t:.2f}")
except:
    print('increase optimizer range')

#I should save a dict here so the sim scripts can automatically access them.

ax.semilogy(True)
plt.minorticks_on()
ax.grid(which='major')
ax.grid(which='minor', alpha=0.25)
#ax.set_title(title+'\n'+str(f4))
ax.set_xlabel("Threshold / Vrms")
ax.set_ylabel("rate [Hz]")
ax.set_xlim(0.0, 15)
ax.set_ylim(1e0, 1e7)
fig.tight_layout()

plt.savefig(title+'.png')
plt.show()
