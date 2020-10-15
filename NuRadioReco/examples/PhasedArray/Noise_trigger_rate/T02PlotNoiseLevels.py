import numpy as np
import scipy
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from radiotools import plthelpers as php
from NuRadioReco.utilities import units

single_rates = [100 * units.Hz, 10 * units.Hz, 1 * units.Hz]

max_entry = -1

pattern = f"pa_trigger_rate_1channels_1xupsampiling"

title = pattern
thresholds, n_triggers, ts, rates = np.loadtxt(str(pattern) + ".txt", unpack=True)

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

entries = np.arange(len(t_avg))
t_avg = np.array(t_avg)[entries > max_entry]
rate_avg_err = np.array(rate_avg_err)[entries > max_entry]
rate_avg = np.array(rate_avg)[entries > max_entry]

t_avg = np.array(t_avg)[rate_avg < 1e6]
rate_avg_err = np.array(rate_avg_err)[rate_avg < 1e6]
rate_avg = np.array(rate_avg)[rate_avg < 1e6]

print("t_avg: ", t_avg)
print("rate_avg: ", rate_avg)

f_intp = interp1d(t_avg, rate_avg, fill_value='extrapolate')

f3 = np.poly1d(np.polyfit(t_avg, np.log10(rate_avg), deg=1, w=1.0 / np.log10(rate_avg_err) ** 2))
f4 = np.poly1d(np.polyfit(t_avg, np.log10(rate_avg), deg=2, w=1.0 / np.log10(rate_avg_err) ** 2))

def f1(x):
    return 10 ** f3(x)
def f2(x):
    return 10 ** f4(x)

xxx = np.linspace(thresholds[0], thresholds[-1] + 1, 1000)
yyy = f1(xxx)
yyy2 = f2(xxx)
fig, ax = plt.subplots(1, 1)
ax.errorbar(t_avg, rate_avg / units.Hz, fmt='o', yerr=rate_avg_err / units.Hz, markersize=4)
ax.plot(xxx, yyy / units.Hz, php.get_color_linestyle(0))
ax.plot(xxx, yyy2 / units.Hz, php.get_color_linestyle(1))

def obj(x, t):
    #return f1(x) - t
    return f2(x) - t
try:
    for rate in single_rates:
        t = scipy.optimize.brentq(obj, 2, 6, args=rate)
        print(f"  {rate/units.Hz:.0f} Hz -> {t:.2f}")
except:
    pass

ax.set_xlim(thresholds[0] - 0.1, thresholds[-1] + 0.5)
ax.semilogy(True)
plt.minorticks_on()
ax.grid(which='major')
ax.grid(which='minor', alpha=0.25)
ax.set_title(title)
ax.set_xlabel("Threshold / Vrms")
ax.set_ylabel("rate [Hz]")
ax.set_xlim(2.0, 4.0)
ax.set_ylim(1e0, 1e6)
fig.tight_layout()
plt.show()
