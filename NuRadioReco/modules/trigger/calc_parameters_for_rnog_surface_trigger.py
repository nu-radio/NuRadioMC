import glob, os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def linear_fit(x, a, b):
    return a * x + b

#measurements are in mV
for file in (glob.glob("/Users/lilly/Software/diode/V_bias_measurements/*")):
    filename = (os.path.split(file)[-1])
    data = np.genfromtxt(file, delimiter=',', skip_header=1)
    xdata = (data[:,0]*1e-3)**2
    ydata = np.abs(data[:,1]*1e-3)
    plt.plot(xdata, ydata, marker='x',label=f'{filename}')
    popt, pcov = curve_fit(linear_fit, xdata, ydata, bounds=(-200,[100,0]))
    print(popt)
    plt.plot(xdata, linear_fit(xdata, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
    plt.legend()
    plt.show()
    #plt.savefig(f'fit_{filename}.png')
