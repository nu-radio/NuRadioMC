import numpy as np
from utilities import units
from matplotlib import pyplot as plt
from ROOT import gROOT
gROOT.LoadMacro("RayTraceRK4.C+")
from ROOT import RayTraceRK4

ff = np.linspace(100 * units.MHz, 1 * units.GHz, 100)
x_start = [800, 0, -300]
x_antenna = [1000, 0, -200]
getres = RayTraceRK4(x_start[0], x_start[1], x_start[2],
                     x_antenna[0], x_antenna[1], x_antenna[2], 0,
                     ff, len(ff))

launch_angle = getres[0]
receive_angle = getres[1]
dt = getres[2]

attenuation = np.array([getres[i + 9] for i in range(len(ff))])

fix, ax = plt.subplots(1, 2)
ax[0].plot(x_start[0] / units.m, x_start[2] / units.m, 'ko', label='start')
ax[0].plot([x_start[0] / units.m, x_start[0] / units.m + 50 * np.cos(launch_angle)],
           [x_start[2] / units.m, x_start[2] / units.m + 50 * np.sin(launch_angle)], 'k-')
ax[0].plot(x_antenna[0] / units.m, x_antenna[2] / units.m, 'kD', label='antenna')
ax[0].plot([x_antenna[0] / units.m, x_antenna[0] / units.m + 50 * np.cos(receive_angle)],
           [x_antenna[2] / units.m, x_antenna[2] / units.m + 50 * np.sin(receive_angle)], 'k-')
ax[0].legend(numpoints=1)
ax[0].set_xlabel("x [m]")
ax[0].set_ylabel("z [m]")

ax[1].plot(ff / units.MHz, attenuation)
ax[1].set_xlabel("frequency [MHz]")
ax[1].set_ylabel("attenuation")
plt.tight_layout()
plt.show()
