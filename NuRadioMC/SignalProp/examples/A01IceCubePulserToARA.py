import matplotlib.pyplot as plt
import numpy as np
import time
from NuRadioMC.SignalProp import analyticraytracing as ray
from NuRadioMC.utilities import medium
from NuRadioReco.utilities import units
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('raytracing')

x1 = [-2 * units.km, -1500. * units.m]  # pulser position
x2 = [0., -200. * units.m]  # ARA antanna

r = ray.ray_tracing_2D(medium.southpole_simple())
solution = r.find_solutions(x1, x2)
fig, ax = plt.subplots(1, 1)
yyy, zzz = r.get_path(x1, x2, solution[0]['C0'])
ax.plot(yyy / units.m, zzz / units.m)
yyy, zzz = r.get_path(x1, x2, solution[1]['C0'])
ax.plot(yyy / units.m, zzz / units.m)
ax.set_ylim(-1600, 0)
ax.set_xlabel('x [m]')
ax.set_ylabel('z [m]')
fig.tight_layout()
fig.savefig("IceCubePulserToARA1.png")

x1 = [-4 * units.km, -1500. * units.m]  # pulser position
x2 = [0., -200. * units.m]  # ARA antanna

r = ray.ray_tracing_2D(medium.southpole_simple())
solution = r.find_solutions(x1, x2)
fig, ax = plt.subplots(1, 1)
yyy, zzz = r.get_path(x1, x2, solution[0]['C0'])
ax.plot(yyy / units.m, zzz / units.m)
yyy, zzz = r.get_path(x1, x2, solution[1]['C0'])
ax.plot(yyy / units.m, zzz / units.m)
ax.set_ylim(-1600, 0)
ax.set_xlabel('x [m]')
ax.set_ylabel('z [m]')
fig.tight_layout()
fig.savefig("IceCubePulserToARA2.png")
plt.show()
