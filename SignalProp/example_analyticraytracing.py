import matplotlib.pyplot as plt
import time
from NuRadioMC.SignalProp import analyticraytraycing as ray
import logging
logging.basicConfig(level=logging.INFO)

x1 = [478., -149.]
x2 = [635., -5.]  # direct ray solution
x3 = [1000., -90.]  # refracted/reflected ray solution
x4 = [700., -149.]  # refracted/reflected ray solution
x5 = [1000., -5.]  # no solution

dt = 0

lss = ['-', '--', ':']
fig, ax = plt.subplots(1, 1)
ax.plot(x1[0], x1[1], 'ko')
for i, x in enumerate([x2, x3, x4, x5]):
    t = time.time()
    results = ray.find_solutions(x1, x, False)
    dt = time.time() - t
    for j, C_0 in enumerate(results):
        yy, zz = ray.get_path(x1, x, C_0)
        ax.plot(yy, zz, 'C{:d}{}'.format(i, lss[j % 3]), label='C0 = {:.2f}'.format(C_0))
        ax.plot(x[0], x[1], 'dC{:d}-'.format(i))
ax.legend()

print("time to calculate rays for 4 points: {:.1f} ms".format(1e3 * dt))
plt.show()

x1 = [0, -5]
x2 = [-1e3, -1e3]
x3 = [-2e3, -2e3]
lss = ['-', '--', ':']
fig, ax = plt.subplots(1, 1)
ax.plot(x1[0], x1[1], 'ko')
for i, x in enumerate([x2, x3]):
    results = ray.find_solutions(x, x1, False)
    for j, C_0 in enumerate(results):
        yy, zz = ray.get_path(x, x1, C_0)
        ax.plot(yy, zz, 'C{:d}{}'.format(i, lss[j % 3]), label='C0 = {:.2f}'.format(C_0))
        ax.plot(x[0], x[1], 'dC{:d}-'.format(i))
ax.legend()
plt.show()