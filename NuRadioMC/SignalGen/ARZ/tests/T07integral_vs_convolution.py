from NuRadioMC.SignalGen.ARZ import ARZ
from NuRadioMC.utilities import units
from matplotlib import pyplot as plt
from radiotools import coordinatesystems as cstrafo
import numpy as np

arz = ARZ.ARZ()
shower_energy = 1e15 * units.eV
shower_type = "HAD"
theta = 57 * units.deg
# thata = 0.8869725607593437 * units.rad
R = 1 * units.km
N = 512 * 2
dt = 0.01 * units.ns
profile_depth, profile_ce = arz.get_shower_profile(shower_energy, shower_type, 0)

A_conv = ARZ.get_vector_potential_convolution(shower_energy, theta, N, dt, profile_depth, profile_ce,
                                     shower_type="HAD", distance=R)
A_conv_far = ARZ.get_vector_potential_convolution_farfield(shower_energy, theta, N, dt, profile_depth, profile_ce,
                                     shower_type="HAD", distance=R)
A_int = ARZ.get_vector_potential_fast(shower_energy, theta, N, dt, profile_depth, profile_ce,
                                     shower_type="HAD", distance=R)

# use viewing angle relative to shower maximum for rotation into spherical coordinate system (that reduced eR component)
cs = cstrafo.cstrafo(zenith=theta, azimuth=0)
A_onsky = cs.transform_from_ground_to_onsky(A_int.T)
A_conv_onsky = cs.transform_from_ground_to_onsky(A_conv)

fig, ax = plt.subplots(1, 1)
tt = np.arange(0, (N + 1) * dt, dt)
# ax.plot(tt, -A_conv[0], label="convolution")
# ax.plot(tt, -A_conv[2], "--", label="convolution")
ax.plot(tt, -A_conv_far[1], "-", label="convolution far field")
ax.plot(tt, -A_conv_onsky[1], "--", label="convolution")

ax.plot(tt, A_onsky[1], ':', label="integral")
# ax.plot(tt, A_onsky[0], ':', label="integral")
# ax.plot(tt, A_int[:, 0], ':', label="integral")
# ax.plot(tt, A_int[:, 2], ':', label="integral")
ax.legend()
fig.tight_layout()
plt.show()
plt.savefig(f"plots/int_conv_{theta/units.deg:.1f}deg.png")

trace_conv = -np.diff(A_conv, axis=1) / dt
# trace_onsky = -np.diff(A_onsky) / dt
trace_int = -np.diff(A_int, axis=0) / dt

fig, ax = plt.subplots(1, 1)
tt = np.arange(0, (N) * dt, dt)
ax.plot(tt, -trace_conv[0], label="convolution")
ax.plot(tt, -trace_conv[2], "--", label="convolution")
# ax.plot(tt, trace_onsky[1], '--', label="integral")
ax.plot(tt, trace_int[:, 0], ':', label="integral")
ax.plot(tt, trace_int[:, 2], ':', label="integral")
ax.legend()
fig.tight_layout()

plt.show()
