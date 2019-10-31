from NuRadioMC.SignalGen.ARZ import ARZ
from NuRadioMC.utilities import units
from matplotlib import pyplot as plt
from radiotools import coordinatesystems as cstrafo
import numpy as np
from NuRadioMC.utilities.fft import time2freq

arz = ARZ.ARZ()
shower_energy = 1e15 * units.eV
shower_type = "HAD"
theta = 55.5 * units.deg
# thata = 0.8869725607593437 * units.rad
R = 10 * units.km
N = 2048 * 4
dt = 0.01 * units.ns
profile_depth, profile_ce = arz.get_shower_profile(shower_energy, shower_type, 0)

Rs = np.array([10, 100, 1000, 10000]) * units.m
linestyles = {Rs[0]:':', Rs[1]:'-.', Rs[2]:'--', Rs[3]:'-'}

for R in Rs:

    A_conv = ARZ.get_vector_potential_convolution(shower_energy, theta, N, dt, profile_depth, profile_ce,
                                         shower_type="HAD", distance=R)
    A_conv_far = ARZ.get_vector_potential_convolution_farfield(shower_energy, theta, N, dt, profile_depth, profile_ce,
                                         shower_type="HAD", distance=R)
    A_int = ARZ.get_vector_potential_fast(shower_energy, theta, N, dt, profile_depth, profile_ce,
                                         shower_type="HAD", distance=R)

    A_int_mod = np.sqrt(A_int[:,0]**2+A_int[:,2]**2)

    tt = np.arange(0, (N + 1) * dt, dt)

    max_freq = 0.5/dt
    Nzeros = 10000
    A_conv_pad = np.concatenate((np.abs(A_conv_far[1]), np.zeros(Nzeros)))
    A_int_pad = np.concatenate((A_int_mod, np.zeros(Nzeros)))
    A_conv_freq = time2freq(A_conv_pad)
    A_int_freq = time2freq(A_int_pad)
    freqs = np.linspace(0, max_freq, len(A_int_freq))
    E_conv_freq = np.abs(freqs * A_conv_freq)
    E_int_freq = np.abs(freqs * A_int_freq)
    plt.loglog(freqs/units.MHz, E_conv_freq, color='blue',
               linestyle=linestyles[R], label = '{:.0f} m'.format(R))
    plt.loglog(freqs/units.MHz, E_int_freq, color='black',
               linestyle=linestyles[R])

plt.plot([],[], linestyle='', label='Black: integral')
plt.plot([],[], linestyle='', label='Blue: convolution (far field)')
plt.title(r'$\theta = ${:.1f}'.format(theta/units.deg))
plt.xlabel('Freqs [MHz]')
plt.ylabel('Electric field module [a.u.]')
plt.xlim((None,3000))
plt.ylim((1e-8,None))
plt.legend()
plt.savefig('Conv_vs_int_{:.1f}deg.pdf'.format(theta/units.deg), format='pdf')
plt.show()
