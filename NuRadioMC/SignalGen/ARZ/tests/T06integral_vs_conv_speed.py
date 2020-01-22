from NuRadioMC.SignalGen.ARZ import ARZ
from NuRadioMC.utilities import units
from matplotlib import pyplot as plt
import numpy as np
import timeit

arz = ARZ.ARZ()
shower_energy = 1e15 * units.eV
shower_type = "HAD"
theta = 56 * units.deg
R = 10 * units.km
N = 512
dt = 0.01 * units.ns
profile_depth, profile_ce = arz.get_shower_profile(shower_energy, shower_type, 0)


def callable():
    return ARZ.get_vector_potential_fast(shower_energy, theta, N, dt, profile_depth, profile_ce,
                                     shower_type="HAD", distance=R)


t_int = timeit.timeit(callable, number=10)


def callable2():
    return ARZ.get_vector_potential_convolution(shower_energy, theta, N, dt, profile_depth, profile_ce,
                                     shower_type="HAD", distance=R)


t_conv = timeit.timeit(callable2, number=10)

print(f"convolution is a factor of {t_int/t_conv:.0f} faster")
