from NuRadioReco.utilities import units
from NuRadioMC.SignalGen.askaryan import get_time_trace, get_frequency_spectrum
import matplotlib.pyplot as plt
import numpy as np

"""
This file shows how to use the electric field modules for standalone calculations.
First we start defining all the input parameters for the function get_time_trace
"""

energy = 0.1 * units.EeV # Shower energy
theta = 58 * units.deg # Viewing angle formed by the shower axis and the vertex-observer line
N = 2048 # Number of samples
dt = 0.1 * units.ns # Time binning
shower_type = 'HAD' # 'HAD' or 'EM'
R = 1 * units.km # Distance to shower vertex
n_index = 1.78 # refractive index

"""
As an example, we will choose the models Alvarez2009, a frequency-domain parameterisation
that is recommended for effective volume calculations because of its speed, and
ARZ2020, a semi-analytical model recommended for reconstructions, since it is the
most accurate model in NuRadioMC
"""
models = ['Alvarez2009', 'ARZ2020']

"""
The times are not returned by the signal generator module, so we have to construct
our own time array using the input time step. We have chosen the middle of the trace
to be at t = 0. When using the ARZ2019 or ARZ2020 modules, the t=0 in our times
array defined like this marks the moment when the signal from the vertex arrives
at the observer's location.
"""
times = np.arange(0, N * dt, dt) # Array containing times
times = times + 0.5 * dt - times.mean()

"""
We are now ready to plot a trace from both models.
"""
for model in models:

    trace = get_time_trace(energy, theta, N, dt, shower_type, n_index, R, model)
    plt.plot(times, trace/(units.microvolt/units.m), label=model)

plt.legend()
plt.xlim((-5, 5))
plt.xlabel('Time [ns]')
plt.ylabel(r'Electric field [$\mu$V/m]')
plt.show()

"""
One important thing to keep in mind is that the Alvarez2009 model varies the
length of the electromagnetic shower randomly, and also the ARZ models chooses
a random shower from the library to account for random fluctuations. This means
that each time we call the functions, the showers are different and so the
electric fields are also different. This is not practical when we want to
calculate the field from a fixed shower at different locations. In order to
do so, we can use the argument same_shower=True.

Let us plot the field from a 0.5 EeV electromagnetic shower, subject to the LPM
effect, calculated with the ARZ2020 model, for three different distances.
The use of same_shower=True guarantees that the shower used for the three
positions is the same. The electric field might look shaky and present different
peaks, due to the LPM stretching the shower.
"""
distances = np.array([500, 750, 1000]) * units.m
energy = 0.5 * units.EeV
shower_type = 'EM'

for R in distances:

    trace = get_time_trace(energy, theta, N, dt, shower_type, n_index, R, model,
                           same_shower=True)
    plt.plot(times, trace/(units.microvolt/units.m), label='{:.0f} m'.format(R/units.m))

plt.legend()
plt.xlim((0, 20))
plt.xlabel('Time [ns]')
plt.ylabel(r'Electric field [$\mu$V m$^{-1}$]')
plt.show()

"""
Using the function get_frequency_spectrum, we can get the spectrum for the same
setup, but using hadronic showers, for instance. The FFT wrapper in NuRadioReco
ensures that our spectrum will be expressed in units of voltage per length per
frequency, and not in adimensional units as it is customary for FFTs.
"""
shower_type = 'HAD'

for R in distances:

    spectrum = get_frequency_spectrum(energy, theta, N, dt, shower_type, n_index, R, model,
                                      same_shower=True)
    spectrum = np.abs(spectrum)
    max_freq = 0.5 / dt
    frequencies = np.linspace(0, max_freq, len(spectrum))
    plt.plot(frequencies/units.MHz, spectrum/(units.microvolt/units.m/units.MHz), label='{:.0f} m'.format(R/units.m))

plt.legend()
plt.xlim((0, 1500))
plt.ylim((0, None))
plt.xlabel('Frequency [MHz]')
plt.ylabel(r'Electric field transform [$\mu$V m$^{-1}$ MHz$^{-1}$]')
plt.show()

"""
If the user wants to use the ARZ model and specify the distance and viewing angle with
respect to the shower, they will have to import the ARZ class from the SignalGen.ARZ.ARZ.py
module and use the get_time_trace function with the argument:

shift_for_xmax=True

This function will return an array with three elements: e_R, e_theta, and e_phi, each
one of them is an array containing the radial, zenithal, and azimuthal component of
the electric field respectively, as seen from the observer position. The radial
direction is the line that joins the observer and the shower maximum.

Let us write a brief example.
"""

from NuRadioMC.SignalGen.ARZ import ARZ

ARZ_object = ARZ.ARZ()

R_from_xmax = 1 * units.km
theta_from_xmax = 58 * units.deg

trace = ARZ_object.get_time_trace(energy, theta_from_xmax, N, dt, shower_type, n_index, R_from_xmax,
                                  shift_for_xmax=True)
