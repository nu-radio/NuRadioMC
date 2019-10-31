from NuRadioReco.utilities import units
import matplotlib.pyplot as plt
import numpy as np
import json

"""
This file plots the tau energy and length profiles percentiles from the
file calculated with PROPOSAL using T08_tau_PROPOSAL.py and also the profiles
from the CSDA calculated using T03_tau_length.py. It also performs a fit to the
tau length mean.

Just run it by typing:
    python T09_plot_PROPOSAL_diff.py
"""

def get_dict(filename):

    with open(filename, 'r') as datafile:
        input = json.load(datafile)

    for key in input:
        input[key] = np.array(input[key])

    return input

file_cont = 'continuous_decay.json'
file_tau = 'tau_proposal_decay.json'
file_taubar = 'taubar_proposal_decay.json'

input_cont = get_dict(file_cont)
input_tau = get_dict(file_tau)
#input_taubar = get_dict(file_taubar)

# PLOTTING ENERGIES

plt.tight_layout()

plt.gcf().subplots_adjust(bottom=0.13)
plt.tick_params(labelsize=12)

plt.loglog(input_cont['energies'], input_cont['tau_energies_mean'], linestyle='-', color='black', label='Mean with cont. PN losses')
plt.loglog(input_tau['energies'], input_tau['tau_energies_mean'], linestyle='-', color='red', label='Mean PROPOSAL with all losses')
plt.loglog(input_cont['energies'], input_cont['energies'], linestyle='-.', color='red', label='Mean, without losses')

plt.fill_between(input_cont['energies'], input_cont['tau_energies_10'], input_cont['tau_energies_90'],
                facecolor='rosybrown', alpha=0.75, interpolate=True, label=r'10% to 90% quantiles, PN')
plt.fill_between(input_tau['energies'], input_tau['tau_energies_10'], input_tau['tau_energies_90'],
                facecolor='skyblue', alpha=0.75, interpolate=True, label=r'10% to 90% quantiles, PROPOSAL')

plt.xlabel('Tau initial energy [eV]', size=16)
plt.ylabel('Tau decay energy [eV]', size=16)
plt.legend(fontsize=11,loc=2)
plt.ylim((1e14, None))
plt.savefig('tau_decay_energy_PROPOSAL_diff.png', format='png')
plt.show()

# PLOTTING LENGTHS

from scipy import constants
tau_rest_lifetime = 290.3 * units.fs
cspeed = constants.c * units.m / units.s
tau_mass = constants.physical_constants['tau mass energy equivalent in MeV'][0] * units.MeV
length_no_losses = input_cont['energies']*tau_rest_lifetime*cspeed / tau_mass

plt.tight_layout()

plt.gcf().subplots_adjust(bottom=0.13)
plt.tick_params(labelsize=12)

plt.loglog(input_cont['energies'], input_cont['lengths_mean']/units.km, linestyle='-', color='black', label='Mean with cont. PN losses')
plt.loglog(input_tau['energies'], input_tau['lengths_mean']/units.km, linestyle='-', color='red', label='Mean PROPOSAL with all losses')
plt.loglog(input_cont['energies'], length_no_losses/units.km, linestyle='-.', color='red', label='Mean, without losses')

plt.fill_between(input_cont['energies'], input_cont['lengths_10']/units.km, input_cont['lengths_90']/units.km,
                facecolor='rosybrown', alpha=0.75, interpolate=True, label=r'10% to 90% quantiles, PN')
plt.fill_between(input_tau['energies'], input_tau['lengths_10']/units.km, input_tau['lengths_90']/units.km,
                facecolor='skyblue', alpha=0.75, interpolate=True, label=r'10% to 90% quantiles, PROPOSAL')

plt.xlabel('Tau initial energy [eV]', size=16)
plt.ylabel('Tau track length [km]', size=16)
plt.legend(fontsize=12)
plt.savefig('tau_decay_length_PROPOSAL_diff.png', format='png')
plt.show()

# FITTING THE 95% PERCENTILE FOR TAU SIMULATIONS

from scipy.optimize import curve_fit

def poly(x, a, b, c, d, e):
    return a + b*x + c*x**2 + d*x**3 + e*x**4

xdata = np.log10(input_tau['energies'])
ydata = np.log10(input_tau['lengths_95'])

results = curve_fit(poly, xdata, ydata)[0]
print("Results for the polynomial fit", results)

plt.plot(input_tau['energies'], input_tau['lengths_95']/units.km, label='95% quantile by PROPOSAL')
plt.plot(input_tau['energies'], 10**poly(xdata, *results)/units.km, label='Polynomial fit')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Tau initial energy [eV]', size=16)
plt.ylabel('Tau track length [km]', size=16)
plt.legend()
plt.show()
