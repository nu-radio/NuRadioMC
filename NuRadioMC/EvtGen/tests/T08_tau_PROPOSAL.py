from NuRadioMC.EvtGen.NuRadioProposal import ProposalFunctions
from NuRadioReco.utilities import units
import matplotlib.pyplot as plt
import numpy as np
import json

"""
This file calculates some percentiles for the tau length and energy distributions
given by PROPOSAL, plots them, and saves them to a json file.

Just run it by typing:
    python T08_tau_PROPOSAL.py
The number of simulated muons per energy can be changed with Ntries, and the number
of energies can be changed with Nenergies
"""

Ntries = 1000
Nenergies = 20

lepton_code = 15
filename = 'tau_proposal_decay.json'

energies = np.linspace(15,20,Nenergies)
energies = 10**energies * units.eV

dists_arr = []
energy_arr = []

proposal_functions = ProposalFunctions(config_file='InfIce')

for Elepton in energies:

    decay_prods = proposal_functions.get_decays([Elepton]*Ntries, [lepton_code]*Ntries)

    dist_hist = decay_prods[:,0]
    E_hist = decay_prods[:,1]

    # Filter energies below 0.1 PeV
    mask = E_hist < 0.1*units.PeV
    E_hist[mask] = 0.1*units.PeV

    dists_arr.append(dist_hist)
    energy_arr.append(E_hist)

    print("energy: ", Elepton/units.PeV)

dists_arr = np.array(dists_arr)
energies_arr = np.array(energy_arr)

print(dists_arr.shape)
print(energies_arr.shape)

dists_05 = np.quantile(dists_arr, 0.05, axis=1)
dists_10 = np.quantile(dists_arr, 0.1, axis=1)
dists_16 = np.quantile(dists_arr, 0.16, axis=1)
dists_mean = np.mean(dists_arr, axis=1)
dists_84 = np.quantile(dists_arr, 0.84, axis=1)
dists_90 = np.quantile(dists_arr, 0.9, axis=1)
dists_95 = np.quantile(dists_arr, 0.95, axis=1)

energies_05 = np.quantile(energies_arr, 0.05, axis=1)
energies_10 = np.quantile(energies_arr, 0.1, axis=1)
energies_16 = np.quantile(energies_arr, 0.16, axis=1)
energies_mean = np.mean(energies_arr, axis=1)
energies_84 = np.quantile(energies_arr, 0.84, axis=1)
energies_90 = np.quantile(energies_arr, 0.9, axis=1)
energies_95 = np.quantile(energies_arr, 0.95, axis=1)

plot = False
if plot:
    plt.tight_layout()

    plt.gcf().subplots_adjust(bottom=0.13)
    plt.tick_params(labelsize=12)

    plt.loglog(energies, energies_mean, linestyle='-', color='red', label='Mean PROPOSAL with all losses')
    plt.loglog(energies, energies, linestyle='--', color='red', label='Mean, without losses')

    plt.fill_between(energies, energies_10, energies_90, facecolor='0.75', interpolate=True, label=r'10% to 90% quantiles')

    plt.xlabel('Tau initial energy [eV]', size=16)
    plt.ylabel('Tau decay energy [eV]', size=16)
    plt.legend(fontsize=12)
    plt.savefig('tau_decay_energy_PROPOSAL.png', format='png')
    plt.show()

output = {}
output['energies'] = list(energies)
output['lengths_05'] = list(dists_05)
output['lengths_10'] = list(dists_10)
output['lengths_16'] = list(dists_16)
output['lengths_84'] = list(dists_84)
output['lengths_90'] = list(dists_90)
output['lengths_95'] = list(dists_95)
output['lengths_mean'] = list(dists_mean)
output['tau_energies_05'] = list(energies_05)
output['tau_energies_10'] = list(energies_10)
output['tau_energies_16'] = list(energies_16)
output['tau_energies_84'] = list(energies_84)
output['tau_energies_90'] = list(energies_90)
output['tau_energies_95'] = list(energies_95)
output['tau_energies_mean'] = list(energies_mean)

with open(filename, 'w+') as data_file:
    json.dump(output, data_file, sort_keys=True, indent=4)
