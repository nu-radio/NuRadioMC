from NuRadioMC.utilities.Veff import export, get_Veff, get_Veff_array
from NuRadioReco.utilities import units
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Check NuRadioMC output')
parser.add_argument('--folder', type=str, default='results',
                    help='path to NuRadioMC simulation output folder')
args = parser.parse_args()

folder = args.folder

data_Veff = get_Veff(folder, point_bins=False)
Veff_array, energies, zenith_bins, trigger_names, weights = get_Veff_array(data_Veff)

zenith_index = 0
trigger_index = 0

Veffs = Veff_array[:, zenith_index, trigger_index, 0]
unc_Veffs = Veff_array[:, zenith_index, trigger_index, 1]

plt.errorbar(energies/units.eV, Veffs/units.km3, unc_Veffs/units.km3, marker='o')
plt.ylabel(r'Effective volume [km$^{3}$]')
plt.xlabel('Neutrino energy [eV]')
plt.xscale('log')
plt.yscale('log')
plt.show()
