import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import h5py
import glob
from scipy import interpolate
import json
import os
import sys
import csv

# Setup logging
from NuRadioReco.utilities.logging import setup_logger
logger = setup_logger(name="")

from NuRadioReco.utilities import units
from NuRadioReco.detector import detector
from NuRadioMC.utilities import fluxes
from NuRadioMC.utilities.Veff import get_Veff_Aeff, get_Veff_Aeff_array, get_index, get_Veff_water_equivalent
from NuRadioMC.examples.Sensitivities import E2_fluxes3 as limits
plt.switch_backend('agg')

if __name__ == "__main__":

    path = 'output'
    if(len(sys.argv) == 1):
        print("no path specified, assuming that hdf5 files are in directory 'output'")
    else:
        path = sys.argv[1]

    data = get_Veff_Aeff(path)
    Veffs, energies, energies_low, energies_up, zenith_bins, utrigger_names = get_Veff_Aeff_array(data)
    # calculate the average over all zenith angle bins (in this case only one bin that contains the full sky)
    Veff = np.average(Veffs[:, :, get_index("all_triggers", utrigger_names), 0], axis=1)
    # we also want the water equivalent effective volume times 4pi
    Veff = get_Veff_water_equivalent(Veff) * 4 * np.pi
    # calculate the uncertainty for the average over all zenith angle bins. The error relative error is just 1./sqrt(N)
    Veff_error = Veff / np.sum(Veffs[:, :, get_index("all_triggers", utrigger_names), 2], axis=1) ** 0.5
    # plot effective volume
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    print(energies / units.eV, Veff / units.km ** 3 / units.sr, Veff_error / units.km ** 3 / units.sr)
    ax.semilogx(True)
    ax.semilogy(True)
    
    plt.legend()
    ax.set_xlabel("neutrino energy [eV]")
    ax.set_ylabel("effective volume [km$^3$ sr]")
    #ax.set_ylabel("effective volume ratio")
    fig.tight_layout()
    fig.savefig("Veff_comparison_trig.pdf")

    # plot expected limit
    fig, ax = limits.get_E2_limit_figure(diffuse=True, show_grand_10k=True, show_grand_200k=False)
    labels = []
    labels = limits.add_limit(ax, labels, energies, Veff,
                              1, 'NuRadioMC example', livetime=3 * units.year, linestyle='-', color='blue', linewidth=3)
    leg = plt.legend(handles=labels, loc=2)
    fig.savefig("limits_stddev_comp_sigma_ratio.pdf")
    plt.show()
