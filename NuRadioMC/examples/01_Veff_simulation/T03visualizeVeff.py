import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import h5py
import glob
from scipy import interpolate
import json
import os

from NuRadioReco.utilities import units
from NuRadioReco.detector import detector
from NuRadioMC.utilities import fluxes
from NuRadioMC.utilities.Veff import get_Veff
from NuRadioMC.examples.Sensitivities import E2_fluxes3 as limits



if __name__ == "__main__":

    try:
        os.path.isdir("output")
    except:
        print("Please move files to folder output. This is the default location.")

    energies, Veff, Veff_error, SNR, trigger_names, deposited  = get_Veff("output")


    # plot effective volume
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    ax.errorbar(energies / units.eV, Veff['all_triggers']/units.km**3 /units.sr,
                yerr=Veff_error['all_triggers']/units.km**3/units.sr, fmt='d-')
    ax.semilogx(True)
    ax.semilogy(True)
    ax.set_xlabel("neutrino energy [eV]")
    ax.set_ylabel("effective volume [km$^3$ sr]")
    fig.tight_layout()
    fig.savefig("Veff.pdf")



    # plot expected limit
    fig, ax = limits.get_E2_limit_figure(diffuse=True,show_grand_10k=True, show_grand_200k=False)
    labels = []
    labels = limits.add_limit(ax, labels, energies, Veff['all_triggers'].T,
                              100, 'NuRadioMC example', livetime=3*units.year, linestyle='-',color='blue',linewidth=3)

    plt.legend(handles=labels, loc=2)
    fig.savefig("limits.pdf")
    plt.show()