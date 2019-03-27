import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from NuRadioReco.detector import detector
import h5py
import glob
from NuRadioReco.utilities import units
from radiotools import plthelpers as php
from scipy import interpolate
import json
from matplotlib import pyplot as plt
import os
from NuRadioMC.utilities import fluxes
from NuRadioMC.utilities.Veff import get_Veff
from NuRadioMC.examples.Sensitivities import E2_fluxes2 as limits




if __name__ == "__main__":
    
    energies, Veff, Veff_error, SNR, trigger_names, deposited  = get_Veff("/Users/cglaser/analysis/simulations/201902_ARZ_Veff/ARIA/Alvarez2000/")
    
    
    # plot effective volume
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    ax.errorbar(energies / units.eV, Veff['all_triggers']/units.km**3 /units.sr,
                yerr=Veff_error['all_triggers']/units.km**3/units.sr, fmt='d-')
    ax.semilogx(True)
    ax.semilogy(True)
    ax.set_xlabel("neutrino energy [eV]")
    ax.set_ylabel("effective volume [km$^3$ sr]")
#     ax.legend()
#     fig.savefig("plots/ARIA_Alv_ARZ.png")
    fig.tight_layout()
    fig.savefig("Veff.png")
    
    
    
    # plot expected limit
    fig, ax = limits.get_E2_limit_figure(show_grand_10k=True, show_grand_200k=False)
    labels = []
    labels = limits.add_limit(ax, labels, energies, Veff['all_triggers'],
                              100, 'NuRadioMC example', fmt='-', livetime=3*units.year, color="blue")
    plt.legend(handles=labels, loc=2)
    fig.savefig("limits.png")
    plt.show()