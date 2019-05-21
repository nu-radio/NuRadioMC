import numpy as np
from NuRadioMC.utilities import units
import os
from scipy import interpolate as intp
import glob
import pickle
import sys
from matplotlib import pyplot as plt

if __name__ == "__main__":
    library_path = sys.argv[1]
    with open(library_path, 'rb') as fin:
        lib = pickle.load(fin)
        
        f, ax = plt.subplots(1, len(lib), sharey=True)
        
        for iS, shower_type in enumerate(lib):
            b = []
            nb = []
            for E in lib[shower_type]:
                nE = len(lib[shower_type][E]['charge_excess'])
                b.append(E)
                nb.append(nE)

            ax[iS].bar(np.log10(b), nE, linewidth=1, edgecolor='k', width=0.05)
            ax[iS].set_xlabel("log10(shower energy [eV])")
            ax[iS].set_title(shower_type)
        ax[0].set_ylabel('number of showers')
        f.tight_layout()
        plt.show()
                
        