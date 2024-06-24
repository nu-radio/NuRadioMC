import numpy as np
from NuRadioReco.utilities import units
from NuRadioReco.utilities import io_utilities
import os
from scipy import interpolate as intp
import glob
import sys
from matplotlib import pyplot as plt
from radiotools import helper as hp
from radiotools import plthelpers as php

if __name__ == "__main__":
    library_path1 = sys.argv[1]
    library_path2 = sys.argv[2]
    lib1 = io_utilities.read_pickle(library_path1)
    lib2 = io_utilities.read_pickle(library_path2)

    lgEE = np.arange(15, 18.001, 0.1)
    # for subfolder in ["HAD", "EM"]:
    for subfolder in ["EM"]:
        for lgE in lgEE:
            E = 10**lgE * units.eV
            Xmax = [[], []]
            total_CE = [[], []]
            if E in lib1[subfolder] and E in lib2[subfolder]:
                CE1 = lib1[subfolder][E]['charge_excess']
                total_CE[0].extend(np.sum(CE1, axis=1))
                CE2 = lib2[subfolder][E]['charge_excess']
                total_CE[1].extend(np.sum(CE2, axis=1))
                Xmax[0].extend(lib1[subfolder][E]['depth'][np.argmax(CE1, axis=1)])
                Xmax[1].extend(lib2[subfolder][E]['depth'][np.argmax(CE2, axis=1)])
            Xmax[0] = np.array(Xmax[0])
            Xmax[1] = np.array(Xmax[1])
            total_CE[0] = np.array(total_CE[0])
            total_CE[1] = np.array(total_CE[1])

            fig, ax = php.get_histograms([Xmax[0]/units.g*units.cm**2, Xmax[1]/units.g*units.cm**2], bins=50, xlabels=["Xmax [g/cm^2]","Xmax [g/cm^2]"])
            fig.suptitle(f"Xmax comparison for {subfolder}, {lgE:.1f}")
            plt.show()
            a = 1/0

