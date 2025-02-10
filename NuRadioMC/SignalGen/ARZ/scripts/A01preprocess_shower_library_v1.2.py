import numpy as np
from NuRadioReco.utilities import units
import os
from scipy import interpolate as intp
import glob
import pickle
import sys

rho = 0.924 * units.g / units.cm**3  # density g cm^-3

if __name__ == "__main__":
    print("usage: python A01preprocess_shower_library.py /path/to/library/ outputfilename")
    path = sys.argv[1]
    out = sys.argv[2]

    library = {}
    for subfolder in ["HAD", "EM"]:
        if(subfolder not in library):
            library[subfolder] = {}

        for file_e in sorted(glob.glob(os.path.join(path, subfolder, "*/*.t1005"))):
            filename = os.path.splitext(os.path.basename(file_e))[0]
            file_p = file_e[:-1] + "6"
            estr = filename.split("eV")[0][2:]
            E = 10**float(estr) * units.eV

            depth_e, N_e = np.loadtxt(file_e, unpack=True, usecols=(1, 2))
            depth_p, N_p = np.loadtxt(file_p, unpack=True, usecols=(1, 2))
            depth_e *= units.g / units.cm**2
            depth_p *= units.g / units.cm**2
            depth_e -= 1000 * units.g/units.cm**2  # all simulations have an artificial offset of 1000 g/cm^2
            depth_p -= 1000 * units.g/units.cm**2
            ce = N_e - N_p
            # sanity check if files electron and positron profiles are compatible
            if (not np.all(depth_e == depth_p)):
                raise ImportError("electron and positron profile have different depths")

            if(E not in library[subfolder]):
                library[subfolder][E] = {}
                library[subfolder][E]['depth'] = depth_e
                library[subfolder][E]['charge_excess'] = []

            library[subfolder][E]['charge_excess'].append(ce)


            if 0:
                from matplotlib import pyplot as plt
                fig, ax = plt.subplots(1, 1)
                ax.plot(depth_e / units.g * units.cm**2, ce)
                ax.set_title("{} E = {:.2g}eV".format(subfolder, E))
                ax.semilogy(True)
                ax.set_ylabel("charge excess")
                ax.set_ylim(10)
                ax.set_xlabel(r"shower depth [g/cm$^2$]")
                fig.tight_layout()
#                 fig.savefig(os.path.join(path, "plots", subfolder, "{}.png".format(filename)))
                plt.show()
                plt.close(fig)
                continue

#             length = depth_e / rho
#             zmax = length.max()
#             xnep = intp.interp1d(length, N_e - N_p, bounds_error=False, fill_value=0)
    with open(os.path.join(path, out), 'wb') as fout:
        pickle.dump(library, fout, protocol=2)
