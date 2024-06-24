import numpy as np
from NuRadioReco.utilities import units
import os
from scipy import interpolate as intp
import glob
import pickle
import sys
import corsika

rho = 0.924 * units.g / units.cm**3  # density g cm^-3

# process Corsika8 simulations of showers in ice

if __name__ == "__main__":
    print("usage: python A01preprocess_shower_library.py /path/to/library/ outputfilename")
    path = sys.argv[1]
    out = sys.argv[2]

    lgEE = np.arange(15, 18.001, 0.1)

    library = {}
    for subfolder in ["HAD", "EM"]:
        if(subfolder not in library):
            library[subfolder] = {}

        if subfolder == "HAD":
            folder = os.path.join(path, "nu")
        elif subfolder == "EM":
            folder = os.path.join(path, "e_minus")

        for iE, lgE in enumerate(lgEE):
            for file_e in sorted(glob.glob(os.path.join(folder, f"lgE_{lgE:.1f}/001*"))):
                try:
                    lib = corsika.Library(file_e)
                except:
                    print(f"can't open {file_e}")
                    continue
                primary = lib.get("primary")
                E = primary.summary['shower_0']["total_energy"] * units.GeV
                profile = lib.get("profile").astype("pandas")  # particle-number values
                n_shower = len(profile.shower.unique())
                for ishow in range(n_shower):
                    dat = profile.loc[profile["shower"] == ishow]
                    CE = dat["electron"]- dat["positron"]
                    # total_CE[i] = np.sum(CE)
                    if(E not in library[subfolder]):
                        library[subfolder][E] = {}
                        library[subfolder][E]['depth'] = dat['X'].to_numpy(dtype='d') * units.g / units.cm**2
                        library[subfolder][E]['charge_excess'] = []
                    library[subfolder][E]['charge_excess'].append(CE)

    with open(os.path.join(path, out), 'wb') as fout:
        pickle.dump(library, fout, protocol=4)
