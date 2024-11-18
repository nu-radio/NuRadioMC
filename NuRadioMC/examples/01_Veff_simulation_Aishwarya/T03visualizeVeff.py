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
    
    ax.errorbar([1.000000000000000000e+16, 4.999999999999999200e+16, 1.000000000000000000e+17, 4.999999999999999360e+17, 1.000000000000000000e+18, 4.999999999999998976e+18, 1.000000000000000000e+19, 4.999999999999999181e+19, 1.000000000000000000e+20, 4.999999999999999345e+20, 1.000000000000000000e+21], [1.562023559628805810e-03, 5.870107281413138034e-02, 2.106248051330824322e-01, 2.226723587838828511e+00, 4.692367957866072636e+00, 1.871731843723915745e+01, 2.998539429730962524e+01, 7.399797546531475234e+01, 1.003989429158390152e+02, 1.702326206060227776e+02, 1.977777875450684917e+02], fmt='d-', label = "Hi Lo Ryan")

    ax.errorbar([1.000000000000000000e+16, 4.999999999999999200e+16, 1.000000000000000000e+17, 4.999999999999999360e+17, 1.000000000000000000e+18, 4.999999999999998976e+18, 1.000000000000000000e+19, 4.999999999999999181e+19, 1.000000000000000000e+20, 4.999999999999999345e+20, 1.000000000000000000e+21], [3.291408518238634739e-03, 1.085753217603100396e-01, 3.700305986024803317e-01, 3.251741895184204090e+00, 6.643688741417114407e+00, 2.481407055394997130e+01, 3.901541943717347749e+01, 9.095621246710383900e+01, 1.212348635693149106e+02, 1.944504530396249891e+02, 2.220904414963438285e+02], fmt='d-', label = "PA_fir_4_32_fake")

    ax.errorbar([10**16.5, 10**17, 10**17.5, 10**18, 10**18.5, 10**19, 10**19.5, 10**20], [0.0001955,0.002674, 0.0199, 0.1455, 0.6517, 2.207, 6.303, 14.07], fmt='d-', label = "Aishwarya Hi lo")
    
    ax.errorbar([10**18, 10**19, 10**20], [0.0732518, 1.58077362, 6.12700164], fmt = 'd-', label = "PA New")
    """ 
    ax.errorbar(energies / units.eV, Veff / units.km ** 3 / units.sr,
                yerr=Veff_error / units.km ** 3 / units.sr, fmt='d-', label = "Hi Lo 100 PeV")
    
    ax.errorbar([1.e+18, 1.e+19, 1.e+20], [1.19091152, 7.5608392, 20.97377673], yerr=[0.12062607, 0.96598154, 1.60080984], fmt='d-', label = "+1 std", color = "red")
    ax.errorbar([1.e+18, 1.e+19, 1.e+20], [1.57091273, 8.78974977, 24.63608811], yerr=[0.13854072, 1.03630998, 1.73495134], fmt='d-', label = "-1 std", color = "red")
    ax.errorbar([1.e+18, 1.e+19, 1.e+20], [1.093249, 7.34827452, 20.06396402],yerr=[0.11734784, 0.94753249, 1.56570448], fmt='d-', label="+2 std", color = "blue")
    ax.errorbar([1.e+18, 1.e+19, 1.e+20], [2.52868721, 10.74183155, 28.40794311], yerr=[0.17577171, 1.14562002, 1.86303631], fmt='d-', label="-2 std", color = "blue")
    """
    print(energies / units.eV, Veff / units.km ** 3 / units.sr, Veff_error / units.km ** 3 / units.sr)
    ax.semilogx(True)
    ax.semilogy(True)
    
    """ 
    path = "/data/i3store/users/avijai/Design_eff_vol.csv"
    count = 0
    martin_energy = []
    martin_effVol = []
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            if (count > 1):
                martin_energy.append(float(row[0]))
                martin_effVol.append(float(row[1])/35 * 4 * np.pi)
            count += 1
    ax.scatter(martin_energy, martin_effVol, label = "design_paper")
    ax.plot(martin_energy, martin_effVol)
    plt.legend()
    
    """
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
