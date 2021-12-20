from NuRadioMC.utilities.Veff import export, get_Veff_Aeff, get_Veff_Aeff_array
from NuRadioReco.utilities import units
import matplotlib.pyplot as plt
import numpy as np
import argparse

"""
This file explains how to use the utilities.Veff module to calculate and plot
the effective volumes of a set of NuRadioMC simulations. To run it with the
default 'results' folder in this directory, just run:

python W04EffectiveVolumes.py

Otherwise, use:

python W04EffectiveVolumes.py --folder path/to/results_folder --output_file output_file.json
"""

parser = argparse.ArgumentParser(description='Check NuRadioMC output')
parser.add_argument('--folder', type=str, default='results',
                    help='path to NuRadioMC simulation output folder')
parser.add_argument('--output_file', type=str, default='results/Veffs.json',
                    help='path to NuRadioMC simulation output folder')
args = parser.parse_args()

if __name__ == "__main__":
    folder = args.folder
    output_file = args.output_file

    """
    The function get_Veff in utilities.Veff calculates effective volumes using the
    path to a folder with NuRadioMC outputs as an argument. In this folder there
    should be several files containing the simulation results for different energies,
    although in our example we will only use one. There can also be several sets
    of files for different zenith bands. get_Veff will return a dictionary with all
    the information to calculate effective volumes for different energies and
    zenith bands.

    IMPORTANT: if the n_events_per_file argument has been used so that the NuRadioMC
    files are split, the module utilities.merge_hdf5 should be used to merge the files.
    Once every energy bin has a unique HDF5 output file, the Veff module can be run.
    """
    data_Veff = get_Veff_Aeff(folder)

    """
    Although data_Veff has all the information we need, it is a bit cumbersome
    to read directly, so that's why we resort to the function get_Veff_array. This
    function returns a 5-element tuple with:
    - An 4-D array with the effective volumes
    - A 1-D array with the centre energies for each bin (if point_bins=False)
    - The zenith bins for each one of the zenith band simulations (in our case, we
    only have a single zenith band equal to the whole sky)
    - A list with the trigger names
    - A 1-D array with the angular weights for each zenith band simulation. If the solid
    angle for a simulation set is larger than for the other, it should carry more weight
    when we try to patch them all together to get the total volumes.
    """
    Veff_array, energies, energies_low, energies_high, zenith_bins, trigger_names = get_Veff_Aeff_array(data_Veff)

    """
    There are some functions with the same syntax to calculate effective areas for
    atmospheric muon simulations: get_Aeff and get_Aeff_array. Keep in mind that the
    latter returns a 4-element tuple, with no weights.

    The structure of the 4-D Veff_array returned is as follows: the first dimension
    chooses the energy, the second chooses the zenith band and the third chooses
    the trigger:

    Veff_item = Veff_array[i_energy, i_zenith, i_trigger]

    Then, each Veff_item has three numbers.
    - Veff_item[0] is the effective volume
    - Veff_item[1] is the poissonian uncertainty of the effective volume
    - Veff_item[2] is the sum of the weights of all the triggering events contained
    in a given energy and zenith band for the chosen trigger.
    - Veff_item[3] is the 68% confidence belt lower limit for the effective volume
    using the Feldman-Cousins method.
    - Veff_item[4] is the 68% confidence belt upper limit for the effective volume 
    using the Feldman-Cousins method.

    For our example, we only have a single file, so we choose the only zenith index.
    We choose as well the first trigger (index 0) just as an example.
    """
    zenith_index = 0
    trigger_index = 0

    # Selecting the effective volumes
    Veffs = Veff_array[:, zenith_index, trigger_index, 0]
    # Selecting the uncertainties
    unc_Veffs = Veff_array[:, zenith_index, trigger_index, 1]

    """
    We plot now the effective volumes in cubic kilometres as a function of the
    neutrino energy in electronvolts, with the uncertainty in cubic kilometres
    as an error bar. For our example, this is going to look just a sad, lonely point,
    but as an exercise you can create more simulations with different energy bins
    and plot the effective volumes with this same file.
    """
    plt.errorbar(energies / units.eV, Veffs / units.km3, unc_Veffs / units.km3, marker='o')
    plt.ylabel(r'Effective volume [km$^{3}$]')
    plt.xlabel('Neutrino energy [eV]')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    """
    To end with, we can export the data to a human-readable json (or yaml) file.
    """
    export(output_file, data_Veff, export_format='json')
