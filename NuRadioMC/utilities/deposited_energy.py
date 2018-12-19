import numpy as np
import h5py
import glob
from NuRadioMC.utilities import units
import json
import os

# Utility functions for calculating the 2D deposited energy vs neutrino energy pyplot

def get_Edeps_bins(Edeps, Ebins, Veffs):
    """
    Returns a 2D array with the effective volumes per deposited energy.
    The first index indicates the neutrino energy bin
    and the second one the deposited energy bin.

    Parameters
    ----------
    Edeps: 2D array
        Deposited energies for all the events and all energy bins. The first index
        indicates the neutrino energy bin and the second one the event number.
    Ebins: array
        Array containing the limits for each neutrino energy bin.
    Veffs: array
        Array containing the total effective volume for every neutrino energy bin.
    """

    Edeps_bins = []
    for Edeps_bin, Veff in zip(Edeps, Veffs):
        hist = np.histogram(Edeps_bin, Ebins)[0]
        try:
            hist = Veff/sum(hist) * np.array(hist)
        except:
            hist = 0 * np.array(hist)
        Edeps_bins.append( list( hist ) )

    return Edeps_bins

def get_Edep(folder):
    """
    Calculates the deposited energy as a function of the incident neutrino energy.
    """
    trigger_names = None
    trigger_names_dict = {}
    Veffs = {}
    SNR = {}
    Veffs_error = {}
    Es = []
    Edeps = {}

    for iF, filename in enumerate(sorted(glob.glob(os.path.join(folder, '*/*.hdf5')))):
        print(filename)
        fin = h5py.File(filename, 'r')
        E = fin.attrs['Emin']
        Emax = fin.attrs['Emax']
        Es.append(E)
        weights = np.array(fin['weights'])
        triggered = np.array(fin['triggered'])
        n_events = fin.attrs['n_events']
        if(trigger_names is None):
            trigger_names = fin.attrs['trigger_names']
            for iT, trigger_name in enumerate(trigger_names):
                Veffs[trigger_name] = []
                Veffs_error[trigger_name] = []
                trigger_names_dict[trigger_name] = iT
                Edeps[trigger_name] = []
            print(trigger_names)
        else:
            if(np.any(trigger_names != fin.attrs['trigger_names'])):
                print("file {} has inconsistent trigger names: {}".format(filename, fin.attrs['trigger_names']))
                raise

        # calculate effective
        density_ice = 0.9167 * units.g / units.cm ** 3
        density_water = 997 * units.kg / units.m ** 3
        rmin = fin.attrs['rmin']
        rmax = fin.attrs['rmax']
        thetamin = fin.attrs['thetamin']
        thetamax = fin.attrs['thetamax']
        dZ = fin.attrs['zmax'] - fin.attrs['zmin']
        V = np.pi * (rmax**2 - rmin**2) * dZ
        Vrms = fin.attrs['Vrms']

        for iT, trigger_name in enumerate(trigger_names):
            triggered = np.array(fin['multiple_triggers'][:, iT], dtype=np.bool)
            Veff = V * density_ice / density_water * 4 * np.pi * np.sum(weights[triggered]) / n_events
            Veffs[trigger_name].append(Veff)
            Veffs_error[trigger_name].append(Veff / np.sum(weights[triggered])**0.5)

            # Selecting the deposited energies of the events that triggered
            #Edeps[trigger_name].append(np.array(fin['deposited_energies'][triggered]))
            Edeps[trigger_name].append(np.array(fin['energies'][triggered]))

    Edeps_bins = {}
    Ebins = list(Es) + [Emax]
    for iT, trigger_name in enumerate(trigger_names):
        Edeps_bins[trigger_name] = get_Edeps_bins(Edeps[trigger_name], Ebins, Veffs[trigger_name])

    return np.array(Es), Veffs, Veffs_error, Edeps_bins, trigger_names, [thetamin, thetamax]

def exportVeffEdepPerZenith(folderlist, outputfile):
    """
    export effective volumes per deposited energy histograms into a human
    readable JSON file. For each neutrino energy bin, the histograms are normalised
    to the effective volume for that particular bin.
    We assume a binning in zenithal angles
    """
    output = {}
    for folder in folderlist:

        Es, Veffs, Veffs_error, Edeps, trigger_names, thetas = get_Edep(folder)
        output[thetas[0]] = {}

        for trigger_name in trigger_names:
            output[thetas[0]][trigger_name] = {}
            output[thetas[0]][trigger_name]['energies'] = list(Es)
            output[thetas[0]][trigger_name]['Veff'] = list(Veffs[trigger_name])
            output[thetas[0]][trigger_name]['Veff_error'] = list(Veffs_error[trigger_name])
            output[thetas[0]][trigger_name]['Veff_per_Edep_bin'] = list(Edeps[trigger_name])

    with open(outputfile, 'w+') as fout:
        json.dump(output, fout, sort_keys=False, indent=4)
