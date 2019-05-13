import numpy as np
import h5py
from scipy import interpolate
import glob
from NuRadioMC.utilities import units
import json
import os


# collection of utility function regarding the calculation of the effective volume of a neutrino detector

def get_triggered(fin):
    """
    Computes an array indicating the triggered events.
    If a double bang is seen, removes the second bang from the actual triggered
    array so as not to count twice the same event for the effective volume.

    Parameters
    ----------
    fin: dictionary
       Dictionary containing the output data sets from the simulation

    Returns
    -------
    triggered: numpy array with bools
       The bools indicate if the events have triggered
    """

    triggered = fin['triggered'][:]

    if (len(triggered) == 0):
        return triggered

    second_bang_indexes = np.argwhere(np.array(fin['n_interaction']) >= 2)
    if(second_bang_indexes.shape[0] > 1):
        second_bang_indexes = np.squeeze(second_bang_indexes)
    elif(second_bang_indexes.shape[0] == 1):
        second_bang_indexes = second_bang_indexes[0]
    else:
        return triggered

    for second_bang_index in second_bang_indexes:
        # Second interactions
        if (fin['event_ids'][second_bang_index-1] == fin['event_ids'][second_bang_index]
            and fin['triggered'][second_bang_index-1] == True
            and fin['triggered'][second_bang_index] == True):

            triggered[second_bang_index] = False

        # Third interactions
        elif (fin['event_ids'][second_bang_index-2] == fin['event_ids'][second_bang_index]
              and fin['triggered'][second_bang_index-2] == True
              and fin['triggered'][second_bang_index] == True):

            triggered[second_bang_index] = False

    return triggered

def get_Veff(folder, trigger_combinations={}, zenithbins=False):
    """
    calculates the effective volume from NuRadioMC hdf5 files

    Parameters
    ----------
    folder: string
        folder conaining the hdf5 files, one per energy
    trigger_combinations: dict, optional
        keys are the names of triggers to calculate. Values are dicts again:
            * 'triggers': list of strings
                name of individual triggers that are combines with an OR
            * 'efficiency': string
                the signal efficiency vs. SNR (=Vmax/Vrms) to use. E.g. 'Chris'
            * 'efficiency_scale': float
                rescaling of the efficiency curve by SNR' = SNR * scale
    zenithbins: bool
        If true, returns the minimum and maximum zenith angles

    Returns
    ----------
    np.array(Es): numpy floats array
        Smallest energy for each bin
    Veffs: floats list
        Effective volumes
    Veffs_error: floats list
        Effective volume uncertainties
    SNR: floats list
        Signal to noise ratios
    trigger_names: string list
        Trigger names
    [thetamin, thetamax]: [float, float]
        Mimimum and maximum zenith angles
    deposited: bool
        True if the energies are deposited energies
        False if the energies are primary neutrino energies
    """
    trigger_names = None
    trigger_names_dict = {}
    Veffs = {}
    SNR = {}
    Veffs_error = {}
    Es = []
    prev_deposited = None
    deposited = False

    for iF, filename in enumerate(sorted(glob.glob(os.path.join(folder, '*.hdf5')))):
        fin = h5py.File(filename, 'r')
        if 'deposited' in fin.attrs:
            deposited = fin.attrs['deposited']
            if prev_deposited is None:
                prev_deposited = deposited
            elif prev_deposited != deposited:
                print("Warning! The deposited parameter is not consistent!")

        if('trigger_names' in fin.attrs):
            trigger_names = fin.attrs['trigger_names']
        if(len(trigger_names) > 0):
            for iT, trigger_name in enumerate(trigger_names):
                Veffs[trigger_name] = []
                Veffs_error[trigger_name] = []
                trigger_names_dict[trigger_name] = iT
            break

    trigger_combinations['all_triggers'] = {'triggers': trigger_names}
    print("Trigger names:", trigger_names)

    for iF, filename in enumerate(sorted(glob.glob(os.path.join(folder, '*.hdf5')))):
        print(filename)
        fin = h5py.File(filename, 'r')
        E = fin.attrs['Emin']
        Es.append(E)

        weights = np.array(fin['weights'])
        #triggered = np.array(fin['triggered'])
        triggered = get_triggered(fin)
        n_events = fin.attrs['n_events']
        if(trigger_names is None):
            trigger_names = fin.attrs['trigger_names']
            for iT, trigger_name in enumerate(trigger_names):
                Veffs[trigger_name] = []
                Veffs_error[trigger_name] = []
                trigger_names_dict[trigger_name] = iT
            print(trigger_names)
        else:
            if(np.any(trigger_names != fin.attrs['trigger_names'])):

                if( triggered.size == 0 and fin.attrs['trigger_names'].size == 0 ):
                    print("file {} has not triggering events. Using trigger names from another file".format(filename))
                else:
                    print("file {} has inconsistent trigger names: {}".format(filename, fin.attrs['trigger_names']))
                    raise

        # calculate effective
        density_ice = 0.9167 * units.g / units.cm ** 3
        density_water = 997 * units.kg / units.m ** 3
        rmin = fin.attrs['rmin']
        rmax = fin.attrs['rmax']
        thetamin = 0 
        thetamax = np.pi
        phimin = 0
        phimax = 2 * np.pi
        if('thetamin' in fin.attrs):
            thetamin = fin.attrs['thetamin']
        if('thetamax' in fin.attrs):
            thetamax = fin.attrs['thetamax']
        if('phimin' in fin.attrs):
            fin.attrs['phimin']
        if('phimax' in fin.attrs):
            fin.attrs['phimax']
        dZ = fin.attrs['zmax'] - fin.attrs['zmin']
        V = np.pi * (rmax**2 - rmin**2) * dZ
        Vrms = fin.attrs['Vrms']

        # Solid angle needed for the effective volume calculations
        omega = np.abs(phimax - phimin) * np.abs( np.cos(thetamin)-np.cos(thetamax) )

        for iT, trigger_name in enumerate(trigger_names):
            triggered = np.array(fin['multiple_triggers'][:, iT], dtype=np.bool)
            Veff = V * density_ice / density_water * omega * np.sum(weights[triggered]) / n_events
            Veffs[trigger_name].append(Veff)
            try:
                Veffs_error[trigger_name].append(Veff / np.sum(weights[triggered])**0.5)
            except:
                Veffs_error[trigger_name].append(np.nan)
#             print("{}: log(E) = {:.3g}, Veff = {:.3f}km^3 st".format(trigger_name, np.log10(E), Veff / units.km**3))

        for trigger_name, values in trigger_combinations.iteritems():
            indiv_triggers = values['triggers']
            if(trigger_name not in Veffs):
                Veffs[trigger_name] = []
                Veffs_error[trigger_name] = []
            triggered = np.zeros_like(fin['multiple_triggers'][:, 0], dtype=np.bool)
            if(isinstance(indiv_triggers, str)):
                triggered = triggered | np.array(fin['multiple_triggers'][:, trigger_names_dict[indiv_triggers]], dtype=np.bool)
            else:
                for indiv_trigger in indiv_triggers:
                    triggered = triggered | np.array(fin['multiple_triggers'][:, trigger_names_dict[indiv_trigger]], dtype=np.bool)
            if 'triggerAND' in values:
                triggered = triggered & np.array(fin['multiple_triggers'][:, trigger_names_dict[values['triggerAND']]], dtype=np.bool)
            if 'notriggers' in values:
                indiv_triggers = values['notriggers']
                if(isinstance(indiv_triggers, str)):
                    triggered = triggered & ~np.array(fin['multiple_triggers'][:, trigger_names_dict[indiv_triggers]], dtype=np.bool)
                else:
                    for indiv_trigger in indiv_triggers:
                        triggered = triggered & ~np.array(fin['multiple_triggers'][:, trigger_names_dict[indiv_trigger]], dtype=np.bool)
            if('min_sigma' in values.keys()):
                if(isinstance(values['min_sigma'], list)):
                    if(trigger_name not in SNR):
                        SNR[trigger_name] = {}
                    masks = np.zeros_like(triggered)
                    for iS in range(len(values['min_sigma'])):
#                         As = np.array(fin['maximum_amplitudes'])
                        As = np.max(np.nan_to_num(fin['max_amp_ray_solution']), axis=-1) # we use the this quantity because it is always computed before noise is added!
                        As_sorted = np.sort(As[:, values['channels'][iS]], axis=1)
                        # the smallest of the three largest amplitudes
                        max_amplitude = As_sorted[:, -values['n_channels'][iS]]
                        mask = np.sum(As[:, values['channels'][iS]] >= (values['min_sigma'][iS] * Vrms), axis=1) >= values['n_channels'][iS]
                        masks = masks | mask
                        if(iS not in SNR[trigger_name]):
                            SNR[trigger_name][iS] = []
                        SNR[trigger_name][iS].append([max_amplitude[mask] / Vrms])
                    triggered = triggered & masks
                else:
                    if(trigger_name not in SNR):
                        SNR[trigger_name] = []
#                     As = np.array(fin['maximum_amplitudes'])
                    As = np.max(np.nan_to_num(fin['max_amp_ray_solution']), axis=-1) # we use the this quantity because it is always computed before noise is added!
    #                 print(As.shape)
#                     print(trigger_name, values['channels'])
                    As_sorted = np.sort(As[:, values['channels']], axis=1)
                    max_amplitude = As_sorted[:, -values['n_channels']]  # the smallest of the three largest amplitudes
    #                 print(np.sum(As_sorted[:, -values['n_channels']] > (values['min_sigma'] * Vrms)))
                    mask = np.sum(As[:, values['channels']] >= (values['min_sigma'] * Vrms), axis=1) >= values['n_channels']
    #                 max_amplitude[~mask] = 0
                    SNR[trigger_name].append(As_sorted[mask] / Vrms)
    #                 print(Vrms, mask.shape, np.sum(mask))
                    triggered = triggered & mask
            if('ray_solution' in values.keys()):
                As = np.array(fin['max_amp_ray_solution'])
                max_amps = np.argmax(As[:, values['ray_channel']], axis=-1)
                sol = np.array(fin['ray_tracing_solution_type'])
#                 print(sol[:,values['ray_channel']][max_amps].shape)
#                 print(max_amps.shape)
#                 a = 1/0
                mask = np.array([sol[i,values['ray_channel'], max_amps[i]] == values['ray_solution'] for i in range(len(max_amps))], dtype=np.bool)
                triggered = triggered & mask

            Veff = V * density_ice / density_water * omega * np.sum(weights[triggered]) / n_events

            if('efficiency' in values.keys()):
                SNReff, eff = np.loadtxt("analysis_efficiency_{}.csv".format(values['efficiency']), delimiter=",", unpack=True)
                get_eff = interpolate.interp1d(SNReff, eff, bounds_error=False, fill_value=(0, eff[-1]))
                As = np.max(np.max(np.nan_to_num(fin['max_amp_ray_solution']), axis=-1)[:,np.append(range(0, 8), range(12, 20))], axis=-1) # we use the this quantity because it is always computed before noise is added!
                if('efficiency_scale' in values.keys()):
                    As *= values['efficiency_scale']
                e = get_eff(As/Vrms)
                Veff = V * density_ice / density_water * omega * np.sum((weights*e)[triggered]) / n_events

            Veffs[trigger_name].append(Veff)
            Veffs_error[trigger_name].append(Veff / np.sum(weights[triggered])**0.5)
    for trigger_name in Veffs.keys():
        Veffs[trigger_name] = np.array(Veffs[trigger_name])
        Veffs_error[trigger_name] = np.array(Veffs_error[trigger_name])

    if zenithbins:
        return np.array(Es), Veffs, Veffs_error, SNR, trigger_names, [thetamin, thetamax], deposited
    else:
        return np.array(Es), Veffs, Veffs_error, SNR, trigger_names, deposited

def exportVeff(filename, trigger_names, Es, Veffs, Veffs_error):
    """
    export effective volumes into a human readable JSON file

    Parameters
    ----------
    filename: string
        the output filename of the JSON file
    trigger_names: list of strings
        the triggers for which the effective volume is exported
    Es: list or array of floats
        the energies
    Veffs: dictionary
        dictionary containing Veffs for each trigger
    Veffs_error: dictionary
        dictionary containing Veff_errors for each trigger


    """
    output = {}
    for trigger_name in trigger_names:
        output[trigger_name] = {}
        output[trigger_name]['energies'] = list(Es)
        output[trigger_name]['Veff'] = list(Veffs[trigger_name])
        output[trigger_name]['Veff_error'] = list(Veffs_error[trigger_name])

    with open(filename, 'w') as fout:
        json.dump(output, fout, sort_keys=True, indent=4)

def exportVeffPerZenith(folderlist, outputfile):
    """
    export effective volumes into a human readable JSON file
    We assume a binning in zenithal angles

    Parameters
    ----------
    folderlist: strings list
        list containing the input folders
    outputfile: string
        name for the output file
    """
    output = {}
    for folder in folderlist:

        Es, Veffs, Veffs_error, SNR, trigger_names, thetas, deposited = get_Veff(folder, zenithbins=True)
        output[thetas[0]] = {}

        for trigger_name in trigger_names:
            output[thetas[0]][trigger_name] = {}
            if deposited:
                output[thetas[0]][trigger_name]['deposited_energies'] = list(Es)
            else:
                output[thetas[0]][trigger_name]['energies'] = list(Es)
            output[thetas[0]][trigger_name]['Veff'] = list(Veffs[trigger_name])
            output[thetas[0]][trigger_name]['Veff_error'] = list(Veffs_error[trigger_name])

    with open(outputfile, 'w+') as fout:

        json.dump(output, fout, sort_keys=True, indent=4)
