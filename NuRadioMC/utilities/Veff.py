import numpy as np
import h5py
from scipy import interpolate
import glob
from NuRadioMC.utilities import units
from six import iteritems
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

    triggered = np.copy(fin['triggered'])

    if (len(triggered) == 0):
        return triggered

    mask_secondaries = np.array(fin['n_interaction']) > 1
    if (True not in mask_secondaries):
        return triggered

    # We count the multiple triggering bangs as a single triggered event
    for event_id in np.unique(np.array(fin['event_ids'])[mask_secondaries]):
        mask_interactions = np.array(fin['event_ids']) == event_id
        multiple_interaction_indexes = np.argwhere(np.array(fin['event_ids']) == event_id)[0]
        if (len(multiple_interaction_indexes) == 1):
            continue

        for int_index in multiple_interaction_indexes[1:]:
            triggered[int_index] = False
        triggered[multiple_interaction_indexes[0]] = True

    return triggered


def get_Aeff_proposal(folder, trigger_combinations={}, zenithbins=False):
    """
    calculates the effective area from NuRadioMC hdf5 files calculated using
    PROPOSAL for propagating surface muons.

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
    Aeffs: floats list
        Effective volumes (m^2)
    Aeffs_error: floats list
        Effective volume uncertainties (m^2)
    SNR: floats list
        Signal to noise ratios
    trigger_names: string list
        Trigger names
    [thetamin, thetamax]: [float, float]
        Mimimum and maximum zenith angles
    """
    trigger_names = None
    trigger_names_dict = {}
    Aeffs = {}
    SNR = {}
    Aeffs_error = {}
    Es = []
    prev_deposited = None
    deposited = False

    for iF, filename in enumerate(sorted(glob.glob(os.path.join(folder, '*.hdf5')))):
        print(f"reading {filename}")
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
                Aeffs[trigger_name] = []
                Aeffs_error[trigger_name] = []
                trigger_names_dict[trigger_name] = iT
            break

    trigger_combinations['all_triggers'] = {'triggers': trigger_names}
    print("Trigger names:", trigger_names)

    for iF, filename in enumerate(sorted(glob.glob(os.path.join(folder, '*.hdf5')))):
        fin = h5py.File(filename, 'r')
        E = fin.attrs['Emin']
        Es.append(E)

        weights = np.array(fin['weights'])
        # triggered = np.array(fin['triggered'])
        triggered = get_triggered(fin)
        n_events = fin.attrs['n_events']
        if(trigger_names is None):
            trigger_names = fin.attrs['trigger_names']
            for iT, trigger_name in enumerate(trigger_names):
                Aeffs[trigger_name] = []
                Aeffs_error[trigger_name] = []
                trigger_names_dict[trigger_name] = iT
            print(trigger_names)
        else:
            if(np.any(trigger_names != fin.attrs['trigger_names'])):

                if(triggered.size == 0 and fin.attrs['trigger_names'].size == 0):
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
        area = np.pi * (rmax ** 2 - rmin ** 2)
        V = area * dZ
        Vrms = fin.attrs['Vrms']

        # Solid angle needed for the effective volume calculations
        omega = np.abs(phimax - phimin) * np.abs(np.cos(thetamin) - np.cos(thetamax))

        for iT, trigger_name in enumerate(trigger_names):
            triggered = np.array(fin['multiple_triggers'][:, iT], dtype=np.bool)
            Aeff = area * np.sum(weights[triggered]) / n_events
            Aeffs[trigger_name].append(Aeff)
            try:
                Aeffs_error[trigger_name].append(Aeff / np.sum(weights[triggered]) ** 0.5)
            except:
                Aeffs_error[trigger_name].append(np.nan)

        for trigger_name, values in iteritems(trigger_combinations):
            indiv_triggers = values['triggers']
            if(trigger_name not in Aeffs):
                Aeffs[trigger_name] = []
                Aeffs_error[trigger_name] = []
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
                        As = np.max(np.nan_to_num(fin['max_amp_ray_solution']), axis=-1)  # we use the this quantity because it is always computed before noise is added!
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
                    As = np.max(np.nan_to_num(fin['max_amp_ray_solution']), axis=-1)  # we use the this quantity because it is always computed before noise is added!
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
                mask = np.array([sol[i, values['ray_channel'], max_amps[i]] == values['ray_solution'] for i in range(len(max_amps))], dtype=np.bool)
                triggered = triggered & mask

            Aeff = area * np.sum(weights[triggered]) / n_events

            if('efficiency' in values.keys()):
                SNReff, eff = np.loadtxt("analysis_efficiency_{}.csv".format(values['efficiency']), delimiter=",", unpack=True)
                get_eff = interpolate.interp1d(SNReff, eff, bounds_error=False, fill_value=(0, eff[-1]))
                As = np.max(np.max(np.nan_to_num(fin['max_amp_ray_solution']), axis=-1)[:, np.append(range(0, 8), range(12, 20))], axis=-1)  # we use the this quantity because it is always computed before noise is added!
                if('efficiency_scale' in values.keys()):
                    As *= values['efficiency_scale']
                e = get_eff(As / Vrms)
                Aeff = area * np.sum((weights * e)[triggered]) / n_events

            Aeffs[trigger_name].append(Aeff)
            Aeffs_error[trigger_name].append(Aeff / np.sum(weights[triggered]) ** 0.5)
    for trigger_name in Aeffs.keys():
        Aeffs[trigger_name] = np.array(Aeffs[trigger_name])
        Aeffs_error[trigger_name] = np.array(Aeffs_error[trigger_name])

    if zenithbins:
        return np.array(Es), Aeffs, Aeffs_error, SNR, trigger_names, [thetamin, thetamax], deposited
    else:
        return np.array(Es), Aeffs, Aeffs_error, SNR, trigger_names, deposited


def get_Veff_water_equivalent(Veff, density_medium=0.917 * units.g / units.cm ** 3, density_water=1 * units.g / units.cm ** 3):
    """
    convenience function to converte the effective volume of a medium with density `density_medium` to the 
    water equivalent effective volume
    
    Parameters
    ----------
    Veff: float or array
        the effective volume
    dentity_medium: float (optional)
        the density of the medium of the Veff simulation (default deep ice)
    density water: float (optional)
        the density of water
        
    Returns: water equivalen effective volume
    """
    return Veff * density_medium / density_water


def get_Veff(folder, trigger_combinations={}, station=101):
    """
    calculates the effective volume from NuRadioMC hdf5 files
    
    the effective volume is NOT normalized to a water equivalent. 

    Parameters
    ----------
    folder: string
        folder conaining the hdf5 files, one per energy
    trigger_combinations: dict, optional
        keys are the names of triggers to calculate. Values are dicts again:
            * 'triggers': list of strings
                name of individual triggers that are combines with an OR
            the following additional options are optional
            * 'efficiency': string
                the signal efficiency vs. SNR (=Vmax/Vrms) to use. E.g. 'Chris'
            * 'efficiency_scale': float
                rescaling of the efficiency curve by SNR' = SNR * scale
            * 'n_reflections': int
                the number of bottom reflections of the ray tracing solution that likely triggered
                assuming that the solution with the shortest travel time caused the trigger, only considering channel 0

    station: int
        the station that should be considered

    Returns
    ----------
    list of dictionary. Each file is one entry. The dictionary keys store all relevant properties
    """
    Veff_output = []
    trigger_names = None
    trigger_names_dict = {}
    prev_deposited = None
    deposited = False

    if(len(glob.glob(os.path.join(folder, '*.hdf5'))) == 0):
        raise FileNotFoundError(f"couldnt find any hdf5 file in folder {folder}")
    for iF, filename in enumerate(sorted(glob.glob(os.path.join(folder, '*.hdf5')))):
        print(f"reading {filename}")
        fin = h5py.File(filename, 'r')
        if 'deposited' in fin.attrs:
            deposited = fin.attrs['deposited']
            if prev_deposited is None:
                prev_deposited = deposited
            elif prev_deposited != deposited:
                raise AttributeError("The deposited parameter is not consistent among the input files!")

        if('trigger_names' in fin.attrs):
            trigger_names = fin.attrs['trigger_names']
        if(len(trigger_names) > 0):
            for iT, trigger_name in enumerate(trigger_names):
                trigger_names_dict[trigger_name] = iT
            break

    trigger_combinations['all_triggers'] = {'triggers': trigger_names}
    print("Trigger names:", trigger_names)

    for iF, filename in enumerate(sorted(glob.glob(os.path.join(folder, '*.hdf5')))):
        fin = h5py.File(filename, 'r')
        out = {}
        E = fin.attrs['Emin']
        if(fin.attrs['Emax'] != E):
            raise AttributeError("min and max energy do not match!")
        out['energy'] = E

        weights = np.array(fin['weights'])
        # triggered = np.array(fin['triggered'])
        triggered = get_triggered(fin)
        n_events = fin.attrs['n_events']
        if(trigger_names is None):
            trigger_names = fin.attrs['trigger_names']
            for iT, trigger_name in enumerate(trigger_names):
                trigger_names_dict[trigger_name] = iT
        else:
            if(np.any(trigger_names != fin.attrs['trigger_names'])):
                if(triggered.size == 0 and fin.attrs['trigger_names'].size == 0):
                    print("file {} has no triggering events. Using trigger names from another file".format(filename))
                else:
                    print("file {} has inconsistent trigger names: {}".format(filename, fin.attrs['trigger_names']))
                    raise

        # calculate effective
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
        area = np.pi * (rmax ** 2 - rmin ** 2)
        V = area * dZ
        Vrms = fin.attrs['Vrms']

        # Solid angle needed for the effective volume calculations
        out['domega'] = np.abs(phimax - phimin) * np.abs(np.cos(thetamin) - np.cos(thetamax))
        out['thetamin'] = thetamin
        out['thetamax'] = thetamax
        out['deposited'] = deposited
        out['Veffs'] = {}
        out['n_triggered_weighted'] = {}
        out['SNRs'] = {}

        if(triggered.size == 0):
            for iT, trigger_name in enumerate(trigger_names):
                out['Veffs'][trigger_name] = [0, 0, 0]
            for trigger_name, values in iteritems(trigger_combinations):
                out['Veffs'][trigger_name] = [0, 0, 0]
        else:
            for iT, trigger_name in enumerate(trigger_names):
                triggered = np.array(fin['multiple_triggers'][:, iT], dtype=np.bool)
                Veff = V * np.sum(weights[triggered]) / n_events
                Veff_error = 0
                if(np.sum(weights[triggered]) > 0):
                    Veff_error = Veff / np.sum(weights[triggered]) ** 0.5
                out['Veffs'][trigger_name] = [Veff, Veff_error, np.sum(weights[triggered])]

            for trigger_name, values in iteritems(trigger_combinations):
                indiv_triggers = values['triggers']
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
                        if(trigger_name not in out['SNR']):
                            out['SNR'][trigger_name] = {}
                        masks = np.zeros_like(triggered)
                        for iS in range(len(values['min_sigma'])):
    #                         As = np.array(fin['maximum_amplitudes'])
                            As = np.max(np.nan_to_num(fin['max_amp_ray_solution']), axis=-1)  # we use the this quantity because it is always computed before noise is added!
                            As_sorted = np.sort(As[:, values['channels'][iS]], axis=1)
                            # the smallest of the three largest amplitudes
                            max_amplitude = As_sorted[:, -values['n_channels'][iS]]
                            mask = np.sum(As[:, values['channels'][iS]] >= (values['min_sigma'][iS] * Vrms), axis=1) >= values['n_channels'][iS]
                            masks = masks | mask
                            out['SNR'][trigger_name][iS] = max_amplitude[mask] / Vrms
                        triggered = triggered & masks
                    else:
    #                     As = np.array(fin['maximum_amplitudes'])
                        As = np.max(np.nan_to_num(fin['max_amp_ray_solution']), axis=-1)  # we use the this quantity because it is always computed before noise is added!
        #                 print(As.shape)
    #                     print(trigger_name, values['channels'])
                        As_sorted = np.sort(As[:, values['channels']], axis=1)
                        max_amplitude = As_sorted[:, -values['n_channels']]  # the smallest of the three largest amplitudes
        #                 print(np.sum(As_sorted[:, -values['n_channels']] > (values['min_sigma'] * Vrms)))
                        mask = np.sum(As[:, values['channels']] >= (values['min_sigma'] * Vrms), axis=1) >= values['n_channels']
        #                 max_amplitude[~mask] = 0
                        out['SNR'][trigger_name] = As_sorted[mask] / Vrms
        #                 print(Vrms, mask.shape, np.sum(mask))
                        triggered = triggered & mask
                if('ray_solution' in values.keys()):
                    As = np.array(fin['max_amp_ray_solution'])
                    max_amps = np.argmax(As[:, values['ray_channel']], axis=-1)
                    sol = np.array(fin['ray_tracing_solution_type'])
    #                 print(sol[:,values['ray_channel']][max_amps].shape)
    #                 print(max_amps.shape)
    #                 a = 1/0
                    mask = np.array([sol[i, values['ray_channel'], max_amps[i]] == values['ray_solution'] for i in range(len(max_amps))], dtype=np.bool)
                    triggered = triggered & mask

                if('n_reflections' in values.keys()):
                    triggered = triggered & (np.array(fin[f'station_{station:d}/ray_tracing_reflection'])[:, 0, 0] == values['n_reflections'])

                Veff = V * np.sum(weights[triggered]) / n_events

                if('efficiency' in values.keys()):
                    SNReff, eff = np.loadtxt("analysis_efficiency_{}.csv".format(values['efficiency']), delimiter=",", unpack=True)
                    get_eff = interpolate.interp1d(SNReff, eff, bounds_error=False, fill_value=(0, eff[-1]))
                    As = np.max(np.max(np.nan_to_num(fin['max_amp_ray_solution']), axis=-1)[:, np.append(range(0, 8), range(12, 20))], axis=-1)  # we use the this quantity because it is always computed before noise is added!
                    if('efficiency_scale' in values.keys()):
                        As *= values['efficiency_scale']
                    e = get_eff(As / Vrms)
                    Veff = V * np.sum((weights * e)[triggered]) / n_events

                out['Veffs'][trigger_name] = [Veff, Veff / np.sum(weights[triggered]) ** 0.5, np.sum(weights[triggered])]
        Veff_output.append(out)

    return Veff_output


def get_Veff_array(data):
    """
    calculates a multi dimensional array of effective volume calculations for fast slicing
    
    the array dimensions are (energy, zenith bin, triggername, 3) where the
    last tuple is the effective volume, its uncertainty and the weighted sum of triggered events
    
    Parameters
    -----------
    data: dict
        the result of the `get_Veff` function
    
    Returns
    --------
     * (n_energy, n_zenith_bins, n_triggernames, 3) dimensional array of floats
     * array of unique energies
     * array of unique zenith bins
     * array of unique trigger names
     
     
    Examples
    ---------
    
    To plot the full sky effective volume for 'all_triggers' do
    
    ```
    output, uenergies, uzenith_bins, utrigger_names = get_Veff_array(data)
    
    
    fig, ax = plt.subplots(1, 1)
    tname = "all_triggers"
    Veff = np.average(output[:,:,get_index(tname, utrigger_names),0], axis=1)
    Vefferror = Veff / np.sum(output[:,:,get_index(tname, utrigger_names),2], axis=1)**0.5
    ax.errorbar(uenergies/units.eV, Veff/units.km**3 * 4 * np.pi, yerr=Vefferror/units.km**3 * 4 * np.pi, fmt='-o', label=tname)
    
    ax.legend()
    ax.semilogy(True)
    ax.semilogx(True)
    fig.tight_layout()
    plt.show()
    ```
    
    
    To plot the effective volume for different declination bands do
    
    ```
    fig, ax = plt.subplots(1, 1)
    tname = "LPDA_2of4_100Hz"
    iZ = 9
    Veff = output[:,iZ,get_index(tname, utrigger_names)]
    ax.errorbar(uenergies/units.eV, Veff[:,0]/units.km**3, yerr=Veff[:,1]/units.km**3,
                label=f"zenith bin {uzenith_bins[iZ][0]/units.deg:.0f} - {uzenith_bins[iZ][1]/units.deg:.0f}")
    
    iZ = 8
    Veff = output[:,iZ,get_index(tname, utrigger_names)]
    ax.errorbar(uenergies/units.eV, Veff[:,0]/units.km**3, yerr=Veff[:,1]/units.km**3,
                label=f"zenith bin {uzenith_bins[iZ][0]/units.deg:.0f} - {uzenith_bins[iZ][1]/units.deg:.0f}")
    iZ = 7
    Veff = output[:,iZ,get_index(tname, utrigger_names)]
    ax.errorbar(uenergies/units.eV, Veff[:,0]/units.km**3, yerr=Veff[:,1]/units.km**3,
                label=f"zenith bin {uzenith_bins[iZ][0]/units.deg:.0f} - {uzenith_bins[iZ][1]/units.deg:.0f}")
    iZ = 10
    Veff = output[:,iZ,get_index(tname, utrigger_names)]
    ax.errorbar(uenergies/units.eV, Veff[:,0]/units.km**3, yerr=Veff[:,1]/units.km**3,
                label=f"zenith bin {uzenith_bins[iZ][0]/units.deg:.0f} - {uzenith_bins[iZ][1]/units.deg:.0f}")
    
    
    ax.legend()
    ax.semilogy(True)
    ax.semilogx(True)
    fig.tight_layout()
    plt.show()
    ```
    
    """
    energies = []
    zenith_bins = []
    trigger_names = []
    for d in data:
        energies.append(d['energy'])
        zenith_bins.append([d['thetamin'], d['thetamax']])
        for triggername in d['Veffs']:
            trigger_names.append(triggername)

    energies = np.array(energies)
    zenith_bins = np.array(zenith_bins)
    trigger_names = np.array(trigger_names)
    uenergies = np.unique(energies)
    uzenith_bins = np.unique(zenith_bins, axis=0)
    utrigger_names = np.unique(trigger_names)
    output = np.zeros((len(uenergies), len(uzenith_bins), len(utrigger_names), 3))
    print(f"unique energies {uenergies}")
    print(f"unique zenith angle bins {uzenith_bins/units.deg}")
    print(f"unique energies {utrigger_names}")

    for d in data:
        iE = np.squeeze(np.argwhere(d['energy'] == uenergies))
        iT = np.squeeze(np.argwhere([d['thetamin'], d['thetamax']] == uzenith_bins))[0][0]
        for triggername, Veff in d['Veffs'].items():
            iTrig = np.squeeze(np.argwhere(triggername == utrigger_names))
            output[iE, iT, iTrig] = Veff
#                 print(f"{iE}  {iT} {iTrig} {Veff}")
    return output, uenergies, uzenith_bins, utrigger_names


def get_index(value, array):
    return np.squeeze(np.argwhere(value == array))


def export(filename, data, trigger_names=None, export_format='yaml'):
    """
    export effective volumes into a human readable JSON or YAML file

    Parameters
    ----------
    filename: string
        the output filename of the JSON file
    data: array
        the output of the `getVeff` function
    trigger_names: list of strings (optional, default None)
        save only specific trigger names, if None all triggers are exported
    export_format: string (default "yaml")
        specify output format, choose
        * "yaml"
        * "json"
    """
    output = []
    for i in range(len(data)):
        tmp = {}
        for key in data[i]:
            if (key != 'Veffs'):
                if isinstance(data[i][key], np.generic):
                    tmp[key] = data[i][key].item()
                else:
                    tmp[key] = data[i][key]
        tmp['Veffs'] = {}
        for trigger_name in data[i]['Veffs']:
            if(trigger_names is None or trigger_name in trigger_names):
                print(trigger_name)
                tmp['Veffs'][trigger_name] = []
                for value in data[i]['Veffs'][trigger_name]:
                    tmp['Veffs'][trigger_name].append(float(value))
        output.append(tmp)

    with open(filename, 'w') as fout:
        if(export_format == 'yaml'):
            import yaml
            yaml.dump(output, fout)
        elif(export_format == 'json'):
            json.dump(output, fout, sort_keys=True, indent=4)


def exportAeffPerZenith(folderlist, outputfile):
    """
    export effective areas into a human readable JSON file
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

        Es, Aeffs, Aeffs_error, SNR, trigger_names, thetas, deposited = get_Aeff_proposal(folder, zenithbins=True)
        output[thetas[0]] = {}

        for trigger_name in trigger_names:
            output[thetas[0]][trigger_name] = {}
            if deposited:
                output[thetas[0]][trigger_name]['deposited_energies'] = list(Es)
            else:
                output[thetas[0]][trigger_name]['energies'] = list(Es)
            output[thetas[0]][trigger_name]['Aeff'] = list(Aeffs[trigger_name])
            output[thetas[0]][trigger_name]['Aeff_error'] = list(Aeffs_error[trigger_name])

    with open(outputfile, 'w+') as fout:

        json.dump(output, fout, sort_keys=True, indent=4)
