import numpy as np
import h5py
from scipy import interpolate
import glob
from six import iteritems
import json
import os
import copy

from NuRadioReco.utilities import units
from NuRadioMC.EvtGen.generator import get_projected_area_cylinder, get_projected_area_cylinder_integral

import logging
logger = logging.getLogger("Veff")
logging.basicConfig()

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
        multiple_interaction_indexes = np.squeeze(np.argwhere(np.array(fin['event_ids']) == event_id))
        if (len(multiple_interaction_indexes) == 1):
            continue

        for int_index in multiple_interaction_indexes[1:]:
            triggered[int_index] = False
        triggered[multiple_interaction_indexes[0]] = True

    return triggered


def get_Aeff_proposal(folder, trigger_combinations={}, station=101):
    """
    Calculates the effective area from NuRadioMC hdf5 files simulated after
    using PROPOSAL as the lepton propagator. The interaction length is already
    factorised thanks to the energy losses returned by PROPOSAL, which in turn
    can trigger our radio array.

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
    Aeff_output = []
    trigger_names = None
    trigger_names_dict = {}
    prev_deposited = None
    deposited = False

    if(len(glob.glob(os.path.join(folder, '*.hdf5'))) == 0):
        raise FileNotFoundError(f"couldnt find any hdf5 file in folder {folder}")
    for iF, filename in enumerate(sorted(glob.glob(os.path.join(folder, '*.hdf5')))):
        logger.info(f"reading {filename}")
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
    logger.info("Trigger names:", trigger_names)

    for iF, filename in enumerate(sorted(glob.glob(os.path.join(folder, '*.hdf5')))):
        fin = h5py.File(filename, 'r')
        out = {}
        E = fin.attrs['Emin']
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
                    logger.warning("file {} has no triggering events. Using trigger names from another file".format(filename))
                else:
                    logger.warning("file {} has inconsistent trigger names: {}".format(filename, fin.attrs['trigger_names']))
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
        # The used area must be the projected area, perpendicular to the incoming
        # flux, which leaves us with the following correction. Remember that the
        # zenith bins must be small for the effective area to be correct.
        proj_area = area * 0.5 * (np.abs(np.cos(thetamin)) + np.abs(np.cos(thetamax)))
        V = area * dZ
        Vrms = fin.attrs['Vrms']

        # Solid angle needed for the effective volume calculations
        out['domega'] = np.abs(phimax - phimin) * np.abs(np.cos(thetamin) - np.cos(thetamax))
        out['thetamin'] = thetamin
        out['thetamax'] = thetamax
        out['deposited'] = deposited
        out['Aeffs'] = {}
        out['n_triggered_weighted'] = {}
        out['SNRs'] = {}

        if(triggered.size == 0):
            for iT, trigger_name in enumerate(trigger_names):
                out['Aeffs'][trigger_name] = [0, 0, 0]
            for trigger_name, values in iteritems(trigger_combinations):
                out['Aeffs'][trigger_name] = [0, 0, 0]
        else:
            for iT, trigger_name in enumerate(trigger_names):
                triggered = np.array(fin['multiple_triggers'][:, iT], dtype=np.bool)
                Aeff = proj_area * np.sum(weights[triggered]) / n_events
                Aeff_error = 0
                if(np.sum(weights[triggered]) > 0):
                    Aeff_error = Aeff / np.sum(weights[triggered]) ** 0.5
                out['Aeffs'][trigger_name] = [Aeff, Aeff_error, np.sum(weights[triggered])]

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
                        As = np.max(np.nan_to_num(fin['max_amp_ray_solution']), axis=-1)  # we use the this quantity because it is always computed before noise is added!

                        As_sorted = np.sort(As[:, values['channels']], axis=1)
                        max_amplitude = As_sorted[:, -values['n_channels']]  # the smallest of the three largest amplitudes
                        mask = np.sum(As[:, values['channels']] >= (values['min_sigma'] * Vrms), axis=1) >= values['n_channels']

                        out['SNR'][trigger_name] = As_sorted[mask] / Vrms
                        triggered = triggered & mask
                if('ray_solution' in values.keys()):
                    As = np.array(fin['max_amp_ray_solution'])
                    max_amps = np.argmax(As[:, values['ray_channel']], axis=-1)
                    sol = np.array(fin['ray_tracing_solution_type'])
                    mask = np.array([sol[i, values['ray_channel'], max_amps[i]] == values['ray_solution'] for i in range(len(max_amps))], dtype=np.bool)
                    triggered = triggered & mask

                Aeff = proj_area * np.sum(weights[triggered]) / n_events

                if('efficiency' in values.keys()):
                    SNReff, eff = np.loadtxt("analysis_efficiency_{}.csv".format(values['efficiency']), delimiter=",", unpack=True)
                    get_eff = interpolate.interp1d(SNReff, eff, bounds_error=False, fill_value=(0, eff[-1]))
                    As = np.max(np.max(np.nan_to_num(fin['max_amp_ray_solution']), axis=-1)[:, np.append(range(0, 8), range(12, 20))], axis=-1)  # we use the this quantity because it is always computed before noise is added!
                    if('efficiency_scale' in values.keys()):
                        As *= values['efficiency_scale']
                    e = get_eff(As / Vrms)
                    Aeff = proj_area * np.sum((weights * e)[triggered]) / n_events

                out['Aeffs'][trigger_name] = [Aeff, Aeff / np.sum(weights[triggered]) ** 0.5, np.sum(weights[triggered])]
        Aeff_output.append(out)

    return Aeff_output


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


def get_Veff(folder, trigger_combinations={}, station=101, correct_zenith_sampling=False):
    """
    calculates the effective volume from NuRadioMC hdf5 files

    the effective volume is NOT normalized to a water equivalent. It is also NOT multiplied with the solid angle (typically 4pi).

    Parameters
    ----------
    folder: string
        folder conaining the hdf5 files, one per energy
    trigger_combinations: dict, optional
        keys are the names of triggers to calculate. Values are dicts again:
            * 'triggers': list of strings
                name of individual triggers that are combined with an OR
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
    correct_zenith_sampling: bool
        if True, correct a zenith sampling from np.sin(zenith) to an isotropic flux for a cylindrical geometry

    Returns
    ----------
    list of dictionary. Each file is one entry. The dictionary keys store all relevant properties
    """
    trigger_combinations = copy.copy(trigger_combinations)
    Veff_output = []
    trigger_names = None
    trigger_names_dict = {}
    prev_deposited = None
    deposited = False

    if(len(glob.glob(os.path.join(folder, '*.hdf5'))) == 0):
        raise FileNotFoundError(f"couldnt find any hdf5 file in folder {folder}")
    for iF, filename in enumerate(sorted(glob.glob(os.path.join(folder, '*.hdf5')))):
        logger.info(f"reading {filename}")
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
    logger.info("Trigger names:", trigger_names)
    for key in trigger_combinations:
        i = -1
        for value in trigger_combinations[key]['triggers']:
            i += 1
            if value not in trigger_names:
                logger.warning(f"trigger {value} not available, removing this trigger from the trigger combination {key}")
                trigger_combinations[key]['triggers'].pop(i)
                i -= 1

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
                    logger.warning("file {} has no triggering events. Using trigger names from another file".format(filename))
                else:
                    logger.error("file {} has inconsistent trigger names: {}".format(filename, fin.attrs['trigger_names']))
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
        # calculate the average projected area in this zenith angle bin
        Aproj_avg = get_projected_area_cylinder_integral(thetamax, R=rmax, d=dZ) - get_projected_area_cylinder_integral(thetamin, R=rmax, d=dZ)
        # by not dividing by dCosTheta we automatically also integrate the solid angle into the weight.
        # Hence, simulations with zenith slices of different zenith angle coverage are possible.
        # Aproj_avg /= np.cos(thetamin) - np.cos(thetamax)
        out['Aproj'] = Aproj_avg
        Vrms = fin.attrs['Vrms']

        if(correct_zenith_sampling):
            if(len(weights) > 0):

                def get_weights(zeniths, thetamin, thetamax, R, d):
                    """
                    calculates a correction to the weight to go from a zenith distribution proportional from
                    theta ~ sin(theta) to an isotropic flux, i.e., the same number of events for the same 
                    projected area perpendicular to the incoming direction.  
                    
                    """
                    zeniths = np.array(zeniths)
                    yy = get_projected_area_cylinder(zeniths, R, d)
                    # calculate the average value of Aproj within the zenith band -> int(Aproc(theta) dcostheta)/int(1, dcostheta)
                    norm = get_projected_area_cylinder_integral(thetamax, R, d) - get_projected_area_cylinder_integral(thetamin, R, d)  # int(Aproc(theta) dcostheta)
                    norm /= (np.cos(thetamin) - np.cos(thetamax))  # int(1, dcostheta)
                    weights = yy / norm
                    logger.debug(f"{thetamin/units.deg:.0f} - {thetamax/units.deg:.0f}: average correction factor {weights.mean():.2f} max = {weights.max():.2f} min = {weights.min():.2f}")
                    return weights

                weights *= get_weights(fin['zeniths'], thetamin, thetamax, rmax, dZ)

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
                            As = np.max(np.nan_to_num(fin['max_amp_ray_solution']), axis=-1)  # we use the this quantity because it is always computed before noise is added!
                            As_sorted = np.sort(As[:, values['channels'][iS]], axis=1)
                            # the smallest of the three largest amplitudes
                            max_amplitude = As_sorted[:, -values['n_channels'][iS]]
                            mask = np.sum(As[:, values['channels'][iS]] >= (values['min_sigma'][iS] * Vrms), axis=1) >= values['n_channels'][iS]
                            masks = masks | mask
                            out['SNR'][trigger_name][iS] = max_amplitude[mask] / Vrms
                        triggered = triggered & masks
                    else:
                        As = np.max(np.nan_to_num(fin['max_amp_ray_solution']), axis=-1)  # we use the this quantity because it is always computed before noise is added!

                        As_sorted = np.sort(As[:, values['channels']], axis=1)
                        max_amplitude = As_sorted[:, -values['n_channels']]  # the smallest of the three largest amplitudes

                        mask = np.sum(As[:, values['channels']] >= (values['min_sigma'] * Vrms), axis=1) >= values['n_channels']
                        out['SNR'][trigger_name] = As_sorted[mask] / Vrms
                        triggered = triggered & mask
                if('ray_solution' in values.keys()):
                    As = np.array(fin['max_amp_ray_solution'])
                    max_amps = np.argmax(As[:, values['ray_channel']], axis=-1)
                    sol = np.array(fin['ray_tracing_solution_type'])
                    mask = np.array([sol[i, values['ray_channel'], max_amps[i]] == values['ray_solution'] for i in range(len(max_amps))], dtype=np.bool)
                    triggered = triggered & mask

                if('n_reflections' in values.keys()):
                    if(np.sum(triggered)):
                        As = np.array(fin[f'station_{station:d}/max_amp_ray_solution'])
                        # find the ray tracing solution that produces the largest amplitude
                        max_amps = np.argmax(np.argmax(As[:, :], axis=-1), axis=-1)
                        # advanced indexing: selects the ray tracing solution per event with the highest amplitude
                        triggered = triggered & (np.array(fin[f'station_{station:d}/ray_tracing_reflection'])[..., max_amps, 0][:, 0] == values['n_reflections'])

                Veff = V * np.sum(weights[triggered]) / n_events

                if('efficiency' in values.keys()):
                    SNReff, eff = np.loadtxt("analysis_efficiency_{}.csv".format(values['efficiency']), delimiter=",", unpack=True)
                    get_eff = interpolate.interp1d(SNReff, eff, bounds_error=False, fill_value=(0, eff[-1]))
                    As = np.max(np.max(np.nan_to_num(fin['max_amp_ray_solution']), axis=-1)[:, np.append(range(0, 8), range(12, 20))], axis=-1)  # we use the this quantity because it is always computed before noise is added!
                    if('efficiency_scale' in values.keys()):
                        As *= values['efficiency_scale']
                    e = get_eff(As / Vrms)
                    Veff = V * np.sum((weights * e)[triggered]) / n_events

                Vefferror = 0
                if(np.sum(weights[triggered]) > 0):
                    Vefferror = Veff / np.sum(weights[triggered]) ** 0.5
                out['Veffs'][trigger_name] = [Veff, Vefferror, np.sum(weights[triggered])]
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
     * array of weights for zenith averaging


    Examples
    ---------

    To plot the full sky effective volume for 'all_triggers' do

    ```
    output, uenergies, uzenith_bins, utrigger_names, zenith_weights = get_Veff_array(data)


    fig, ax = plt.subplots(1, 1)
    tname = "all_triggers"
    Veff = np.average(output[:,:,get_index(tname, utrigger_names),0], axis=1, weights=zenith_weights)
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
    weights = np.ones((len(uenergies), len(uzenith_bins)))
    logger.debug(f"unique energies {uenergies}")
    logger.debug(f"unique zenith angle bins {uzenith_bins/units.deg}")
    logger.debug(f"unique energies {utrigger_names}")

    for d in data:
        iE = np.squeeze(np.argwhere(d['energy'] == uenergies))
        iT = np.squeeze(np.argwhere([d['thetamin'], d['thetamax']] == uzenith_bins))[0][0]
        for triggername, Veff in d['Veffs'].items():
            iTrig = np.squeeze(np.argwhere(triggername == utrigger_names))
            output[iE, iT, iTrig] = Veff

    for d in data:
        iE = np.squeeze(np.argwhere(d['energy'] == uenergies))
        iT = np.squeeze(np.argwhere([d['thetamin'], d['thetamax']] == uzenith_bins))[0][0]
        if('Aproj' in d):
            weights[iE, iT] = d['Aproj']
    for iE in range(len(uenergies)):
        weights[iE] /= np.sum(weights[iE])

    return output, uenergies, uzenith_bins, utrigger_names, weights


def get_Aeff_array(data):
    """
    calculates a multi dimensional array of effective area calculations for fast slicing

    the array dimensions are (energy, zenith bin, triggername, 3) where the
    last tuple is the effective area, its uncertainty and the weighted sum of triggered events

    Parameters
    -----------
    data: dict
        the result of the `get_Aeff` function

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
    output, uenergies, uzenith_bins, utrigger_names = get_Aeff_array(data)


    fig, ax = plt.subplots(1, 1)
    tname = "all_triggers"
    Aeff = np.average(output[:,:,get_index(tname, utrigger_names),0], axis=1)
    Aefferror = Aeff / np.sum(output[:,:,get_index(tname, utrigger_names),2], axis=1)**0.5
    ax.errorbar(uenergies/units.eV, Aeff/units.km**3 * 4 * np.pi, yerr=Aefferror/units.km**3 * 4 * np.pi, fmt='-o', label=tname)

    ax.legend()
    ax.semilogy(True)
    ax.semilogx(True)
    fig.tight_layout()
    plt.show()
    ```


    To plot the effective area for different declination bands do

    ```
    fig, ax = plt.subplots(1, 1)
    tname = "LPDA_2of4_100Hz"
    iZ = 9
    Aeff = output[:,iZ,get_index(tname, utrigger_names)]
    ax.errorbar(uenergies/units.eV, Aeff[:,0]/units.km**2, yerr=Aeff[:,1]/units.km**2,
                label=f"zenith bin {uzenith_bins[iZ][0]/units.deg:.0f} - {uzenith_bins[iZ][1]/units.deg:.0f}")

    iZ = 8
    Aeff = output[:,iZ,get_index(tname, utrigger_names)]
    ax.errorbar(uenergies/units.eV, Aeff[:,0]/units.km**2, yerr=Aeff[:,1]/units.km**2,
                label=f"zenith bin {uzenith_bins[iZ][0]/units.deg:.0f} - {uzenith_bins[iZ][1]/units.deg:.0f}")
    iZ = 7
    Aeff = output[:,iZ,get_index(tname, utrigger_names)]
    ax.errorbar(uenergies/units.eV, Aeff[:,0]/units.km**2, yerr=Aeff[:,1]/units.km**2,
                label=f"zenith bin {uzenith_bins[iZ][0]/units.deg:.0f} - {uzenith_bins[iZ][1]/units.deg:.0f}")
    iZ = 10
    Aeff = output[:,iZ,get_index(tname, utrigger_names)]
    ax.errorbar(uenergies/units.eV, Aeff[:,0]/units.km**2, yerr=Aeff[:,1]/units.km**2,
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
        for triggername in d['Aeffs']:
            trigger_names.append(triggername)

    energies = np.array(energies)
    zenith_bins = np.array(zenith_bins)
    trigger_names = np.array(trigger_names)
    uenergies = np.unique(energies)
    uzenith_bins = np.unique(zenith_bins, axis=0)
    utrigger_names = np.unique(trigger_names)
    output = np.zeros((len(uenergies), len(uzenith_bins), len(utrigger_names), 3))
    logger.debug(f"unique energies {uenergies}")
    logger.debug(f"unique zenith angle bins {uzenith_bins/units.deg}")
    logger.debug(f"unique energies {utrigger_names}")

    for d in data:
        iE = np.squeeze(np.argwhere(d['energy'] == uenergies))
        iT = np.squeeze(np.argwhere([d['thetamin'], d['thetamax']] == uzenith_bins))[0][0]
        for triggername, Aeff in d['Aeffs'].items():
            iTrig = np.squeeze(np.argwhere(triggername == utrigger_names))
            output[iE, iT, iTrig] = Aeff

    return output, uenergies, uzenith_bins, utrigger_names


def get_index(value, array):
    return np.squeeze(np.argwhere(value == array))


def export(filename, data, trigger_names=None, export_format='yaml'):
    """
    export effective volumes (or effective areas) into a human readable JSON or YAML file

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
            if (key not in  ['Veffs', 'Aeffs']):
                if isinstance(data[i][key], np.generic):
                    tmp[key] = data[i][key].item()
                else:
                    tmp[key] = data[i][key]
        for key in ["Veffs", "Aeffs"]:
            if(key in data[i]):
                tmp[key] = {}
                for trigger_name in data[i][key]:
                    if(trigger_names is None or trigger_name in trigger_names):
                        logger.info(trigger_name)
                        tmp[key][trigger_name] = []
                        for value in data[i][key][trigger_name]:
                            tmp[key][trigger_name].append(float(value))
        output.append(tmp)

    with open(filename, 'w') as fout:
        if(export_format == 'yaml'):
            import yaml
            yaml.dump(output, fout)
        elif(export_format == 'json'):
            json.dump(output, fout, sort_keys=True, indent=4)
