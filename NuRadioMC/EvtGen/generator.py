# -*- coding: utf-8 -*-
import logging

STATUS = 31


class NuRadioMCLogger(logging.Logger):

    def status(self, msg, *args, **kwargs):
        if self.isEnabledFor(STATUS):
            self._log(STATUS, msg, args, **kwargs)


logging.setLoggerClass(NuRadioMCLogger)
logging.addLevelName(STATUS, 'STATUS')
logger = logging.getLogger("NuRadioMC-EvtGen")
assert isinstance(logger, NuRadioMCLogger)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
logger.setLevel(logging.INFO)

import numpy as np
import NuRadioMC
from NuRadioReco.utilities import units
from NuRadioMC.utilities import inelasticities
from NuRadioReco.utilities import version
from six import iterkeys, iteritems
from scipy import constants
from scipy.integrate import quad
from scipy.interpolate import interp1d
import scipy.interpolate as interpolate
from scipy.optimize import fsolve
from scipy.interpolate import RectBivariateSpline
import h5py
import time
from NuRadioMC.simulation.simulation import pretty_time_delta
import os
import math
from numpy.random import Generator, Philox
import copy

VERSION_MAJOR = 2
VERSION_MINOR = 2

HEADER = """
# all quantities are in the default NuRadioMC units (i.e., meters, radians and eV)
# all geometry quantities are in the NuRadioMC default local coordinate system:
#     coordinate origin is at the surface
#     x axis is towards Easting, y axis towards Northing, z axis upwards
#     zenith/theta angle is defined with respect to z axis, i.e. 0deg = upwards, 90deg = towards horizon, 180deg = downwards
#     azimuth/phi angle counting northwards from East
#
# the collumns are defined as follows
# 1. event id (integer)
# 2. neutrino flavor (integer) encoded as using PDG numbering scheme, particles have positive sign, anti-particles have negative sign, relevant for us are:
#       12: electron neutrino
#       14: muon neutrino
#       16: tau neutrino
# 3. energy of neutrino (double)
# 4. charge or neutral current interaction (string, one of ['cc', 'nc']
# 5./6./7. position of neutrino interaction vertex in cartesian coordinates (x, y, z) (in default NuRadioMC local coordinate system)
# 8. zenith/theta angle of neutrino direction (pointing to where it came from, i.e. opposite to the direction of propagation)
# 9. azimuth/phi angle of neutrino direction (pointing to where it came from, i.e. opposite to the direction of propagation)
# 10. inelasticity (the fraction of neutrino energy that goes into the hadronic part)
#
"""
# Mass energy equivalent of the tau lepton
tau_mass = constants.physical_constants['tau mass energy equivalent in MeV'][0] * units.MeV
# Lifetime of the tau (rest frame). Taken from PDG
tau_rest_lifetime = 290.3 * units.fs
density_ice = 0.9167 * units.g / units.cm ** 3
cspeed = constants.c * units.m / units.s


def load_input_hdf5(filename):
    """
    reads input file into memory

    Parameters
    ----------
    filename: string
        Name of the file

    Returns
    -------
    fin: dictionary
        Dictionary containing the elements in filename
    """
    h5fin = h5py.File(filename, 'r')
    fin = {}
    for key, value in iteritems(h5fin):
        fin[key] = np.array(value)
    h5fin.close()
    return fin


def get_tau_95_length(energies):
    """
    Returns a fit to the 95% percentile of the tau track length calculated
    with PROPOSAL. We calculate the 95% percentile for the largest energy.
    """

    coeffs = [6.80016451e+02, -1.61902120e+02, 1.42383021e+01, -5.47388025e-01, 7.79239697e-03]
    log_length = 0

    log_energy_eV = np.log10(np.max(energies) / units.eV)

    for ipow, coeff in enumerate(coeffs):
        log_length += coeff * (log_energy_eV) ** ipow

    return 10 ** log_length * units.m


def write_events_to_hdf5(filename, data_sets, attributes, n_events_per_file=None,
                         start_file_id=0):
    """
    writes NuRadioMC input parameters to hdf5 file

    this function can automatically split the dataset up into multiple files for easy multiprocessing

    Parameters
    ----------
    filename: string
        the desired output filename (if multiple files are generated, a 'part000x' is appended to the filename
    data_sets: dict
        a dictionary with the data sets
    attributes: dict
        a dictionary containing the meta attributes
    n_events_per_file: int (optional, default None)
        the number of events per file
    additional_interactions: dict or None (default)
        a dictionary containing potential additional interactions, such as the second tau interaction vertex.
    """

    n_events = attributes['n_events']
    logger.info("saving {} events in total".format(n_events))
    total_number_of_events = attributes['n_events']

    if "start_event_id" not in attributes:
        attributes["start_event_id"] = 0  # backward compatibility

    if(n_events_per_file is None):
        n_events_per_file = n_events
    else:
        n_events_per_file = int(n_events_per_file)
    iFile = -1
    evt_id_first = data_sets["event_group_ids"][0]
    evt_id_last_previous = 0  # save the last event id of the previous file
    start_index = 0
    n_events_total = 0
    while True:
        iFile += 1
        filename2 = filename
        evt_ids_this_file = np.unique(data_sets["event_group_ids"])[iFile * n_events_per_file : (iFile + 1) * n_events_per_file]
        if(len(evt_ids_this_file) == 0):
            logger.info("no more events to write in file {}".format(iFile))
            break

        if((iFile > 0) or (n_events_per_file < n_events)):
            filename2 = filename + ".part{:04}".format(iFile + start_file_id)
        fout = h5py.File(filename2, 'w')
        fout.attrs['VERSION_MAJOR'] = VERSION_MAJOR
        fout.attrs['VERSION_MINOR'] = VERSION_MINOR
        fout.attrs['header'] = HEADER
        for key, value in attributes.items():
            fout.attrs[key] = value
        fout.attrs['total_number_of_events'] = total_number_of_events

        evt_id_first = evt_ids_this_file[0]
        evt_id_last = evt_ids_this_file[-1]

        tmp = np.squeeze(np.argwhere(data_sets["event_group_ids"] == evt_id_last))  # set stop index such that last event is competely in file
        if(tmp.size == 1):
            stop_index = tmp + 1
        else:
            stop_index = tmp[-1] + 1
#         if(evt_id_last >= n_events):
#             evt_id_last = n_events
#             stop_index = len(data_sets["event_group_ids"])
#         else:
#             tmp = np.squeeze(np.argwhere(data_sets["event_group_ids"] > evt_id_last))  # set stop index such that last event is competely in file
#             if(tmp.size == 1):
#                 stop_index = tmp
#             else:
#                 stop_index = tmp[0]

        for key in data_sets:
            data_sets[key] = np.array(data_sets[key])
        for key, value in data_sets.items():
            if value.dtype.kind == 'U':
                fout[key] = np.array(value, dtype=h5py.string_dtype(encoding='utf-8'))[start_index:stop_index]
            else:
                fout[key] = value[start_index:stop_index]

        # determine the number of events in this file (which is NOT the same as the entries in the file)
        # case 1) this is not the last file -> number of events is difference between last event id of the current and previous file + 1
        # case 2) it is the last file -> total number of simulated events - last event id of previous file
        # case 3) it is the first file -> last event id + 1 - start_event_id
        # case 4) it is the first and last file -> total number of simulated events
        evt_ids_next_file = np.unique(data_sets["event_group_ids"])[(iFile + 1) * n_events_per_file : (iFile + 2) * n_events_per_file]
        n_events_this_file = None
        if(iFile == 0 and len(evt_ids_next_file) == 0):  # case 4
            n_events_this_file = total_number_of_events
        elif(len(evt_ids_next_file) == 0):  # last file -> case 2
            n_events_this_file = total_number_of_events - (evt_id_last_previous + 1) + attributes['start_event_id']
        elif(iFile == 0):  # case 3
            n_events_this_file = evt_id_last - attributes['start_event_id'] + 1
        else:  # case 1
            n_events_this_file = evt_id_last - evt_id_last_previous

        logger.status('writing file {} with {} events (id {} - {}) and {} entries'.format(filename2, n_events_this_file, evt_id_first,
                                                                                  evt_id_last, stop_index - start_index))
        fout.attrs['n_events'] = n_events_this_file
        fout.close()
        n_events_total += n_events_this_file

        start_index = stop_index

        evt_id_last_previous = evt_id_last
        if(evt_id_last == n_events):  # break while loop if all events are saved
            break
    logger.info("wrote {} events in total".format(n_events_total))


def primary_energy_from_deposited(Edep, ccnc, flavor, inelasticity):
    """
    Calculates the primary energy of the neutrino from the deposited
    energy in the medium.

    Parameters
    ----------
    Edep: float
        deposited energy
    ccnc: string
        indicates 'nc', neutral current; 'cc', charged current
    flavor: int
        neutrino flavor
    inelasticity: float
        inelasticity of the interaction
    """

    if (ccnc == 'nc'):
        return Edep / inelasticity
    elif (ccnc == 'cc'):
        if (np.abs(flavor) == 12):
            return Edep
        elif (np.abs(flavor) == 14):
            return Edep / inelasticity
        elif (np.abs(flavor) == 16):
            return Edep / inelasticity  # TODO: change this for taus


def ice_cube_nu_fit(energy, slope=-2.19, offset=1.01):
    # from https://doi.org/10.22323/1.301.1005
    # ApJ slope=-2.13, offset=0.9
    flux = 3 * offset * (energy / (100 * units.TeV)) ** slope * 1e-18 * \
        (units.GeV ** -1 * units.cm ** -2 * units.second ** -1 * units.sr ** -1)
    return flux


def get_GZK_1(energy):
    """
    model of (van Vliet et al., 2019, https://arxiv.org/abs/1901.01899v1) of the cosmogenic neutrino ﬂux
    for a source evolution parameter of m = 3.4,
    a spectral index of the injection spectrum of α = 2.5, a cut-oﬀ rigidity of R = 100 EeV,
    and a proton fraction of 10% at E = 10^19.6 eV
    """
    E, J = np.loadtxt(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                     '../examples/Sensitivities/ReasonableNeutrinos1.txt'))
    E *= units.GeV
    J *= units.GeV * units.cm ** -2 * units.s ** -1 * units.sr ** -1 / E ** 2
    get_flux = interpolate.interp1d(E, J, fill_value=0, bounds_error=False)
    return get_flux(energy)


def get_energy_from_flux(Emin, Emax, n_events, flux, rnd=None):
    """
    returns randomly distribution of energy according to a flux

    Parameters
    ----------
    Emin: float
        minumum energy
    Emax: float
        maximum energy
    n_event: int
        number of events to generate
    flux: function
        must return flux as function of energy in units of events per energy, time, solid angle and area
    rnd: random generator object
        if None is provided, a new default random generator object is initialized

    Returns: array of energies
    """
    if(rnd is None):
        rnd = np.random.default_rng()
    xx_edges = np.linspace(Emin, Emax, 10000000)
    xx = 0.5 * (xx_edges[1:] + xx_edges[:-1])
    yy = flux(xx)
    cum_values = np.zeros(xx_edges.shape)
    cum_values[1:] = np.cumsum(yy * np.diff(xx_edges))
    inv_cdf = interpolate.interp1d(cum_values, xx_edges)
    r = rnd.uniform(0, cum_values.max(), n_events)
    return inv_cdf(r)


def get_product_position_time(data_sets, product, iE):

    """
    Calculates the position of a product particle given by the NuRadioProposal
    module that has been created by a PROPOSAL lepton propagation.

    Parameters
    ----------
    data_sets: dictionary
        Dictionary with the data sets from the generating functions
        (generate_eventlist_cylinder and generate_surface_muons)
    product: secondary_properties class from NuRadioPROPOSAL
        Contains the properties of the shower-inducing particle given by PROPOSAL
    iE: int
        Number of the event in data_sets corresponding to the product particle

    Returns
    -------
    x, y, z, time: tuple
        The 3-D position of the shower-inducing product particle and the time
        elapsed since the first interaction to the present interaction
    """

    dist = product.distance
    prop_time = dist / cspeed
    x = data_sets["xx"][iE] - dist * np.sin(data_sets["zeniths"][iE]) * np.cos(data_sets["azimuths"][iE])
    y = data_sets["yy"][iE] - dist * np.sin(data_sets["zeniths"][iE]) * np.sin(data_sets["azimuths"][iE])
    z = data_sets["zz"][iE] - dist * np.cos(data_sets["zeniths"][iE])

    return x, y, z, prop_time


def get_energies(n_events, Emin, Emax, spectrum_type, rnd=None):
    """
    generates a random distribution of enrgies following a certain spectrum
    
    Parameters
    -----------
    n_events: int
        the total number of events
    Emin: float
        the minimal energy
    Emax: float
        the maximum energy
    spectrum_type: string
        defines the probability distribution for which the neutrino energies are generated
        
        * 'log_uniform': uniformly distributed in the logarithm of energy
        * 'E-?': E to the -? spectrum where ? can be any float
        * 'IceCube-nu-2017': astrophysical neutrino flux measured with IceCube muon sample (https://doi.org/10.22323/1.301.1005)
        * 'GZK-1': GZK neutrino flux model from van Vliet et al., 2019, https://arxiv.org/abs/1901.01899v1 for
            10% proton fraction (see get_GZK_1 function for details)
        * 'GZK-1+IceCube-nu-2017': a combination of the cosmogenic (GZK-1) and astrophysical (IceCube nu 2017) flux
        
    rnd: random generator object
        if None is provided, a new default random generator object is initialized
    """
    if(rnd is None):
        rnd = np.random.default_rng()
    logger.debug("generating energies")
    if(spectrum_type == 'log_uniform'):
        energies = 10 ** rnd.uniform(np.log10(Emin), np.log10(Emax), n_events)
    elif(spectrum_type.startswith("E-")):  # enerate an E^gamma spectrum.
        gamma = float(spectrum_type[1:])
        gamma += 1
        Nmin = (Emin) ** gamma
        Nmax = (Emax) ** gamma

        def get_inverse_spectrum(N, gamma):
            return np.exp(np.log(N) / gamma)

        energies = get_inverse_spectrum(rnd.uniform(Nmax, Nmin, size=n_events), gamma)
    elif(spectrum_type == "GZK-1"):
        """
        model of (van Vliet et al., 2019, https://arxiv.org/abs/1901.01899v1) of the cosmogenic neutrino ﬂux
        for a source evolution parameter of m = 3.4,
        a spectral index of the injection spectrum of α = 2.5, a cut-oﬀ rigidity of R = 100 EeV,
        and a proton fraction of 10% at E = 10^19.6 eV
        """
        energies = get_energy_from_flux(Emin, Emax, n_events, get_GZK_1, rnd)
    elif(spectrum_type == "IceCube-nu-2017"):
        energies = get_energy_from_flux(Emin, Emax, n_events, ice_cube_nu_fit, rnd)
    elif(spectrum_type == "GZK-1+IceCube-nu-2017"):

        def J(E):
            return ice_cube_nu_fit(E) + get_GZK_1(E)

        energies = get_energy_from_flux(Emin, Emax, n_events, J, rnd)
    else:
        logger.error("spectrum {} not implemented".format(spectrum_type))
        raise NotImplementedError("spectrum {} not implemented".format(spectrum_type))
    return energies


def set_volume_attributes(volume, proposal, attributes):
    """
    helper function that interprets the volume settings and sets the relevant quantities as hdf5 attributes.
    
    Parameters
    ----------
    volume: dictionarty
        dict specifying the volume
    proposal: bool
        specifies if secondary interaction via proposal are calculated
    attributes: dictionary
        dict storing hdf5 attributes
    """
    n_events = attributes['n_events']
    if("fiducial_rmax" in volume):  # user specifies a cylinder
        if('fiducial_rmin' in volume):
            attributes['fiducial_rmin'] = volume['fiducial_rmin']
        else:
            attributes['fiducial_rmin'] = 0
        attributes['fiducial_rmax'] = volume['fiducial_rmax']
        attributes['fiducial_zmin'] = volume['fiducial_zmin']
        attributes['fiducial_zmax'] = volume['fiducial_zmax']

        rmin = attributes['fiducial_rmin']
        rmax = attributes['fiducial_rmax']
        zmin = attributes['fiducial_zmin']
        zmax = attributes['fiducial_zmax']
        volume_fiducial = np.pi * (rmax ** 2 - rmin ** 2) * (zmax - zmin)

        if('full_rmax' in volume):
            rmax = volume['full_rmax']
        if('full_rmin' in volume):
            rmin = volume['full_rmin']
        if('full_zmax' in volume):
            zmax = volume['full_zmax']
        if('full_zmin' in volume):
            zmin = volume['full_zmin']

        # We increase the radius of the cylinder according to the tau track length
        if(proposal):
            logger.info("simulation of second interactions via PROPOSAL activated")
            if("full_rmin" in volume):
                rmin = volume['full_rmin']
            else:
                rmin = attributes['fiducial_rmin'] / 3.
            if('full_rmax' in volume):
                rmax = volume['full_rmax']
            else:
                tau_95_length = get_tau_95_length(attributes['Emax'])
                # check if thetamax/thetamin are in same quadrant
                is_same_quadrant = np.sign(np.cos(attributes['thetamin']))==np.sign(np.cos(attributes['thetamax']))
                if is_same_quadrant:
                    max_horizontal_dist = np.abs(max(np.abs(np.tan(attributes['thetamin'])), np.abs(np.tan(attributes['thetamax']))) * volume['fiducial_zmin'])  # calculates the maximum horizontal distance through the ice
                else:
                    # not same quadrant: theta range surpasses pi/2 -> horizontal trajectories
                    # geometric trajectory is infinite, but not acutally used. distance is limited by the min() to allowed length below
                    max_horizontal_dist = np.inf
                d_extent = min(tau_95_length, max_horizontal_dist)
                logger.info(f"95% quantile of tau decay length is {tau_95_length/units.km:.1f}km. Maximum horizontal distance for zenith range of {attributes['thetamin']/units.deg:.0f} - {attributes['thetamax']/units.deg:.0f}deg and depth of {volume['fiducial_zmin']/units.km:.1f}km is {max_horizontal_dist/units.km:.1f}km -> extending horizontal volume by {d_extent/units.km:.1f}km")

                rmax = d_extent + attributes['fiducial_rmax']
            if('full_zmax' in volume):
                zmax = volume['full_zmax']
            else:
                zmax = attributes['fiducial_zmax'] / 3.
            if('full_zmin' in volume):
                zmin = volume['full_zmin']
            else:
                zmin = attributes['fiducial_zmin']
        volume_full = np.pi * (rmax ** 2 - rmin ** 2) * (zmax - zmin)
        # increase the total number of events such that we end up with the same number of events in the fiducial volume
        n_events = int(n_events * volume_full / volume_fiducial)
        logger.info(f"increasing rmax from {attributes['fiducial_rmax']/units.km:.01f}km to {rmax/units.km:.01f}km, zmax from {attributes['fiducial_zmax']/units.km:.01f}km to {zmax/units.km:.01f}km")
        logger.info(f"decreasing rmin from {attributes['fiducial_rmin']/units.km:.01f}km to {rmin/units.km:.01f}km")
        logger.info(f"decreasing zmin from {attributes['fiducial_zmin']/units.km:.01f}km to {zmin/units.km:.01f}km")
        logger.info(f"increasing number of events to {n_events:.6g}")
        attributes['n_events'] = n_events

        attributes['rmin'] = rmin
        attributes['rmax'] = rmax
        attributes['zmin'] = zmin
        attributes['zmax'] = zmax

        V = np.pi * (rmax ** 2 - rmin ** 2) * (zmax - zmin)
        attributes['volume'] = V  # save full simulation volume to simplify effective volume calculation
        attributes['area'] = np.pi * (rmax ** 2 - rmin ** 2)
    elif("fiducial_xmax" in volume):  # user specifies a cube
        attributes['fiducial_xmax'] = volume['fiducial_xmax']
        attributes['fiducial_xmin'] = volume['fiducial_xmin']
        attributes['fiducial_ymax'] = volume['fiducial_ymax']
        attributes['fiducial_ymin'] = volume['fiducial_ymin']
        attributes['fiducial_zmin'] = volume['fiducial_zmin']
        attributes['fiducial_zmax'] = volume['fiducial_zmax']

        xmin = attributes['fiducial_xmin']
        xmax = attributes['fiducial_xmax']
        ymin = attributes['fiducial_ymin']
        ymax = attributes['fiducial_ymax']
        zmin = attributes['fiducial_zmin']
        zmax = attributes['fiducial_zmax']
        volume_fiducial = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
        if('full_xmax' in volume):
            xmin = volume['full_xmin']
            xmax = volume['full_xmax']
            ymin = volume['full_ymin']
            ymax = volume['full_ymax']
            zmin = volume['full_zmin']
            zmax = volume['full_zmax']

        # We increase the simulation volume according to the tau track length
        if(proposal):
            logger.info("simulation of second interactions via PROPOSAL activated")
            if('full_xmax' not in volume):  # assuming that also full_xmin, full_ymin, full_ymax are not set.
                # extent fiducial by tau decay length
                tau_95_length = get_tau_95_length(attributes['Emax'])
                # check if thetamax/thetamin are in same quadrant
                is_same_quadrant = np.sign(np.cos(attributes['thetamin']))==np.sign(np.cos(attributes['thetamax']))
                if is_same_quadrant:
                    max_horizontal_dist = np.abs(max(np.abs(np.tan(attributes['thetamin'])), np.abs(np.tan(attributes['thetamax']))) * volume['fiducial_zmin'])  # calculates the maximum horizontal distance through the ice
                else:
                    # not same quadrant: theta range surpasses pi/2 -> horizontal trajectories
                    # geometric trajectory is infinite, but not acutally used. distance is limited by the min() to allowed length below
                    max_horizontal_dist = np.inf
                d_extent = min(tau_95_length, max_horizontal_dist)
                logger.info(f"95% quantile of tau decay length is {tau_95_length/units.km:.1f}km. Maximum horizontal distance for zenith range of {attributes['thetamin']/units.deg:.0f} - {attributes['thetamax']/units.deg:.0f}deg and depth of {volume['fiducial_zmin']/units.km:.1f}km is {max_horizontal_dist/units.km:.1f}km -> extending horizontal volume by {d_extent/units.km:.1f}km")
                xmax += d_extent
                xmin -= d_extent
                ymax += d_extent
                ymin -= d_extent
        logger.info(f"increasing xmax from {attributes['fiducial_xmax']/units.km:.01f}km to {xmax/units.km:.01f}km, ymax from {attributes['fiducial_ymax']/units.km:.01f}km to {ymax/units.km:.01f}km, zmax from {attributes['fiducial_zmax']/units.km:.01f}km to {zmax/units.km:.01f}km")
        logger.info(f"decreasing xmin from {attributes['fiducial_xmin']/units.km:.01f}km to {xmin/units.km:.01f}km, ymin from {attributes['fiducial_ymin']/units.km:.01f}km to {ymin/units.km:.01f}km")
        logger.info(f"decreasing zmin from {attributes['fiducial_zmin']/units.km:.01f}km to {zmin/units.km:.01f}km")

        volume_full = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
        n_events = int(n_events * volume_full / volume_fiducial)
        logger.info(f"increasing number of events from {attributes['n_events']} to {n_events:.6g}")
        attributes['n_events'] = n_events

        attributes['xmin'] = xmin
        attributes['xmax'] = xmax
        attributes['ymin'] = ymin
        attributes['ymax'] = ymax
        attributes['zmin'] = zmin
        attributes['zmax'] = zmax

        V = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
        attributes['volume'] = V  # save full simulation volume to simplify effective volume calculation
        attributes['area'] = (xmax - xmin) * (ymax - ymin)
    else:
        raise AttributeError(f"'fiducial_rmin' or 'fiducial_rmax' is not part of 'volume'")


def generate_vertex_positions(attributes, n_events, rnd=None):
    """
    helper function that generates the vertex position randomly distributed in simulation volume. 
    The relevant quantities are also saved into the hdf5 attributes
    
    Parameters
    ----------
    attributes: dicitionary
        dict storing hdf5 attributes
    rnd: random generator object
        if None is provided, a new default random generator object is initialized
    """
    if(rnd is None):
        rnd = np.random.default_rng()
    if("fiducial_rmax" in attributes):  # user specifies a cylinder
        rr_full = rnd.uniform(attributes['rmin'] ** 2, attributes['rmax'] ** 2, n_events) ** 0.5
        phiphi = rnd.uniform(0, 2 * np.pi, n_events)
        xx = rr_full * np.cos(phiphi)
        yy = rr_full * np.sin(phiphi)
        zz = rnd.uniform(attributes['zmin'], attributes['zmax'], n_events)
        return xx, yy, zz
    elif("fiducial_xmax" in attributes):  # user specifies a cube
        xx = rnd.uniform(attributes['xmin'], attributes['xmax'], n_events)
        yy = rnd.uniform(attributes['ymin'], attributes['ymax'], n_events)
        zz = rnd.uniform(attributes['zmin'], attributes['zmax'], n_events)
        return xx, yy, zz
    else:
        raise AttributeError(f"'fiducial_rmin' or 'fiducial_rmax' is not part of 'attributes'")


def intersection_box_ray(bounds, ray):
    """
    this function calculates the intersection between a ray and an axis-aligned box
    code adapted from https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
    
    Parameters
    ----------
    box: array with shape (2,3)
        definition of box with two points
    ray: array with shape (2,3)
        definiton of ray using origin and direction 3-dim vectors
    """
    orig = np.array(ray[0])
    direction = np.array(ray[1])
    invdir = 1 / direction
    sign = np.zeros(3, dtype=int)
    sign[0] = (invdir[0] < 0)
    sign[1] = (invdir[1] < 0)
    sign[2] = (invdir[2] < 0)

    tmin = (bounds[sign[0]][0] - orig[0]) * invdir[0]
    tmax = (bounds[1 - sign[0]][0] - orig[0]) * invdir[0]
    tymin = (bounds[sign[1]][1] - orig[1]) * invdir[1]
    tymax = (bounds[1 - sign[1]][1] - orig[1]) * invdir[1]

    if ((tmin > tymax) or (tymin > tmax)):
        return False
    if (tymin > tmin):
        tmin = tymin
    if (tymax < tmax):
        tmax = tymax

    tzmin = (bounds[sign[2]][2] - orig[2]) * invdir[2]
    tzmax = (bounds[1 - sign[2]][2] - orig[2]) * invdir[2]

    if ((tmin > tzmax) or (tzmin > tmax)):
        return False
    if (tzmin > tmin):
        tmin = tzmin
    if (tzmax < tmax):
        tmax = tzmax
    t = tmin
    if (t < 0):
        t = tmax
        if (t < 0):
            return False  # I think this removes events where the box is behind the the neutrino interaction which is what we want
    return True


def get_intersection_volume_neutrino(attributes, vertex, direction):
    if('xmax' in attributes):  # cube volume
        bounds = np.array([[attributes['fiducial_xmin'], attributes['fiducial_ymin'], attributes['fiducial_zmin']],
                           [attributes['fiducial_xmax'], attributes['fiducial_ymax'], attributes['fiducial_zmax']]])
        ray = np.array([vertex, direction])
        cut = intersection_box_ray(bounds, ray)
#         if(cut == False):
#             print(f"rejecting event with vertex {vertex[0]:.0f}, {vertex[1]:.0f}, {vertex[2]:.0f} and direction {direction[0]:.1f}, {direction[1]:.1f}, {direction[2]:.1f}")
        return cut

    else:  # cylinder volume, not yet implemented
        return True


def is_in_fiducial_volume(attributes, X):
    r = (X[0] ** 2 + X[1] ** 2) ** 0.5
    if('fiducial_rmin' in attributes):
        if(r >= attributes['fiducial_rmin'] and r <= attributes['fiducial_rmax'] and X[2] >= attributes['fiducial_zmin'] and X[2] <= attributes['fiducial_zmax']):
            return True
        else:
            return False
    elif('fiducial_xmax' in attributes):
        low = np.array([attributes['fiducial_xmin'], attributes['fiducial_ymin'], attributes['fiducial_zmin']])
        up = np.array([attributes['fiducial_xmax'], attributes['fiducial_ymax'], attributes['fiducial_zmax']])
        return np.all(np.logical_and(low <= X, X <= up))
    else:
        raise AttributeError("neither 'fiducial_rmin' nor 'fiducial_xmax' is in attributes.")


def mask_arrival_azimuth(data_sets, fiducial_rmax):

    # Now we filter the events as a function of their arrival direction to
    # save computing time. Those events that won't make it to the fiducial
    # cylinder are discarded.

    rhos = np.sqrt(data_sets['xx'] ** 2 + data_sets['yy'] ** 2)

    # Let us considered our problem seen from above, projected on the z = vertex_z plane.
    # tangent_angle is the angle a tangent to the cylinder that reaches the vertex makes
    # with the antenna-vertex line on that plane.
    sine_tangent_angles = fiducial_rmax / rhos
    sine_tangent_angles[sine_tangent_angles > 1] = 1
    tangent_angles = np.arcsin(sine_tangent_angles)
    phis_low = 2 * np.pi - tangent_angles
    phis_high = tangent_angles
    phis_0 = np.arctan2(data_sets['yy'], data_sets['xx'])
    # NuRadioMC azimuth angles span [0, 2pi), unlike the result of arctan2: [-pi, pi)
    phis_0[phis_0 < 0] += 2 * np.pi
    phis = data_sets["azimuths"] - phis_0  # phi is the azimuth angle of the incoming neutrino if
                                           # we take phi = 0 as the vertex position
    phis[phis < 0] += 2 * np.pi

    mask_phi = [ (phi > phi_low and phi < 2 * np.pi) or (phi < phi_high and phi > 0) or rho < fiducial_rmax
                 for phi, phi_low, phi_high, rho in zip(phis, phis_low, phis_high, rhos) ]

    mask_phi = np.array(mask_phi)

    return mask_phi


def generate_surface_muons(filename, n_events, Emin, Emax,
                           volume,
                           thetamin=0.*units.rad, thetamax=np.pi * units.rad,
                           phimin=0.*units.rad, phimax=2 * np.pi * units.rad,
                           start_event_id=1,
                           plus_minus='mix',
                           n_events_per_file=None,
                           spectrum='log_uniform',
                           start_file_id=0,
                           config_file='SouthPole',
                           proposal_kwargs={},
                           log_level=None,
                           max_n_events_batch=1e5,
                           seed=None):
    """
    Event generator for surface muons

    Generates muons at the surface for the atmospheric muon acceptance studies.
    All events are saved in an hdf5 file.

    Parameters
    ----------
    filename: string
        the output filename of the hdf5 file
    n_events: int
        number of events to generate
    Emin: float
        the minimum neutrino energy (energies are randomly chosen assuming a
        uniform distribution in the logarithm of the energy)
    Emax: float
        the maximum neutrino energy (energies are randomly chosen assuming a
        uniform distribution in the logarithm of the energy)
    volume: dict
        a dictionary specifying the simulation volume
        can be either a cylinder spefified via the keys

        * fiducial_rmin: float
            lower r coordinate of fiducial volume (the fiducial volume needs to be chosen large enough such that no events outside of it will trigger)
        * fiducial_rmax: float
            upper r coordinate of fiducial volume (the fiducial volume needs to be chosen large enough such that no events outside of it will trigger)
        * fiducial_zmin: float
            lower z coordinate of fiducial volume (the fiducial volume needs to be chosen large enough such that no events outside of it will trigger)
        * fiducial_zmax: float
            upper z coordinate of fiducial volume (the fiducial volume needs to be chosen large enough such that no events outside of it will trigger)
        * full_rmin: float (optional)
            lower r coordinate of simulated volume (if not set it is set to the fiducial volume)
        * full_rmax: float (optional)
            upper r coordinate of simulated volume (if not set it is set to the fiducial volume)
        * full_zmin: float (optional)
            lower z coordinate of simulated volume (if not set it is set to the fiducial volume)
        * full_zmax: float (optional)
            upper z coordinate of simulated volume (if not set it is set to the fiducial volume)

        or a cube specified with 

        * fiducial_xmin: float
            lower x coordinate of fiducial volume (the fiducial volume needs to be chosen large enough such that no events outside of it will trigger)
        * fiducial_xmax: float
            upper x coordinate of fiducial volume (the fiducial volume needs to be chosen large enough such that no events outside of it will trigger)
        * fiducial_ymin: float
            lower y coordinate of fiducial volume (the fiducial volume needs to be chosen large enough such that no events outside of it will trigger)
        * fiducial_ymax: float
            upper y coordinate of fiducial volume (the fiducial volume needs to be chosen large enough such that no events outside of it will trigger)
        * fiducial_zmin: float
            lower z coordinate of fiducial volume (the fiducial volume needs to be chosen large enough such that no events outside of it will trigger)
        * fiducial_zmax: float
            upper z coordinate of fiducial volume (the fiducial volume needs to be chosen large enough such that no events outside of it will trigger)
        * full_xmin: float (optional)
            lower x coordinate of simulated volume (if not set it is set to the fiducial volume)
        * full_xmax: float (optional)
            upper x coordinate of simulated volume (if not set it is set to the fiducial volume)
        * full_ymin: float (optional)
            lower y coordinate of simulated volume (if not set it is set to the fiducial volume)
        * full_ymax: float (optional)
            upper y coordinate of simulated volume (if not set it is set to the fiducial volume)
        * full_zmin: float (optional)
            lower z coordinate of simulated volume (if not set it is set to the fiducial volume)
        * full_zmax: float (optional)
            upper z coordinate of simulated volume (if not set it is set to the fiducial volume)
        
    thetamin: float
        lower zenith angle for neutrino arrival direction
    thetamax: float
        upper zenith angle for neutrino arrival direction
    phimin: float
        lower azimuth angle for neutrino arrival direction
    phimax: float
        upper azimuth angle for neutrino arrival direction
    start_event: int
        default: 1
        event number of first event
    plus_minus: string
        if 'plus': generates only positive muons
        if 'minus': generates only negative muons
        else generates positive and negative muons randomly
    n_events_per_file: int or None
        the maximum number of events per output files. Default is None, which
        means that all events are saved in one file. If 'n_events_per_file' is
        smaller than 'n_events' the event list is split up into multiple files.
        This is useful to split up the computing on multiple cores.
    spectrum: string
        defines the probability distribution for which the neutrino energies are generated
        
        * 'log_uniform': uniformly distributed in the logarithm of energy
        * 'E-?': E to the -? spectrum where ? can be any float
        * 'IceCube-nu-2017': astrophysical neutrino flux measured with IceCube muon sample (https://doi.org/10.22323/1.301.1005)
        * 'GZK-1': GZK neutrino flux model from van Vliet et al., 2019, https://arxiv.org/abs/1901.01899v1 
            for 10% proton fraction (see get_GZK_1 function for details)
        * 'GZK-1+IceCube-nu-2017': a combination of the cosmogenic (GZK-1) and astrophysical (IceCube nu 2017) flux
        
    start_file_id: int (default 0)
        in case the data set is distributed over several files, this number specifies the id of the first file
        (useful if an existing data set is extended)
        if True, generate deposited energies instead of primary neutrino energies
    config_file: string
        The user can specify the path to their own config file or choose among
        the three available options:
        
        * 'SouthPole', a config file for the South Pole (spherical Earth). It
            consists of a 2.7 km deep layer of ice, bedrock below and air above.
        * 'MooresBay', a config file for Moore's Bay (spherical Earth). It
            consists of a 576 m deep ice layer with a 2234 m deep water layer below,
            and bedrock below that.
        * 'InfIce', a config file with a medium of infinite ice
        * 'Greenland', a config file for Summit Station, Greenland (spherical Earth),
            same as SouthPole but with a 3 km deep ice layer.
        
        .. Important:: If these options are used, the code is more efficient if the
            user requests their own "path_to_tables" and "path_to_tables_readonly",
            pointing them to a writable directory
            If one of these three options is chosen, the user is supposed to edit
            the corresponding config_PROPOSAL_xxx.json.sample file to include valid
            table paths and then copy this file to config_PROPOSAL_xxx.json.
        
    proposal_kwargs: dict
        additional kwargs that are passed to the get_secondaries_array function of the NuRadioProposal class
    log_level: logging log level or None
        sets the log level in the event generation. None means system default.
    max_n_events_batch: int (default 1e6)
        the maximum numbe of events that get generated per batch. Relevant if a fiducial volume cut is applied)
    seed: None of int
        seed of the random state
    """
    rnd = Generator(Philox(seed))
    if(log_level is not None):
        logger.setLevel(log_level)
    t_start = time.time()
    max_n_events_batch = int(max_n_events_batch)
    from NuRadioMC.EvtGen.NuRadioProposal import ProposalFunctions
    proposal_functions = ProposalFunctions(config_file=config_file)

    attributes = {}
    n_events = int(n_events)

    # save current NuRadioMC version as attribute
    # save NuRadioMC and NuRadioReco versions
    attributes['NuRadioMC_EvtGen_version'] = NuRadioMC.__version__
    attributes['NuRadioMC_EvtGen_version_hash'] = version.get_NuRadioMC_commit_hash()

    attributes['n_events'] = n_events
    attributes['start_event_id'] = start_event_id

    if (plus_minus == 'plus'):
        flavor = [-13]
    elif (plus_minus == 'minus'):
        flavor = [13]
    else:
        flavor = [13, -13]

    attributes['flavors'] = flavor
    attributes['Emin'] = Emin
    attributes['Emax'] = Emax
    attributes['thetamin'] = thetamin
    attributes['thetamax'] = thetamax
    attributes['phimin'] = phimin
    attributes['phimax'] = phimax
    attributes['deposited'] = False

    data_sets_fiducial = {}
    proposal_time = 0

    set_volume_attributes(volume, proposal=False, attributes=attributes)
    n_events = attributes['n_events']  # important! the number of events might have been increased by the set_volume_attributes function
    n_batches = int(np.ceil(n_events / max_n_events_batch))
    for i_batch in range(n_batches):  # do generation of events in batches
        data_sets = {}
        n_events_batch = max_n_events_batch
        if(i_batch + 1 == n_batches):  # last batch?
            n_events_batch = n_events - (i_batch * max_n_events_batch)
        data_sets["xx"], data_sets["yy"], data_sets["zz"] = generate_vertex_positions(attributes=attributes, n_events=n_events_batch, rnd=rnd)
        data_sets["zz"] = np.zeros_like(data_sets["yy"])  # muons interact at the surface

        # generate neutrino vertices randomly
        data_sets["azimuths"] = rnd.uniform(phimin, phimax, n_events_batch)
        # zenith directions are distruted as sin(theta) (to make the distribution istotropic) * cos(theta) (to account for the projection onto the surface)
        data_sets["zeniths"] = np.arcsin(rnd.uniform(np.sin(thetamin) ** 2, np.sin(thetamax) ** 2, n_events_batch) ** 0.5)

        data_sets["event_group_ids"] = np.arange(i_batch * max_n_events_batch, i_batch * max_n_events_batch + n_events_batch, dtype=int) + start_event_id
        data_sets["n_interaction"] = np.ones(n_events_batch, dtype=int)
        data_sets["vertex_times"] = np.zeros(n_events_batch, dtype=np.float)

        # generate neutrino flavors randomly

        data_sets["flavors"] = np.array([flavor[i] for i in rnd.integers(0, high=len(flavor), size=n_events_batch)])

        data_sets["energies"] = get_energies(n_events_batch, Emin, Emax, spectrum, rnd)

        # generate charged/neutral current randomly
        data_sets["interaction_type"] = [ '' ] * n_events_batch

        # generate inelasticity
        data_sets["inelasticity"] = np.zeros(n_events_batch)

        data_sets["energies"] = np.array(data_sets["energies"])
        data_sets["muon_energies"] = np.copy(data_sets["energies"])

        # create dummy entries for shower energies and types
        data_sets['shower_energies'] = np.zeros(n_events_batch)
        data_sets['shower_type'] = ['had'] * n_events_batch

        init_time = time.time()
        # Initialising data_sets_fiducial with empty values
        for key in data_sets:
            if(key not in data_sets_fiducial):
                data_sets_fiducial[key] = []
        logger.info(f"processing batch {i_batch+1:.4g}/{n_batches:.4g} with {n_events_batch:.6g} events ({len(data_sets_fiducial['event_group_ids'])} showers in fiducial volume so far.)")

        E_all_leptons = data_sets["energies"]
        lepton_codes = data_sets["flavors"]
        lepton_positions = [ (x, y, z) for x, y, z in zip(data_sets["xx"], data_sets["yy"], data_sets["zz"]) ]
        lepton_directions = [ (-np.sin(theta) * np.cos(phi), -np.sin(theta) * np.sin(phi), -np.cos(theta))
                            for theta, phi in zip(data_sets["zeniths"], data_sets["azimuths"])]

        if('fiducial_rmax' in attributes):
            mask_phi = mask_arrival_azimuth(data_sets, attributes['fiducial_rmax'])  # this currently only works for cylindrical volumes
        else:
            mask_phi = np.ones(len(data_sets["event_group_ids"]), dtype=np.bool)
        # TODO: combine with `get_intersection_volume_neutrino` function
        for iE, event_id in enumerate(data_sets["event_group_ids"]):
            if not mask_phi[iE]:
                continue

            # calculate if the lepton/neutrino direction intersects the fiducial simulation volume
            geometry_selection = get_intersection_volume_neutrino(attributes,
                                                                  [data_sets['xx'][iE], data_sets['yy'][iE], data_sets['zz'][iE]],
                                                                  lepton_directions[iE])
    #         geometry_selection = True
            if geometry_selection:

                products_array = proposal_functions.get_secondaries_array(np.array([E_all_leptons[iE]]),
                                                                           np.array([lepton_codes[iE]]),
                                                                           np.array([lepton_positions[iE]]),
                                                                           np.array([lepton_directions[iE]]),
                                                                           **proposal_kwargs)
                products = products_array[0]

                n_interaction = 1

                for product in products:
                    x, y, z, vertex_time = get_product_position_time(data_sets, product, iE)
                    if(is_in_fiducial_volume(attributes, np.array([x, y, z]))):
                        # the energy loss or particle is in our fiducial volume
                        # save parent muon if one of its induced showers interacts in the fiducial volume
                        if(n_interaction == 1):
                            for key in iterkeys(data_sets):
                                data_sets_fiducial[key].append(data_sets[key][iE])
                            n_interaction = 2

                        for key in iterkeys(data_sets):
                            data_sets_fiducial[key].append(data_sets[key][iE])

                        data_sets_fiducial['n_interaction'][-1] = n_interaction  # specify that new event is a secondary interaction
                        n_interaction += 1
                        data_sets_fiducial['shower_energies'][-1] = product.energy
                        data_sets_fiducial['inelasticity'][-1] = 1
                        # interaction_type is either 'had' or 'em' for proposal products
                        data_sets_fiducial['interaction_type'][-1] = product.shower_type
                        data_sets_fiducial['shower_type'][-1] = product.shower_type
                        data_sets_fiducial['xx'][-1] = x
                        data_sets_fiducial['yy'][-1] = y
                        data_sets_fiducial['zz'][-1] = z

                        # Calculating vertex interaction time with respect to the primary neutrino
                        data_sets_fiducial['vertex_times'][-1] = vertex_time

                        # Flavors are particle codes taken from NuRadioProposal.py
                        data_sets_fiducial['flavors'][-1] = product.code
        proposal_time += time.time() - init_time

    time_per_evt = proposal_time / len(data_sets_fiducial['flavors'])
    logger.info(f"Time per event: {time_per_evt*1e3:.01f}ms")
    logger.info(f"Total time {pretty_time_delta(proposal_time)}")

    logger.status(f"number of fiducial showers {len(data_sets_fiducial['flavors'])}")

    # If there are no fiducial showers, passing an empty data_sets_fiducial to
    # write_events_to_hdf5 will cause the program to crash. However, we need
    # the output file to have empty data sets but also to have the total
    # number of input muons even though none of them triggers, so as not to
    # bias an effective volume calculation done with several files.
    # As a solution, we take a muon neutrino event (not an atmospheric muon)
    # at the top of the ice, and since its inelasticity is zero, it won't create
    # an electric field or trigger.
    if len(data_sets_fiducial["event_group_ids"]) == 0:
        for key, value in data_sets.items():
            data_sets_fiducial[key] = np.array([data_sets[key][0]])
        data_sets_fiducial['flavors'] = np.array([14])
        data_sets_fiducial['shower_energies'] = np.array([0])

    data_sets_fiducial["shower_ids"] = np.arange(0, len(data_sets_fiducial['shower_energies']), dtype=int)
    write_events_to_hdf5(filename, data_sets_fiducial, attributes, n_events_per_file=n_events_per_file, start_file_id=start_file_id)
    logger.status(f"finished in {pretty_time_delta(time.time() - t_start)}")
    return None


def generate_eventlist_cylinder(filename, n_events, Emin, Emax,
                                volume,
                                thetamin=0.*units.rad, thetamax=np.pi * units.rad,
                                phimin=0.*units.rad, phimax=2 * np.pi * units.rad,
                                start_event_id=1,
                                flavor=[12, -12, 14, -14, 16, -16],
                                n_events_per_file=None,
                                spectrum='log_uniform',
                                deposited=False,
                                proposal=False,
                                proposal_config='SouthPole',
                                start_file_id=0,
                                log_level=None,
                                proposal_kwargs={},
                                max_n_events_batch=1e5,
                                write_events=True,
                                seed=None,
                                interaction_type="ccnc"):
    """
    Event generator

    Generates neutrino interactions, i.e., vertex positions, neutrino directions,
    neutrino flavor, charged currend/neutral current and inelastiviy distributions.
    All events are saved in an hdf5 file.

    Parameters
    ----------
    filename: string
        the output filename of the hdf5 file
    n_events: int
        number of events to generate
    Emin: float
        the minimum neutrino energy
    Emax: float
        the maximum neutrino energy
    volume: dict
        a dictionary specifying the simulation volume
        can be either a cylinder spefified via the keys

        * fiducial_rmin: float
            lower r coordinate of fiducial volume (the fiducial volume needs to be chosen large enough such that no events outside of it will trigger)
        * fiducial_rmax: float
            upper r coordinate of fiducial volume (the fiducial volume needs to be chosen large enough such that no events outside of it will trigger)
        * fiducial_zmin: float
            lower z coordinate of fiducial volume (the fiducial volume needs to be chosen large enough such that no events outside of it will trigger)
        * fiducial_zmax: float
            upper z coordinate of fiducial volume (the fiducial volume needs to be chosen large enough such that no events outside of it will trigger)
        * full_rmin: float (optional)
            lower r coordinate of simulated volume (if not set it is set to 1/3 of the fiducial volume, if second vertices are not activated it is set to the fiducial volume)
        * full_rmax: float (optional)
            upper r coordinate of simulated volume (if not set it is set to the fiducial volume + the 95% quantile of the tau decay length, if second vertices are not activated it is set to the fiducial volume)
        * full_zmin: float (optional)
            lower z coordinate of simulated volume (if not set it is set to the fiducial volume - the tau decay length, if second vertices are not activated it is set to the fiducial volume)
        * full_zmax: float (optional)
            upper z coordinate of simulated volume (if not set it is set to 1/3 of the fiducial volume , if second vertices are not activated it is set to the fiducial volume)
        
        or a cube specified with

        * fiducial_xmin: float
            lower x coordinate of fiducial volume (the fiducial volume needs to be chosen large enough such that no events outside of it will trigger)
        * fiducial_xmax: float
            upper x coordinate of fiducial volume (the fiducial volume needs to be chosen large enough such that no events outside of it will trigger)
        * fiducial_ymin: float
            lower y coordinate of fiducial volume (the fiducial volume needs to be chosen large enough such that no events outside of it will trigger)
        * fiducial_ymax: float
            upper y coordinate of fiducial volume (the fiducial volume needs to be chosen large enough such that no events outside of it will trigger)
        * fiducial_zmin: float
            lower z coordinate of fiducial volume (the fiducial volume needs to be chosen large enough such that no events outside of it will trigger)
        * fiducial_zmax: float
            upper z coordinate of fiducial volume (the fiducial volume needs to be chosen large enough such that no events outside of it will trigger)
        * full_xmin: float (optional)
            lower x coordinate of simulated volume (if not set it is set to the fiducial volume - the 95% quantile of the tau decay length, if second vertices are not activated it is set to the fiducial volume)
        * full_xmax: float (optional)
            upper x coordinate of simulated volume (if not set it is set to the fiducial volume + the 95% quantile of the tau decay length, if second vertices are not activated it is set to the fiducial volume)
        * full_ymin: float (optional)
            lower y coordinate of simulated volume (if not set it is set to the fiducial volume - the 95% quantile of the tau decay length, if second vertices are not activated it is set to the fiducial volume)
        * full_ymax: float (optional)
            upper y coordinate of simulated volume (if not set it is set to the fiducial volume + the 95% quantile of the tau decay length, if second vertices are not activated it is set to the fiducial volume)
        * full_zmin: float (optional)
            lower z coordinate of simulated volume (if not set it is set to 1/3 of the fiducial volume, if second vertices are not activated it is set to the fiducial volume)
        * full_zmax: float (optional)
            upper z coordinate of simulated volume (if not set it is set to the fiducial volume - the tau decay length, if second vertices are not activated it is set to the fiducial volume)
    
    thetamin: float
        lower zenith angle for neutrino arrival direction (default 0deg)
    thetamax: float
        upper zenith angle for neutrino arrival direction
    phimin: float
        lower azimuth angle for neutrino arrival direction
    phimax: float
         upper azimuth angle for neutrino arrival direction
    start_event_id: int
        default: 1
        event number of first event
    flavor: array of ints
        default: [12, -12, 14, -14, 16, -16]
        specify which neutrino flavors to generate. A uniform distribution of
        all specified flavors is assumed.

        The neutrino flavor (integer) encoded as using PDF numbering scheme,
        particles have positive sign, anti-particles have negative sign,
        relevant for us are:

        * 12: electron neutrino
        * 14: muon neutrino
        * 16: tau neutrino

    n_events_per_file: int or None
        the maximum number of events per output files. Default is None, which
        means that all events are saved in one file. If 'n_events_per_file' is
        smaller than 'n_events' the event list is split up into multiple files.
        This is useful to split up the computing on multiple cores.
    spectrum: string
        defines the probability distribution for which the neutrino energies are generated

        * 'log_uniform': uniformly distributed in the logarithm of energy
        * 'E-?': E to the -? spectrum where ? can be any float
        * 'IceCube-nu-2017': astrophysical neutrino flux measured with IceCube muon sample (https://doi.org/10.22323/1.301.1005)
        * 'GZK-1': GZK neutrino flux model from van Vliet et al., 2019, https://arxiv.org/abs/1901.01899v1 
            for 10% proton fraction (see get_GZK_1 function for details)
        * 'GZK-1+IceCube-nu-2017': a combination of the cosmogenic (GZK-1) and astrophysical (IceCube nu 2017) flux

    deposited: bool
        if True, generate deposited energies instead of primary neutrino energies
    proposal: bool
        if True, the tau and muon secondaries are calculated using PROPOSAL
    proposal_config: string or path
        The user can specify the path to their own config file or choose among
        the three available options:

        * 'SouthPole', a config file for the South Pole (spherical Earth). It
          consists of a 2.7 km deep layer of ice, bedrock below and air above.
        * 'MooresBay', a config file for Moore's Bay (spherical Earth). It
          consists of a 576 m deep ice layer with a 2234 m deep water layer below,
          and bedrock below that.
        * 'InfIce', a config file with a medium of infinite ice
        * 'Greenland', a config file for Summit Station, Greenland (spherical Earth),
          same as SouthPole but with a 3 km deep ice layer.

        .. Important:: If these options are used, the code is more efficient if the
            user requests their own "path_to_tables" and "path_to_tables_readonly",
            pointing them to a writable directory
            If one of these three options is chosen, the user is supposed to edit
            the corresponding config_PROPOSAL_xxx.json.sample file to include valid
            table paths and then copy this file to config_PROPOSAL_xxx.json.

    start_file_id: int (default 0)
        in case the data set is distributed over several files, this number specifies the id of the first file
        (useful if an existing data set is extended)
    log_level: logging log level or None
        sets the log level in the event generation. None means system default.
    proposal_kwargs: dict
        additional kwargs that are passed to the get_secondaries_array function of the NuRadioProposal class
    max_n_events_batch: int (default 1e6)
        the maximum numbe of events that get generated per batch. Relevant if a fiducial volume cut is applied)
    write_events: bool
        if True (default) the event list is written to an output file
        if False the event datasets + atrributes are returned
    seed: None of int
        seed of the random state
    interaction_type: string
        the interaction type. default is "ccnc" which randomly choses neutral current (NC) or charged-current (CC) interactions. 
        The use can also specify "nc" or "cc" to exclusively simulate NC or CC interactions
    """
    rnd = Generator(Philox(seed))
    t_start = time.time()
    if(log_level is not None):
        logger.setLevel(log_level)
    if proposal:
        from NuRadioMC.EvtGen.NuRadioProposal import ProposalFunctions
        proposal_functions = ProposalFunctions(config_file=proposal_config)
    max_n_events_batch = int(max_n_events_batch)
    attributes = {}
    n_events = int(n_events)

    # save current NuRadioMC version as attribute
    # save NuRadioMC and NuRadioReco versions
    attributes['NuRadioMC_EvtGen_version'] = NuRadioMC.__version__
    attributes['NuRadioMC_EvtGen_version_hash'] = version.get_NuRadioMC_commit_hash()
    attributes['start_event_id'] = start_event_id
    attributes['n_events'] = n_events
    attributes['flavors'] = flavor
    attributes['Emin'] = Emin
    attributes['Emax'] = Emax
    attributes['thetamin'] = thetamin
    attributes['thetamax'] = thetamax
    attributes['phimin'] = phimin
    attributes['phimax'] = phimax
    attributes['deposited'] = deposited

    data_sets = {}
    data_sets_fiducial = {}

    time_proposal = 0

    set_volume_attributes(volume, proposal=proposal, attributes=attributes)
    n_events = attributes['n_events']  # important! the number of events might have been increased by the generate vertex function
    n_batches = int(np.ceil(n_events / max_n_events_batch))
    for i_batch in range(n_batches):  # do generation of events in batches
        n_events_batch = max_n_events_batch
        if(i_batch + 1 == n_batches):  # last batch?
            n_events_batch = n_events - (i_batch * max_n_events_batch)
        logger.info(f"processing batch {i_batch+1:.2g}/{n_batches:.2g} with {n_events_batch:.2g} events")
        data_sets["xx"], data_sets["yy"], data_sets["zz"] = generate_vertex_positions(attributes=attributes, n_events=n_events_batch, rnd=rnd)

        data_sets["azimuths"] = rnd.uniform(phimin, phimax, n_events_batch)
        data_sets["zeniths"] = np.arccos(rnd.uniform(np.cos(thetamax), np.cos(thetamin), n_events_batch))

    #     fmask = (rr_full >= fiducial_rmin) & (rr_full <= fiducial_rmax) & (data_sets["zz"] >= fiducial_zmin) & (data_sets["zz"] <= fiducial_zmax)  # fiducial volume mask

        logger.debug("generating event ids")
        data_sets["event_group_ids"] = np.arange(i_batch * max_n_events_batch, i_batch * max_n_events_batch + n_events_batch) + start_event_id
        logger.debug("generating number of interactions")
        data_sets["n_interaction"] = np.ones(n_events_batch, dtype=int)
        data_sets["vertex_times"] = np.zeros(n_events_batch, dtype=np.float)

        # generate neutrino flavors randomly
        logger.debug("generating flavors")
        data_sets["flavors"] = np.array([flavor[i] for i in rnd.integers(0, high=len(flavor), size=n_events_batch)])

        # generate energies randomly
        data_sets["energies"] = get_energies(n_events_batch, Emin, Emax, spectrum, rnd)
        # generate charged/neutral current randomly
        logger.debug("interaction type")
        if(interaction_type == "ccnc"):
            data_sets["interaction_type"] = inelasticities.get_ccnc(n_events_batch, rnd=rnd)
        elif(interaction_type == "cc"):
            data_sets["interaction_type"] = np.full(n_events_batch, "cc", dtype='U2')
        elif(interaction_type == "nc"):
            data_sets["interaction_type"] = np.full(n_events_batch, "nc", dtype='U2')

        # generate inelasticity
        logger.debug("generating inelasticities")
        data_sets["inelasticity"] = inelasticities.get_neutrino_inelasticity(n_events_batch, rnd=rnd)

        if deposited:
            data_sets["energies"] = [primary_energy_from_deposited(Edep, ccnc, flavor, inelasticity) \
                                    for Edep, ccnc, flavor, inelasticity in \
                                    zip(data_sets["energies"], data_sets["interaction_type"], \
                                    data_sets["flavors"], data_sets["inelasticity"])]
            data_sets["energies"] = np.array(data_sets["energies"])

        # all interactions will produce a hadronic shower, add this information to the input file
        data_sets['shower_energies'] = data_sets['energies'] * data_sets['inelasticity']
        data_sets['shower_type'] = ['had'] * n_events_batch

        logger.debug("adding EM showers")
        # now add EM showers if appropriate
        em_shower_mask = (data_sets["interaction_type"] == "cc") & (np.abs(data_sets['flavors']) == 12)

        for key in data_sets:  # transform datatype to list so that inserting elements is faster
            data_sets[key] = list(data_sets[key])
        n_inserted = 0
        for i in np.arange(n_events_batch, dtype=int)[em_shower_mask]:  # loop over all events where an EM shower needs to be inserted
            for key in data_sets:
                data_sets[key].insert(i + 1 + n_inserted, data_sets[key][i + n_inserted])  # copy event
            data_sets['shower_energies'][i + 1 + n_inserted] = (1 - data_sets['inelasticity'][i + 1 + n_inserted]) * data_sets['energies'][i + 1 + n_inserted]
            data_sets['shower_type'][i + 1 + n_inserted] = 'em'
            n_inserted += 1

        logger.debug("converting to numpy arrays")
        # make all arrays numpy arrays
        for key in data_sets:
            data_sets[key] = np.array(data_sets[key])

        if proposal:
            logger.debug("starting proposal simulation")
            init_time = time.time()
            # Initialising data_sets_fiducial with empty values
            for key, value in iteritems(data_sets):
                if(key not in data_sets_fiducial):
                    data_sets_fiducial[key] = []

            # we need to be careful to not double cound events. electron CC interactions apear twice in the event list
            # because of the two distinct showers that get created. Because second interactions are only calculated
            # for mu and tau cc interactions, this is not a problem.
            mask_tau_cc = (data_sets["interaction_type"] == 'cc') & (np.abs(data_sets["flavors"]) == 16)
            mask_mu_cc = (data_sets["interaction_type"] == 'cc') & (np.abs(data_sets["flavors"]) == 14)
            mask_leptons = mask_tau_cc | mask_mu_cc

            E_all_leptons = (1 - data_sets["inelasticity"]) * data_sets["energies"]
            lepton_codes = copy.copy(data_sets["flavors"])
            lepton_codes[lepton_codes == 14] = 13
            lepton_codes[lepton_codes == -14] = -13
            lepton_codes[lepton_codes == 16] = 15
            lepton_codes[lepton_codes == -16] = -15

            if("fiducial_rmax" in attributes):
                mask_phi = mask_arrival_azimuth(data_sets, attributes['fiducial_rmax'])
                mask_leptons = mask_leptons & mask_phi
                # TODO: combine with `get_intersection_volume_neutrino` function

            lepton_positions = [ (x, y, z) for x, y, z in zip(data_sets["xx"], data_sets["yy"], data_sets["zz"]) ]
            lepton_positions = np.array(lepton_positions)
            lepton_directions = [ (-np.sin(theta) * np.cos(phi), -np.sin(theta) * np.sin(phi), -np.cos(theta))
                                for theta, phi in zip(data_sets["zeniths"], data_sets["azimuths"])]
            lepton_directions = np.array(lepton_directions)

            for iE, event_id in enumerate(data_sets["event_group_ids"]):
                first_inserted = False

                x_nu = data_sets['xx'][iE]
                y_nu = data_sets['yy'][iE]
                z_nu = data_sets['zz'][iE]
                # Appending event if it interacts within the fiducial volume
                if(is_in_fiducial_volume(attributes, np.array([x_nu, y_nu, z_nu]))):
                    for key in iterkeys(data_sets):
                        data_sets_fiducial[key].append(data_sets[key][iE])

                    first_inserted = True

                if mask_leptons[iE]:
                    geometry_selection = get_intersection_volume_neutrino(attributes,
                                                          [data_sets['xx'][iE], data_sets['yy'][iE], data_sets['zz'][iE]],
                                                          lepton_directions[iE])
                    if geometry_selection:
                        products_array = proposal_functions.get_secondaries_array(np.array([E_all_leptons[iE]]),
                                                                                   np.array([lepton_codes[iE]]),
                                                                                   np.array([lepton_positions[iE]]),
                                                                                   np.array([lepton_directions[iE]]),
                                                                                   **proposal_kwargs)
                        products = products_array[0]
                        n_interaction = 2
                        for product in products:

                            x, y, z, vertex_time = get_product_position_time(data_sets, product, iE)
                            if(is_in_fiducial_volume(attributes, np.array([x, y, z]))):

                                # the energy loss or particle is in our fiducial volume

                                # If the energy loss or particle is in the fiducial volume but the parent
                                # neutrino does not interact there, we add it to know its properties.
                                if not first_inserted:
                                    copies = 2
                                    first_inserted = True
                                else:
                                    copies = 1

                                for icopy in range(copies):
                                    for key in iterkeys(data_sets):
                                        data_sets_fiducial[key].append(data_sets[key][iE])

                                data_sets_fiducial['n_interaction'][-1] = n_interaction  # specify that new event is a secondary interaction
                                n_interaction += 1
                                data_sets_fiducial['shower_energies'][-1] = product.energy
                                data_sets_fiducial['inelasticity'][-1] = np.nan
                                # interaction_type is either 'had' or 'em' for proposal products
                                data_sets_fiducial['interaction_type'][-1] = product.shower_type
                                data_sets_fiducial['shower_type'][-1] = product.shower_type

                                data_sets_fiducial['xx'][-1] = x
                                data_sets_fiducial['yy'][-1] = y
                                data_sets_fiducial['zz'][-1] = z

                                # Calculating vertex interaction time with respect to the primary neutrino
                                data_sets_fiducial['vertex_times'][-1] = vertex_time

                                # Flavors are particle codes taken from NuRadioProposal.py
                                data_sets_fiducial['flavors'][-1] = product.code
            time_proposal = time.time() - init_time
        else:
            if(n_batches == 1):
                data_sets_fiducial = data_sets
            else:
                for key in data_sets:
                    if(key not in data_sets_fiducial):
                        data_sets_fiducial[key] = []
                    data_sets_fiducial[key].extend(data_sets[key])

    time_per_evt = time_proposal / (n_events + 1)
    logger.info(f"Time per event (PROPOSAL only): {time_per_evt*1e3:.4f} ms")
    logger.info(f"Total time (PROPOSAL only) {pretty_time_delta(time_proposal)}")

    logger.info(f"number of fiducial showers {len(data_sets_fiducial['xx'])}")

    # assign every shower a unique id
    data_sets_fiducial["shower_ids"] = np.arange(0, len(data_sets_fiducial['shower_energies']), dtype=int)
    # make the event group ids consecutive, this is useful if secondary interactions are simulated where many of the
    # initially generated neutrinos don't end up in the fiducial volume
    data_sets_fiducial['event_group_ids'] = np.asarray(data_sets_fiducial['event_group_ids'])
    egids = data_sets_fiducial['event_group_ids']
    uegids, uegids_inverse = np.unique(egids, return_inverse=True)
    data_sets_fiducial['event_group_ids'] = uegids_inverse + start_event_id

    if(write_events):
        write_events_to_hdf5(filename, data_sets_fiducial, attributes, n_events_per_file=n_events_per_file, start_file_id=start_file_id)
        logger.status(f"finished in {pretty_time_delta(time.time() - t_start)}")
    else:
        for key, value in data_sets_fiducial.items():
            if value.dtype.kind == 'U':
                data_sets_fiducial[key] = np.array(value, dtype=h5py.string_dtype(encoding='utf-8'))

        logger.status(f"finished in {pretty_time_delta(time.time() - t_start)}")
        return data_sets_fiducial, attributes
