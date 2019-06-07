# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import numpy as np
from NuRadioMC.utilities import units
from NuRadioMC.utilities import inelasticities
from six import iterkeys, iteritems
from scipy import constants
from scipy.integrate import quad
from scipy.interpolate import interp1d
import scipy.interpolate as interpolate
from scipy.optimize import fsolve
from scipy.interpolate import RectBivariateSpline
import h5py
import os
import logging
logger = logging.getLogger("EventGen")
logging.basicConfig()

VERSION_MAJOR = 1
VERSION_MINOR = 1

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

def create_interp(filename):
    """
    Creates RectBivariateSpline functions for interpolating the decay
    times and energies from filename

    Parameters
    ----------
    filename: string
        name of the hdf5 file containing the table

    Returns
    -------
    (interp_time, interp_energies): tuple of RectBivariateSpline functions
    """
    fin = load_input_hdf5(filename)

    log_time_bins = np.log10(fin['rest_times'])
    log_energy_bins = np.log10(fin['initial_energies'])

    f_time = RectBivariateSpline(log_time_bins, log_energy_bins, np.log10(fin['decay_times']) )
    f_energies = RectBivariateSpline(log_time_bins, log_energy_bins, np.log10(fin['decay_energies']) )

    def interp_time(time, energy):
        return 10**f_time(np.log10(time), np.log10(energy))

    def interp_energies(time, energy):
        return 10**f_energies(np.log10(time), np.log10(energy))

    return (interp_time, interp_energies)

def mean_energy_loss(energy):
    """
    Returns the mean energy loss of a tau per g/cm2 as a function of the tau energy
    This function is a linear (in log scale) approximation above 1 PeV to the
    curve found in https://doi.org/10.1016/j.astropartphys.2006.11.003

    Parameters
    ----------
    energy: float
       Tau energy

    Returns
    -------
    Energy loss per amount of matter (float)
    """
    E0 = 1*units.PeV
    if (energy > E0):
        b1 = 1.e-7 * units.cm ** 2 / units.g
        b2 = 1.8e-7 * units.cm ** 2 / units.g
        return b1*energy + b2*energy*np.log10(energy/E0)
    else:
        return 0.

def get_tau_decay_rest(energy):
    """
    Calculates the random tau decay time without time dilation
    """
    # The tau decay time is taken assuming an exponential decay
    # and applying the inverse transform method
    tau_decay_rest = -np.log(1 - np.random.uniform(0, 1)) * tau_rest_lifetime

    return tau_decay_rest

def get_tau_decay_time(energy):
    """
    Calculates the random tau decay time taking into account time dilation
    """

    gamma = energy / tau_mass  # tau_mass must be in natural units (c = 1)
    tau_decay_rest = get_tau_decay_rest(energy)
    tau_decay_time = gamma * tau_decay_rest

    return tau_decay_time

def get_decay_time_losses(energy, distmax, average=False, compare=False, user_time=None):
    """
    Calculates the decay time assuming photonuclear energy losses above
    1 PeV and using the quasi-continuous approximation.
    See https://doi.org/10.1016/j.astropartphys.2006.11.003 for details.

    Parameters
    ----------
    energy: float
        energy of the incident neutrino
    distmax: float
        maximum distance for which we calculate energy losses.
        It should be similar to the maximal dimension of the simulation volume.
    average: bool
        If False, a random decay time at rest is calculated
        If True, the tau mean lifetime at rest is used
    compare: bool
        If True, returns a tuple with the decay time with losses and without
        If False, only the decay time with losses is returned
    user_time: float
        If user_time is not None, the tau decay time in rest frame is taken as
        equal to user_time and the average flag is ignored.

    Returns
    -------
    decay_time: float
        Tau decay time with photonuclear losses
    energy_decay: float
        Tau energy at the time of decay
    decay_time_no_losses: float
        Tau decay time without losses
    """
    E0 = 1*units.PeV
    if (energy <= E0):
        #raise ValueError('Energy is equal to or less than 1 PeV. Returning decay time without energy loss.')
        if user_time is None:
            return get_tau_decay_time(energy), energy
        else:
            gamma = energy/tau_mass
            return gamma*user_time, energy

    # At these energies, we can use the speed of light as the tau speed
    timemax = distmax/cspeed
    Estep = energy/1000.

    times = [0.]
    timebreak = None
    energies = np.arange(energy, E0, -Estep)
    if (energies[-1] != E0):
        energies = np.append(energies, E0)

    # This function returns the inverse of the energy loss, needed for the
    # calculation of the ellapsed times
    def loss_int(E):
        return 1./mean_energy_loss(E)

    # We loop over the energies and integrate the inverse of the energy loss
    # so that we obtain the corresponding time at which the particle has a
    # given energy.
    for finalenergy in energies[1:]:

        # If the energy is less than 1 PeV, we stop.
        if (finalenergy < E0):
            timebreak = quad(loss_int, E0, energy)/density_ice/cspeed
            times.append(timebreak)
            break

        time = quad(loss_int, finalenergy, energy)[0]/density_ice/cspeed
        times.append(time)
        if (time > timemax):
            break

    energies = energies[0:len(times)]

    if user_time is not None:
        tau_decay_rest = user_time
    elif not average:
        tau_decay_rest = get_tau_decay_rest(energy)
    else:
        tau_decay_rest = tau_rest_lifetime

    # We use an interpolation for having the energies as a function of time
    energies_interp = interp1d(times, energies)

    # This function returns the Lorentz factor for a given time t
    def gamma(t):
        if timebreak is not None and t > timebreak:
            return E0/tau_mass
        elif (t > timemax):
            return energies[-1]/tau_mass
        elif (t > times[-1]):
            return E0/tau_mass
        elif (t < 0):
            return np.inf
        else:
            return energies_interp(t)/tau_mass

    # This function returns the inverse of the Lorentz factor at a time t
    def inv_gamma(t):
        return 1./gamma(t)

    # This function integrates the inverse of the Lorentz factor in order
    # to obtain the proper time for the tau between the times t0 and t1
    def proper_time(gamma_function, t0, t1):
        return quad(inv_gamma, t0, t1)[0]

    # This function returns the difference between the proper time of the tau
    # at a time t and its decay time (in the tau rest frame)
    def times_diff(t):
        return proper_time(gamma, 0, t)-tau_decay_rest

    # We obtain the decay time for the observer finding the roots for the
    # difference between the proper time and the decay time in the rest frame
    decay_time = fsolve(times_diff,1e3*units.ns)[0]
    energy_decay = gamma(decay_time)*tau_mass

    if not compare:
        return decay_time, energy_decay
    else:
        decay_time_no_losses = tau_decay_rest*gamma(0)
        return decay_time, decay_time_no_losses, energy_decay


def get_tau_speed(energy):
    """
    Calculates the speed of the tau lepton
    """

    gamma = energy / tau_mass
    if (gamma < 1):
        #raise ValueError('The energy is less than the tau mass. Returning zero speed')
        return 0
    beta = np.sqrt(1 - 1 / gamma ** 2)

    return beta * constants.c * units.m / units.s

def get_tau_decay_length(energy, distmax=0, table=None):
    """
    calculates the decay length of the tau

    Parameters
    ----------
    energy: float
       Tau energy
    distmax: float
    maximum distance for which we calculate energy losses.
    It should be similar to the maximal dimension of the simulation volume.
    table: RectBivariateSpline type function. See get_decay_time_tab.

    Returns
    -------
    decay_time, decay_energy: float, float
       Tau decay time and tau decay energy
    """

    if (energy <= 1*units.PeV):
        decay_time = get_tau_decay_time(energy)
        v = get_tau_speed(energy)
        return decay_time * v, energy
    else:
        if table is None:
            decay_time, decay_energy = get_decay_time_losses(energy, distmax)
        else:
            decay_time, decay_energy = get_decay_time_tab(table, energy)
        return decay_time * cspeed, decay_energy

def get_decay_time_tab(table, energy, time=None):
    """
    Calculates the decay time assuming photonuclear energy losses above
    1 PeV and using the quasi-continuous approximation.
    See https://doi.org/10.1016/j.astropartphys.2006.11.003 for details.
    This version uses tabulated histograms to speed up the computation.

    Parameters
    ----------
    table: tuple of 2 RectBivariateSpline type functions
        table[0](time,energy) must interpolate the decay time in lab frame
        table[1](time,energy) must interpolate the decay energy in lab frame
    energy: float
        energy of the incident neutrino
    time: float
        If time is not None, the tau decay time in rest frame is taken as
        equal to time. If time is None, a random time is drawn.

    Returns
    -------
    decay_time: float
        Tau decay time with photonuclear losses
    energy_decay: float
        Tau energy at the time of decay
    """

    if time is None:
        time = get_tau_decay_rest(energy)

    decay_time = table[0](time, energy)[0,0]
    decay_energy = table[1](time, energy)[0,0]

    return decay_time, decay_energy

def get_tau_decay_vertex(x, y, z, E, zenith, azimuth, distmax, table=None):
    """
     Let us assume that the tau has the same direction as the tau neutrino
     to calculate the vertex of the second shower

     Parameters
     ----------
     x: float
        x coordinate of the vertex position
     y: float
        y coordinate of the vertex position
     z: float
        z coordinate of the vertex position
     E: float
        Tau energy after neutrino interaction
     zenith: float
        Zenith arrival direction
     azimuth: float
        Azimuth arrival direction
     distmax: float
        maximum distance for which we calculate energy losses.
        It should be similar to the maximal dimension of the simulation volume.
     table: tuple of 2 RectBivariateSpline type functions

     Returns
     -------
     second_vertex_x: float
        x coordinate of the decay position
     second_vertex_y: float
        y coordinate of the decay position
     second_vertex_z: float
        z coordinate of the decay position
     decay_energy: float
        Tau energy at the moment of decay
    """
    L, decay_energy = get_tau_decay_length(E, distmax, table)
    second_vertex_x = L
    second_vertex_x *= np.sin(zenith) * np.cos(azimuth)
    second_vertex_x += x

    second_vertex_y = L
    second_vertex_y *= np.sin(zenith) * np.sin(azimuth)
    second_vertex_y += y

    second_vertex_z = L
    second_vertex_z *= np.cos(zenith)
    second_vertex_z += z
    return second_vertex_x, second_vertex_y, second_vertex_z, decay_energy

def get_tau_cascade_properties(tau_energy):
    """
    Given the energy of a decaying tau, calculates the properties of the
    resulting cascade.

    Parameters
    ----------
    tau_energy: float
       Tau energy at the moment of decay

    Returns
    -------
    cascade_energy: float
        The energy of the resulting cascade
    cascade_type: string
        Decay type: 'tau_had', 'tau_em', or 'tau_mu'
    """
    # TODO: calculate cascade energy
    # TODO: include the rest of the particles produced

    branch = inelasticities.random_tau_branch()
    products = inelasticities.inelasticity_tau_decay(tau_energy, branch)
    return products, branch



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
    n_events = len(np.unique(data_sets['event_ids']))
    logger.info("saving {} events in total".format(n_events))
    total_number_of_events = attributes['n_events']
    
    if "start_event_id" not in attributes:
        attributes["start_event_id"] = 0  # backward compatibility
    
    if(n_events_per_file is None):
        n_events_per_file = n_events
    else:
        n_events_per_file = int(n_events_per_file)
    iFile = -1
    evt_id_first = data_sets['event_ids'][0]
    evt_id_last_previous = 0  # save the last event id of the previous file
    start_index = 0
    n_events_total = 0
    while True:
        iFile += 1
        filename2 = filename
        evt_ids_this_file = np.unique(data_sets['event_ids'])[iFile * n_events_per_file : (iFile + 1) * n_events_per_file]
        if(len(evt_ids_this_file) == 0):
            logger.info("no more events to write in file {}".format(iFile))
            break

        if((iFile > 0) or (n_events_per_file < n_events)):
            filename2 = filename + ".part{:04}".format(iFile + 1 + start_file_id)
        fout = h5py.File(filename2, 'w')
        fout.attrs['VERSION_MAJOR'] = VERSION_MAJOR
        fout.attrs['VERSION_MINOR'] = VERSION_MINOR
        fout.attrs['header'] = HEADER
        for key, value in attributes.iteritems():
            fout.attrs[key] = value
        fout.attrs['total_number_of_events'] = total_number_of_events

        evt_id_first = evt_ids_this_file[0]
        evt_id_last = evt_ids_this_file[-1]

        tmp = np.squeeze(np.argwhere(data_sets['event_ids'] == evt_id_last))  # set stop index such that last event is competely in file
        if(tmp.size == 1):
            stop_index = tmp + 1
        else:
            stop_index = tmp[-1] + 1
#         if(evt_id_last >= n_events):
#             evt_id_last = n_events
#             stop_index = len(data_sets['event_ids'])
#         else:
#             tmp = np.squeeze(np.argwhere(data_sets['event_ids'] > evt_id_last))  # set stop index such that last event is competely in file
#             if(tmp.size == 1):
#                 stop_index = tmp
#             else:
#                 stop_index = tmp[0]

        for key, value in data_sets.iteritems():
            fout[key] = value[start_index:stop_index]

        # determine the number of events in this file (which is NOT the same as the entries in the file)
        # case 1) this is not the last file -> number of events is difference between last event id of the current and previous file + 1
        # case 2) it is the last file -> total number of simulated events - last event id of previous file
        # case 3) it is the first file -> last event id + 1 - start_event_id
        # case 4) it is the first and last file -> total number of simulated events
        evt_ids_next_file = np.unique(data_sets['event_ids'])[(iFile + 1) * n_events_per_file : (iFile + 2) * n_events_per_file]
        n_events_this_file = None
        if(iFile == 0 and len(evt_ids_next_file) == 0):  # case 4
            n_events_this_file = total_number_of_events
        elif(len(evt_ids_next_file) == 0): # last file -> case 2
            n_events_this_file = total_number_of_events - evt_id_last_previous + attributes['start_event_id']
        elif(iFile == 0): # case 3
            n_events_this_file = evt_id_last - attributes['start_event_id'] + 1
        else: # case 1
            n_events_this_file = evt_id_last - evt_id_last_previous

        print('writing file {} with {} events (id {} - {}) and {} entries'.format(filename2, n_events_this_file, evt_id_first,
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
        return Edep/inelasticity
    elif (ccnc == 'cc'):
        if (np.abs(flavor) == 12):
            return Edep
        elif (np.abs(flavor) == 14):
            return Edep/inelasticity
        elif (np.abs(flavor) == 16):
            return Edep/inelasticity # TODO: change this for taus
        
def ice_cube_nu_fit(energy, slope=-2.19, offset=1.01):
    # from https://doi.org/10.22323/1.301.1005
    # ApJ slope=-2.13, offset=0.9
    flux = 3 * offset * (energy / (100 * units.TeV))**slope * 1e-18 * \
        (units.GeV**-1 * units.cm**-2 * units.second**-1 * units.sr**-1)
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
    J *= units.GeV * units.cm**-2 * units.s**-1 * units.sr**-1 / E**2
    get_flux = interpolate.interp1d(E, J, fill_value=0, bounds_error=False)
    return get_flux(energy)

def get_energy_from_flux(Emin, Emax, n_events, flux):
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
        
    Returns: array of energies
    """
    
    xx_edges = np.linspace(Emin, Emax, 10000000)
    xx = 0.5 * (xx_edges[1:] + xx_edges[:-1])
    yy = flux(xx)
    cum_values = np.zeros(xx_edges.shape)
    cum_values[1:] = np.cumsum(yy * np.diff(xx_edges))
    inv_cdf = interpolate.interp1d(cum_values, xx_edges)
    r = np.random.uniform(0, cum_values.max(), n_events)
    return inv_cdf(r)

def generate_eventlist_cylinder(filename, n_events, Emin, Emax,
                                fiducial_rmin, fiducial_rmax, fiducial_zmin, fiducial_zmax,
                                full_rmin=None, full_rmax=None, full_zmin=None, full_zmax=None,
                                thetamin=0.*units.rad, thetamax=np.pi*units.rad,
                                phimin=0.*units.rad, phimax=2*np.pi*units.rad,
                                start_event_id=1,
                                flavor=[12, -12, 14, -14, 16, -16],
                                n_events_per_file=None,
                                spectrum='log_uniform',
                                add_tau_second_bang=False,
                                tabulated_taus=True,
                                deposited=False,
                                start_file_id=0):
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
        the minimum neutrino energy (energies are randomly chosen assuming a
        uniform distribution in the logarithm of the energy)
    Emax: float
        the maximum neutrino energy (energies are randomly chosen assuming a
        uniform distribution in the logarithm of the energy)

    fiducial_rmin: float
        lower r coordinate of fiducial volume (the fiducial volume needs to be chosen large enough such that no events outside of it will trigger)
    fiducial_rmax: float
        upper r coordinate of fiducial volume (the fiducial volume needs to be chosen large enough such that no events outside of it will trigger)
    fiducial_zmin: float
        lower z coordinate of fiducial volume (the fiducial volume needs to be chosen large enough such that no events outside of it will trigger)
    fiducial_zmax: float
        upper z coordinate of fiducial volume (the fiducial volume needs to be chosen large enough such that no events outside of it will trigger)
    full_rmin: float (default None)
        lower r coordinate of simulated volume (if None it is set to 1/3 of the fiducial volume, if second vertices are not activated it is set to the fiducial volume)
    full_rmax: float (default None)
        upper r coordinate of simulated volume (if None it is set to 5x the fiducial volume, if second vertices are not activated it is set to the fiducial volume)
    full_zmin: float (default None)
        lower z coordinate of simulated volume (if None it is set to 1/3 of the fiducial volume, if second vertices are not activated it is set to the fiducial volume)
    full_zmax: float (default None)
        upper z coordinate of simulated volume (if None it is set to 5x the fiducial volume, if second vertices are not activated it is set to the fiducial volume)
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
        * 'GZK-1': GZK neutrino flux model from van Vliet et al., 2019, https://arxiv.org/abs/1901.01899v1 for
                   10% proton fraction (see get_GZK_1 function for details)
        * 'GZK-1+IceCube-nu-2017': a combination of the cosmogenic (GZK-1) and astrophysical (IceCube nu 2017) flux
    add_tau_second_bang: bool
        if True simulate second vertices from tau decays
    tabulated_taus: bool
        if True the tau decay properties are taken from a table
    deposited: bool
        If True, generate deposited energies instead of primary neutrino energies
    start_file_id: int (default 0)
        in case the data set is distributed over several files, this number specifies the id of the first file
        (useful if an existing data set is extended)
    """
    attributes = {}
    n_events = int(n_events)
    attributes['n_events'] = n_events
    attributes['start_event_id'] = start_event_id

    attributes['fiducial_rmin'] = fiducial_rmin
    attributes['fiducial_rmax'] = fiducial_rmax
    attributes['fiducial_zmin'] = fiducial_zmin
    attributes['fiducial_zmax'] = fiducial_zmax

    if(full_rmin is None):
        if(add_tau_second_bang):
            full_rmin = fiducial_rmin / 3.
        else:
            full_rmin = fiducial_rmin
    if(full_rmax is None):
        if(add_tau_second_bang):
            full_rmax = fiducial_rmax * 5.
        else:
            full_rmax = fiducial_rmax
    if(full_zmin is None):
        if(add_tau_second_bang):
            full_zmin = fiducial_zmin * 5.
        else:
            full_zmin = fiducial_zmin
    if(full_zmax is None):
        if(add_tau_second_bang):
            full_zmax = fiducial_zmax / 3.
        else:
            full_zmax = fiducial_zmax

    attributes['rmin'] = full_rmin
    attributes['rmax'] = full_rmax
    attributes['zmin'] = full_zmin
    attributes['zmax'] = full_zmax
    attributes['flavors'] = flavor
    attributes['Emin'] = Emin
    attributes['Emax'] = Emax
    attributes['thetamin'] = thetamin
    attributes['thetamax'] = thetamax
    attributes['phimin'] = phimin
    attributes['phimax'] = phimax
    attributes['deposited'] = deposited

    data_sets = {}
    # generate neutrino vertices randomly
    data_sets["azimuths"] = np.random.uniform(phimin, phimax, n_events)
    u = np.random.uniform(np.cos(thetamax), np.cos(thetamin), n_events)
    data_sets["zeniths"] = np.arccos(u)  # generates distribution that is uniform in cos(theta)

    rr_full = np.random.triangular(full_rmin, full_rmax, full_rmax, n_events)
    data_sets["xx"] = rr_full * np.cos(data_sets["azimuths"])
    data_sets["yy"] = rr_full * np.sin(data_sets["azimuths"])
    data_sets["zz"] = np.random.uniform(full_zmin, full_zmax, n_events)
    fmask = (rr_full >= fiducial_rmin) & (rr_full <= fiducial_rmax) & (data_sets["zz"] >= fiducial_zmin) & (data_sets["zz"] <= fiducial_zmax)  # fiducial volume mask

    data_sets["event_ids"] = np.arange(n_events) + start_event_id
    data_sets["n_interaction"] = np.ones(n_events, dtype=np.int)

    # generate neutrino flavors randomly
    data_sets["flavors"] = np.array([flavor[i] for i in np.random.randint(0, high=len(flavor), size=n_events)])
    """
    #from AraSim nue:nueb:numu:numub:nutau:nutaub = 0.78: 0.22: 0.61: 0.39: 0.61: 0.39
    flaRnd = np.random.uniform(0., 3., n_events)
    flavors = np.ones(n_events, dtype = np.int64)
    for i, r in enumerate(flaRnd):
        if (r <= 0.78):
            flavors[i] = flavor[0]
        elif (r <= 1.0):
            flavors[i] = flavor[1]
        elif (r <= 1.61):
            flavors[i] = flavor[2]
        elif (r <= 2.0):
            flavors[i] = flavor[3]
        elif (r <= 2.61):
            flavors[i] = flavor[4]
        else:
            flavors[i] = flavor[5]
    """
    # generate energies randomly
    if(spectrum == 'log_uniform'):
        data_sets["energies"] = 10 ** np.random.uniform(np.log10(Emin), np.log10(Emax), n_events)
    elif(spectrum.startswith("E-")):  # enerate an E^gamma spectrum.  
        gamma = float(spectrum[1:])
        gamma += 1
        Nmin = (Emin)**gamma
        Nmax = (Emax)**gamma
        def get_inverse_spectrum(N, gamma):
            return np.exp(np.log(N)/gamma)
        data_sets["energies"] = get_inverse_spectrum(np.random.uniform(Nmax, Nmin, size=n_events), gamma)
    elif(spectrum == "GZK-1"):
        """
        model of (van Vliet et al., 2019, https://arxiv.org/abs/1901.01899v1) of the cosmogenic neutrino ﬂux
        for a source evolution parameter of m = 3.4, 
        a spectral index of the injection spectrum of α = 2.5, a cut-oﬀ rigidity of R = 100 EeV,
        and a proton fraction of 10% at E = 10^19.6 eV
        """
        data_sets["energies"] = get_energy_from_flux(Emin, Emax, n_events, get_GZK_1)
    elif(spectrum == "IceCube-nu-2017"):
        data_sets["energies"] = get_energy_from_flux(Emin, Emax, n_events, ice_cube_nu_fit)
    elif(spectrum == "GZK-1+IceCube-nu-2017"):
        def J(E):
            return ice_cube_nu_fit(E) + get_GZK_1(E)
        data_sets["energies"] = get_energy_from_flux(Emin, Emax, n_events, J)
    else:
        logger.error("spectrum {} not implemented".format(spectrum))
        raise NotImplementedError("spectrum {} not implemented".format(spectrum))

    # generate charged/neutral current randomly
    data_sets["interaction_type"] = inelasticities.get_ccnc(n_events)

    # generate inelasticity
    data_sets["inelasticity"] = inelasticities.get_neutrino_inelasticity(n_events)
    """
    #from AraSim
    epsilon = np.log10(energies / 1e9)
    inelasticity = pickY(flavors, ccncs, epsilon)
    """
    if deposited:
        data_sets["energies"] = [primary_energy_from_deposited(Edep, ccnc, flavor, inelasticity) \
                                for Edep, ccnc, flavor, inelasticity in \
                                zip(data_sets["energies"], data_sets["interaction_type"], \
                                data_sets["flavors"], data_sets["inelasticity"])]
        data_sets["energies"] = np.array(data_sets["energies"])

    data_sets_fiducial = {}

    if not add_tau_second_bang:
        # save only events with interactions in fiducial volume
        for key, value in iteritems(data_sets):
            data_sets_fiducial[key] = value[fmask]

    else:
        # Initialising data_sets_fiducial with empty values
        for key, value in iteritems(data_sets):
            data_sets_fiducial[key] = []

        if tabulated_taus:
            cdir = os.path.dirname(__file__)
            table = create_interp(os.path.join(cdir, 'decay_library.hdf5'))
        else:
            table = None

        mask = (data_sets["interaction_type"] == 'cc') & (np.abs(data_sets["flavors"]) == 16)
        logger.info("{} taus are created in nu tau interactions -> checking if tau decays in fiducial volume".format(np.sum(mask)))
        n_taus = 0
        for event_id in data_sets["event_ids"]:
            iE = event_id - start_event_id

            first_inserted = False

            x = data_sets['xx'][iE]
            y = data_sets['yy'][iE]
            z = data_sets['zz'][iE]
            r = (x ** 2 + y ** 2)**0.5

            # Appending event if it interacts within the fiducial volume
            if ( r >= fiducial_rmin and r <= fiducial_rmax ):
                if ( z >= fiducial_zmin and z <= fiducial_zmax ):

                    for key in iterkeys(data_sets):
                        data_sets_fiducial[key].append(data_sets[key][iE])

                    first_inserted = True

            if (data_sets["interaction_type"][iE] == 'cc' and np.abs(data_sets["flavors"][iE]) == 16):

                Etau = (1 - data_sets["inelasticity"][iE]) * data_sets["energies"][iE]

                # first calculate if tau decay is still in our fiducial volume
                x, y, z, decay_energy = get_tau_decay_vertex(data_sets["xx"][iE], data_sets["yy"][iE], data_sets["zz"][iE],
                                               Etau, data_sets["zeniths"][iE], data_sets["azimuths"][iE],
                                               np.sqrt(4*(full_rmax-full_rmin)**2+(full_zmax-full_zmin)**2), table=table)

                r = (x ** 2 + y ** 2)**0.5
                if( r >= fiducial_rmin and r <= fiducial_rmax ):
                    if(z >= fiducial_zmin and z <= fiducial_zmax):  # z coordinate is negative
                        # the tau decay is in our fiducial volume

                        n_taus += 1
                        # If the tau decays in the fiducial volume but the parent neutrino does not
                        # interact there, we add it to know its properties.
                        if not first_inserted:
                            copies = 2
                        else:
                            copies = 1

                        for icopy in range(copies):
                            for key in iterkeys(data_sets):
                                data_sets_fiducial[key].append(data_sets[key][iE])

                        y_cascade, cascade_type = get_tau_cascade_properties(decay_energy)
                        data_sets_fiducial['n_interaction'][-1] = 2 # specify that new event is a second interaction
                        data_sets_fiducial['energies'][-1] = decay_energy
                        data_sets_fiducial['inelasticity'][-1] = y_cascade
                        data_sets_fiducial['interaction_type'][-1] = cascade_type
                        # TODO: take care of the tau_mu
                        data_sets_fiducial['xx'][-1] = x
                        data_sets_fiducial['yy'][-1] = y
                        data_sets_fiducial['zz'][-1] = z

                        # set flavor to tau
                        data_sets_fiducial['flavors'][-1] = 15 * np.sign(data_sets['flavors'][iE])  # keep particle/anti particle nature
        logger.info("added {} tau decays to the event list".format(n_taus))

        # Transforming every array into a numpy array and copying it back to
        # data_sets_fiducial
        for key in iterkeys(data_sets):
            data_sets_fiducial[key] = np.array(data_sets_fiducial[key])

    write_events_to_hdf5(filename, data_sets_fiducial, attributes, n_events_per_file=n_events_per_file, start_file_id=start_file_id)


def split_hdf5_input_file(input_filename, output_filename, number_of_events_per_file):
    """
    splits up an existing hdf5 file into multiple subfiles

    Parameters
    ----------
    input_filename: string
        the input filename
    output_filename: string
        the desired output filename (if multiple files are generated, a 'part000x' is appended to the filename
    n_events_per_file: int (optional, default None)
        the number of events per file
    """
    fin = h5py.File(input_filename, 'r')
    data_sets = {}
    attributes = {}
    for key, value in fin.items():
        if isinstance(value, h5py.Dataset):  # the loop is also over potential subgroupu that we don't want to consider here
            data_sets[key] = np.array(value)
    for key, value in fin.attrs.items():
        attributes[key] = value

    fin.close()

    write_events_to_hdf5(output_filename, data_sets, attributes, n_events_per_file=number_of_events_per_file)
