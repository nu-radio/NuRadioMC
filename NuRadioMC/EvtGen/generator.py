from __future__ import absolute_import, division, print_function
import numpy as np
from NuRadioMC.utilities import units
from six import iterkeys, iteritems
from scipy import constants
import h5py
import logging
logger = logging.getLogger("EventGen")

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
tau_rest_time = 290.3 * units.fs

def get_tau_decay_time(energy):
    """
    Calculates the random tau decay time taking into account time dilation
    """

    gamma = energy / tau_mass  # tau_mass must be in natural units (c = 1)
    tau_mean_time = gamma * tau_rest_time

    # The tau decay time is taken assuming an exponential decay
    # and applying the inverse transform method
    tau_decay_time = -np.log(1 - np.random.uniform(0, 1)) * tau_mean_time

    return tau_decay_time

def get_tau_speed(energy):
    """
    Calculates the speed of the tau lepton
    """


    gamma = energy / tau_mass
    if (gamma < 1):
        raise ValueError('The energy is less than the tau mass. Returning zero speed')
        return 0
    beta = np.sqrt(1 - 1 / gamma ** 2)

    return beta * constants.c * units.m / units.s

def get_tau_decay_length(energy):
    """
    calculates the decay length of the tau
    """
    decay_time = get_tau_decay_time(energy)
    v = get_tau_speed(energy)
    return decay_time * v

def get_tau_decay_vertex(x, y, z, E, zenith, azimuth):
    """
     Let us assume that the tau has the same direction as the tau neutrino
     to calculate the vertex of the second shower
     This must be changed in the future
    """
    L = get_tau_decay_length(E)
    second_vertex_x = L
    second_vertex_x *= np.sin(zenith) * np.cos(azimuth)
    second_vertex_x += x

    second_vertex_y = L
    second_vertex_y *= np.sin(zenith) * np.sin(azimuth)
    second_vertex_y += y

    second_vertex_z = L
    second_vertex_z *= np.cos(zenith)
    second_vertex_z += z
    return second_vertex_x, second_vertex_y, second_vertex_z


def write_events_to_hdf5(filename, data_sets, attributes, n_events_per_file=None):
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
            filename2 = filename + ".part{:04}".format(iFile + 1)
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
            n_events_this_file = evt_id_last  - attributes['start_event_id']
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
                                add_tau_larger_volume=False):
                                deposited=False):
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
        lower r coordinate of simulated volume (if None it is set to 3x the fiducial volume)
    full_rmax: float (default None)
        upper r coordinate of simulated volume (if None it is set to 3x the fiducial volume)
    full_zmin: float (default None)
        lower z coordinate of simulated volume (if None it is set to 3x the fiducial volume)
    full_zmax: float (default None)
        upper z coordinate of simulated volume (if None it is set to 3x the fiducial volume)
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
        * 'E-1': 1 over E spectrum
    deposited: bool
        If True, generate deposited energies instead of primary neutrino energies
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
        full_rmin = fiducial_rmin / 3.
    if(full_rmax is None):
        full_rmax = fiducial_rmax * 5.
    if(full_zmin is None):
        full_zmin = fiducial_zmin * 5.
    if(full_zmax is None):
        full_zmax = fiducial_zmax / 3.

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
    rr_full = np.random.triangular(full_rmin, full_rmax, full_rmax, n_events)
    phiphi = np.random.uniform(0, 2 * np.pi, n_events)
    data_sets["xx"] = rr_full * np.cos(phiphi)
    data_sets["yy"] = rr_full * np.sin(phiphi)
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
    else:
        logger.error("spectrum {} not implemented".format(spectrum))
        raise NotImplementedError("spectrum {} not implemented".format(spectrum))

    # generate charged/neutral current randomly (ported from ShelfMC)
    rnd = np.random.uniform(0., 1., n_events)
    data_sets["ccncs"] = np.ones(n_events, dtype='S2')
    for i, r in enumerate(rnd):
        #    if (r <= 0.6865254):#from AraSim
        if(r <= 0.7064):
            data_sets["ccncs"][i] = 'cc'
        else:
            data_sets["ccncs"][i] = 'nc'

    # generate neutrino direction randomly
    data_sets["azimuths"] = np.random.uniform(phimin, phimax, n_events)
    u = np.random.uniform(np.cos(thetamax), np.cos(thetamin), n_events)
    data_sets["zeniths"] = np.arccos(u)  # generates distribution that is uniform in cos(theta)

    # generate inelasticity (ported from ShelfMC)
    R1 = 0.36787944
    R2 = 0.63212056
    data_sets["inelasticity"] = (-np.log(R1 + np.random.uniform(0., 1., n_events) * R2)) ** 2.5
    """
    #from AraSim
    epsilon = np.log10(energies / 1e9)
    inelasticity = pickY(flavors, ccncs, epsilon)
    """
    if deposited:
        data_sets["energies"] = [primary_energy_from_deposited(Edep, ccnc, flavor, inelasticity) \
                                for Edep, ccnc, flavor, inelasticity in \
                                zip(data_sets["energies"], data_sets["ccncs"], \
                                data_sets["flavors"], data_sets["inelasticity"])]

    # save only events with interactions in fiducial volume
    data_sets_fiducial = {}
    for key, value in iteritems(data_sets):
        data_sets_fiducial[key] = value[fmask]

    if add_tau_second_bang:
        mask = (data_sets["ccncs"] == 'cc') & (np.abs(data_sets["flavors"]) == 16)  # select nu_tau cc interactions
        logger.info("{} taus are created in nu tau interactions -> checking if tau decays in fiducial volume".format(np.sum(mask)))
        n_taus = 0
        for event_id in data_sets["event_ids"][mask]:
            iE = event_id - start_event_id

            Etau = (1 - data_sets["inelasticity"][iE]) * data_sets["energies"][iE]
            # first calculate if tau decay is still in our fiducial volume
            x, y, z = get_tau_decay_vertex(data_sets["xx"][iE], data_sets["yy"][iE], data_sets["zz"][iE],
                                           Etau, data_sets["zeniths"][iE], data_sets["azimuths"][iE])
            logger.debug("tau energy = {:.2g}eV, decay length = {:.2f}km -> decay at {:.2f}, {:.2f}, {:.2f}".format(Etau/units.eV,
                                                            get_tau_decay_length(Etau)/units.km, x/units.km, y/units.km, z/units.km))

            r = (x ** 2 + y ** 2)**0.5
            if(r >= fiducial_rmin and r <= fiducial_rmax ):
                if(z >= fiducial_zmin and z <= fiducial_zmax):  # z coordinate is negative
                    # the tau decay is in our fiducial volume

                    n_taus += 1  # we change the datasets during the loop, to still have the correct indices, we need to keep track of the number of events we inserted

                    # insert second vertex after the first neutrino interaction
                    # two possible cases
                    # 1) first interaction is not in fiducial volume -> insert event such that event ids are increasing
                    # 2) first interaction is in fiducial volume -> find correct index
                    if(event_id in data_sets['event_ids']):  # case 2
                        iE2 = np.squeeze(np.argwhere(data_sets_fiducial['event_ids'] == event_id))
                    else:  # case 1
                        iE2 = np.squeeze(np.argwhere(data_sets_fiducial['event_ids'] < event_id))[-1]
                    for key in iterkeys(data_sets):
                        data_sets_fiducial[key] = np.insert(data_sets_fiducial[key], iE2, data_sets[key][iE])
                    iE2 += 1
                    data_sets_fiducial['n_interaction'][iE2] = 2  # specify that new event is a second interaction

                    # Calculating the energy of the tau from the neutrino energy
                    data_sets_fiducial['energies'][iE2] = Etau
                    data_sets_fiducial['xx'][iE2] = x
                    data_sets_fiducial['yy'][iE2] = y
                    data_sets_fiducial['zz'][iE2] = z

                    # set flavor to tau
                    data_sets_fiducial['flavors'][iE2] = 15 * np.sign(data_sets_fiducial['flavors'][iE2])  # keep particle/anti particle nature
        print("added {} tau decays to the event list".format(n_taus))
   write_events_to_hdf5(filename, data_sets, attributes, n_events_per_file=n_events_per_file)


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
