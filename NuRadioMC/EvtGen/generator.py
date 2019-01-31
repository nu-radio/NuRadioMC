from __future__ import absolute_import, division, print_function
import numpy as np
from NuRadioMC.utilities import units
from six import iterkeys
from scipy import constants
import h5py

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

    gamma = energy/tau_mass # tau_mass must be in natural units (c = 1)
    tau_mean_time = gamma * tau_rest_time

    # The tau decay time is taken assuming an exponential decay
    # and applying the inverse transform method
    tau_decay_time = -np.log(1 - np.random.uniform(0,1)) * tau_mean_time

    return tau_decay_time

def get_tau_speed(energy):
    """
    Calculates the speed of the tau lepton
    """

    gamma = energy/tau_mass
    if (gamma < 1):
        raise ValueError('The energy is less than the tau mass. Returning zero speed')
        return 0
    beta = np.sqrt(1 - 1/gamma**2)

    return beta * constants.c*units.m/units.s

def get_tau_vertex(data_sets, iE):

    decay_time = get_tau_decay_time(data_sets['energies'][iE])
    tau_speed = get_tau_speed(data_sets['energies'][iE])

    # Let us assume that the tau has the same direction as the tau neutrino
    # to calculate the vertex of the second shower
    # This must be changed in the future

    second_vertex_x  = tau_speed * decay_time
    second_vertex_x *= np.sin(data_sets['zeniths'][iE]) * np.cos(data_sets['azimuths'][iE])
    second_vertex_x += data_sets['xx'][iE]
    data_sets['xx'][iE] = second_vertex_x

    second_vertex_y  = tau_speed * decay_time
    second_vertex_y *= np.sin(data_sets['zeniths'][iE]) * np.sin(data_sets['azimuths'][iE])
    second_vertex_y += data_sets['yy'][iE]
    data_sets['yy'][iE] = second_vertex_y

    second_vertex_z  = tau_speed * decay_time
    second_vertex_z *= np.cos(data_sets['zeniths'][iE])
    second_vertex_z += data_sets['zz'][iE]
    data_sets['zz'][iE] = second_vertex_z

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
    total_number_of_events = n_events
    if('n_events' in attributes):
        total_number_of_events = attributes['n_events']
    if(n_events_per_file is None):
        n_events_per_file = n_events
    else:
        n_events_per_file = int(n_events_per_file)
    iFile = -1
    evt_id_first = data_sets['event_ids'][0]
    start_index = 0
    while True:
        iFile += 1
        filename2 = filename
        if((iFile > 0) or (n_events_per_file < n_events)):
            filename2 = filename + ".part{:04}".format(iFile + 1)
        fout = h5py.File(filename2, 'w')
        fout.attrs['VERSION_MAJOR'] = VERSION_MAJOR
        fout.attrs['VERSION_MINOR'] = VERSION_MINOR
        fout.attrs['header'] = HEADER
        for key, value in attributes.iteritems():
            fout.attrs[key] = value
        fout.attrs['total_number_of_events'] = total_number_of_events

        evt_id_last = evt_id_first + n_events_per_file - 1

        if(evt_id_last >= n_events):
            evt_id_last = n_events
            stop_index = len(data_sets['event_ids'])
        else:
            tmp = np.squeeze(np.argwhere(data_sets['event_ids'] > evt_id_last)) # set stop index such that last event is competely in file
            if(tmp.size == 1):
                stop_index = tmp
            else:
                stop_index = tmp[0]

        print('writing file {} with {} events'.format(filename2, 1 + evt_id_last - evt_id_first))

        for key, value in data_sets.iteritems():
            fout[key] = value[start_index:stop_index]

        fout.attrs['n_events'] = len(np.unique(np.array(fout['event_ids'])))
        fout.close()

        start_index = stop_index
        evt_id_first = evt_id_last + 1
        if(evt_id_last == n_events):  # break while loop if all events are saved
            break



def generate_eventlist_cylinder(filename, n_events, Emin, Emax,
                                rmin, rmax, zmin, zmax,
                                start_event_id=1,
                                flavor=[12, -12, 14, -14, 16, -16],
                                n_events_per_file=None,
                                spectrum='log_uniform',
                                addTauSecondBang=False):
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
    rmin: float
        lower r coordinate of simulated volume
    rmax: float
        upper r coordinate of simulated volume
    zmin: float
        lower z coordinate of simulated volume
    zmax: float
        upper z coordinate of simulated volume
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

    """
    attributes = {}
    attributes['rmin'] = rmin
    attributes['rmax'] = rmax
    attributes['zmin'] = zmin
    attributes['zmax'] = zmax
    attributes['flavors'] = flavor
    attributes['Emin'] = Emin
    attributes['Emax'] = Emax
    data_sets = {}

    n_events = int(n_events)
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
    elif(spectrum == 'E-1'):
        pass
    else:
#         logger.error("spectrum {} not implemented".format(spectrum))
        raise NotImplementedError("spectrum {} not implemented".format(spectrum))

    # generate charged/neutral current randomly (ported from ShelfMC)
    rnd = np.random.uniform(0., 1., n_events)
    ccncs = np.ones(n_events, dtype='S2')
    for i, r in enumerate(rnd):
        #    if (r <= 0.6865254):#from AraSim
        if(r <= 0.7064):
            ccncs[i] = 'cc'
        else:
            ccncs[i] = 'nc'
    data_sets["ccncs"] = ccncs

    # generate neutrino vertices randomly
    rr = np.random.triangular(rmin, rmax, rmax, n_events)
    phiphi = np.random.uniform(0, 2 * np.pi, n_events)
    data_sets["xx"] = rr * np.cos(phiphi)
    data_sets["yy"] = rr * np.sin(phiphi)
    data_sets["zz"] = np.random.uniform(zmin, zmax, n_events)

    # generate neutrino direction randomly
    data_sets["azimuths"] = np.random.uniform(0, 360 * units.deg, n_events)
    u = np.random.uniform(-1, 1, n_events)
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

    if addTauSecondBang:
        mask = (data_sets['ccncs'] == 'cc') & (np.abs(data_sets['flavors']) == 16)  # select nu_tau cc interactions
        n_taus = 0
        for event_id in data_sets['event_ids'][mask]:
            iE = event_id - start_event_id + n_taus
            n_taus += 1  # we change the datasets during the loop, to still have the correct indices, we need to keep track of the number of events we inserted

            # insert second vertex after the first neutrino interaction
            for key in iterkeys(data_sets):
                data_sets[key] = np.insert(data_sets[key], iE, data_sets[key][iE])
            iE += 1
            data_sets['n_interaction'][iE] = 2  # specify that new event is a second interaction

            # Calculating the energy of the tau from the neutrino energy
            data_sets['energies'][iE] = (1-data_sets['inelasticity'][iE-1])*data_sets['energies'][iE-1]

            decay_time = get_tau_decay_time(data_sets['energies'][iE])

            # Let us assume that the tau has the same direction as the tau neutrino
            # to calculate the vertex of the second shower
            # This must be changed in the future

            second_vertex_x  = get_tau_speed(data_sets['energies'][iE]) * decay_time
            second_vertex_x *= np.sin(data_sets['zeniths'][iE]) * np.cos(data_sets['azimuths'][iE])
            second_vertex_x += data_sets['xx'][iE]
            data_sets['xx'][iE] = second_vertex_x

            second_vertex_y  = get_tau_speed(data_sets['energies'][iE]) * decay_time
            second_vertex_y *= np.sin(data_sets['zeniths'][iE]) * np.sin(data_sets['azimuths'][iE])
            second_vertex_y += data_sets['yy'][iE]
            data_sets['yy'][iE] = second_vertex_y

            # Calculation of the tau decay vertex
            get_tau_vertex(data_sets, iE)

            # set flavor to tau
            data_sets['flavors'][iE] = 15 * np.sign(data_sets['flavors'][iE])  # keep particle/anti particle nature

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
        if isinstance(value, h5py.Dataset): # the loop is also over potential subgroupu that we don't want to consider here
            data_sets[key] = np.array(value)
    for key, value in fin.attrs.items():
        attributes[key] = value

    fin.close()

    write_events_to_hdf5(output_filename, data_sets, attributes, n_events_per_file=number_of_events_per_file)



if __name__ == '__main__':
    # define simulation volume
    xmin = -3 * units.km
    xmax = 3 * units.km
    ymin = -3 * units.km
    ymax = 3 * units.km
    zmin = -2.7 * units.km
    zmax = 0 * units.km
    generate_eventlist_cylinder('1e19.hdf5', 1e3, 1e19 * units.eV, 1e19 * units.eV,
                                0, 3*units.km, zmin, zmax)
