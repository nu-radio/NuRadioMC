from __future__ import absolute_import, division, print_function
import numpy as np
from NuRadioMC.utilities import units
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
# 2. neutrino flavor (integer) encoded as using PDF numbering scheme, particles have positive sign, anti-particles have negative sign, relevant for us are:
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
    """
    n_events = len(data_sets.values()[0])
    total_number_of_events = n_events
    if('n_events' in attributes):
        total_number_of_events = attributes['n_events']
    if(n_events_per_file is None):
        n_events_per_file = n_events
    else:
        n_events_per_file = int(n_events_per_file)
    for iFile in range(np.int(np.ceil(n_events / n_events_per_file))):
        filename2 = filename
        if((iFile > 0) or (n_events_per_file < n_events)):
            filename2 = filename + ".part{:04}".format(iFile + 1)
        print('writing file {}'.format(filename2))
        fout = h5py.File(filename2, 'w')
        fout.attrs['VERSION_MAJOR'] = VERSION_MAJOR
        fout.attrs['VERSION_MINOR'] = VERSION_MINOR
        fout.attrs['header'] = HEADER
        for key, value in attributes.iteritems():
            fout.attrs[key] = value
        fout.attrs['total_number_of_events'] = total_number_of_events

        for key, value in data_sets.iteritems():
            fout[key] = value[iFile * n_events_per_file:(iFile + 1) * n_events_per_file]
        fout.attrs['n_events'] = len(fout[data_sets.keys()[0]])

        fout.close()

def write_events_to_hdf5_new(filename, data_sets, attributes, n_events_per_file=None):
    """
    writes NuRadioMC input parameters to the new style of hdf5 file
    
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
    """
    n_events = len(data_sets.values()[0])
    total_number_of_events = n_events
    if('n_events' in attributes):
        total_number_of_events = attributes['n_events']
    if(n_events_per_file is None):
        n_events_per_file = n_events
    else:
        n_events_per_file = int(n_events_per_file)
    for iFile in range(np.int(np.ceil(n_events / n_events_per_file))):
        filename2 = filename
        if((iFile > 0) or (n_events_per_file < n_events)):
            filename2 = filename + ".part{:04}".format(iFile + 1)
        print('writing file {}'.format(filename2))
        fout = h5py.File(filename2, 'w')
        fout.attrs['VERSION_MAJOR'] = VERSION_MAJOR
        fout.attrs['VERSION_MINOR'] = VERSION_MINOR
        fout.attrs['header'] = HEADER
        for key, value in attributes.iteritems():
            fout.attrs[key] = value
        fout.attrs['total_number_of_events'] = total_number_of_events

        comp_type = np.dtype([('azimuths', 'f8'), ('ccncs', np.str_, 2), ('energies', 'f8'), ('event_ids', 'i4'), ('flavors', 'i4'), ('inelasticity', 'f8'), ('xx', 'f8'), ('yy', 'f8'), ('zeniths', 'f8'), ('zz', 'f8')])
        dataset = fout.create_dataset("Event_input", (n_events_per_file, ), comp_type)
        for key in data_sets:
            dataset[key] = data_sets[key][iFile * n_events_per_file:(iFile + 1) * n_events_per_file]
        fout.attrs['n_events'] = len(dataset)

        fout.close()



def generate_eventlist_cylinder(filename, n_events, Emin, Emax,
                                rmin, rmax, zmin, zmax,
                                thetamin=0.*units.rad, thetamax=np.pi*units.rad,
                                phimin=0.*units.rad, phimax=2*np.pi*units.rad, 
                                start_event_id=1,
                                flavor=[12, -12, 14, -14, 16, -16],
                                n_events_per_file=None,
                                spectrum='log_uniform'):
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
    generate_eventlist_cylinder('1e19.hdf5', 1e6, 1e19 * units.eV, 1e19 * units.eV,
                                0, 3*units.km, zmin, zmax)
