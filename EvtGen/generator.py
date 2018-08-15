from __future__ import absolute_import, division, print_function
import numpy as np
from NuRadioMC.utilities import units
import h5py

VERSION_MAJOR = 1
VERSION_MINOR = 0

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


def generate_eventlist_cuboid(filename, n_events, Emin, Emax,
                       xmin, xmax, ymin, ymax, zmin, zmax,
                       start_event_id=1,
                       flavor=[12, -12, 14, -14, 16, -16],
                       n_events_per_file=None):
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
    xmin: float
        lower x coordinate of simulated volume
    xmax: float
        upper x coordinate of simulated volume
    ymin: float
        lower y coordinate of simulated volume
    ymax: float
        upper y coordinate of simulated volume
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

    """
    n_events = int(n_events)
    event_ids = np.arange(n_events) + start_event_id

    # generate neutrino flavors randomly
    flavors = np.array([flavor[i] for i in np.random.randint(0, high=len(flavor), size=n_events)])
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
    energies = 10 ** np.random.uniform(np.log10(Emin), np.log10(Emax), n_events)

    # generate charged/neutral current randomly (ported from ShelfMC)
    rnd = np.random.uniform(0., 1., n_events)
    ccncs = np.ones(n_events, dtype='S2')
    for i, r in enumerate(rnd):
#        if (r <= 0.6865254):#from AraSim
        if(r <= 0.7064):
            ccncs[i] = 'cc'
        else:
            ccncs[i] = 'nc'

    # generate neutrino vertices randomly
    xx = np.random.uniform(xmin, xmax, n_events)
    yy = np.random.uniform(ymin, ymax, n_events)
    zz = np.random.uniform(zmin, zmax, n_events)

    # generate neutrino direction randomly
    azimuths = np.random.uniform(0, 360 * units.deg, n_events)
    u = np.random.uniform(-1, 1, n_events)
    zeniths = np.arccos(u)  # generates distribution that is uniform in cos(theta)

    # generate inelasticity (ported from ShelfMC)
    R1 = 0.36787944
    R2 = 0.63212056
    inelasticity = (-np.log(R1 + np.random.uniform(0., 1., n_events) * R2)) ** 2.5

    if(n_events_per_file is None):
        n_events_per_file = n_events
    else:
        n_events_per_file = int(n_events_per_file)
    for iFile in range(np.int(np.ceil(n_events / n_events_per_file))):
        filename2 = filename
        if((iFile > 0) or (n_events_per_file < n_events)):
            filename2 = filename + ".part{:04}".format(iFile + 1)
        fout = h5py.File(filename2, 'w')
        fout.attrs['VERSION_MAJOR'] = VERSION_MAJOR
        fout.attrs['VERSION_MINOR'] = VERSION_MINOR
        fout.attrs['header'] = HEADER

        fout.attrs['xmin'] = xmin
        fout.attrs['xmax'] = xmax
        fout.attrs['ymin'] = ymin
        fout.attrs['ymax'] = ymax
        fout.attrs['zmin'] = zmin
        fout.attrs['zmax'] = zmax
        fout.attrs['flavors'] = flavor
        fout.attrs['Emin'] = Emin
        fout.attrs['Emax'] = Emax

        fout['event_ids'] = event_ids[iFile * n_events_per_file:(iFile + 1) * n_events_per_file]
        fout['flavors'] = flavors[iFile * n_events_per_file:(iFile + 1) * n_events_per_file]
        fout['energies'] = energies[iFile * n_events_per_file:(iFile + 1) * n_events_per_file]
        fout['ccncs'] = ccncs[iFile * n_events_per_file:(iFile + 1) * n_events_per_file]
        fout['xx'] = xx[iFile * n_events_per_file:(iFile + 1) * n_events_per_file]
        fout['yy'] = yy[iFile * n_events_per_file:(iFile + 1) * n_events_per_file]
        fout['zz'] = zz[iFile * n_events_per_file:(iFile + 1) * n_events_per_file]
        fout['zeniths'] = zeniths[iFile * n_events_per_file:(iFile + 1) * n_events_per_file]
        fout['azimuths'] = azimuths[iFile * n_events_per_file:(iFile + 1) * n_events_per_file]
        fout['inelasticity'] = inelasticity[iFile * n_events_per_file:(iFile + 1) * n_events_per_file]
        fout.close()

def generate_eventlist_cylinder(filename, n_events, Emin, Emax,
                       rmin, rmax, zmin, zmax,
                       start_event_id=1,
                       flavor=[12, -12, 14, -14, 16, -16],
                       n_events_per_file=None):
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

    """
    n_events = int(n_events)
    event_ids = np.arange(n_events) + start_event_id

    # generate neutrino flavors randomly
    flavors = np.array([flavor[i] for i in np.random.randint(0, high=len(flavor), size=n_events)])
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
    energies = 10 ** np.random.uniform(np.log10(Emin), np.log10(Emax), n_events)

    # generate charged/neutral current randomly (ported from ShelfMC)
    rnd = np.random.uniform(0., 1., n_events)
    ccncs = np.ones(n_events, dtype='S2')
    for i, r in enumerate(rnd):
    #    if (r <= 0.6865254):#from AraSim
        if(r <= 0.7064):
            ccncs[i] = 'cc'
        else:
            ccncs[i] = 'nc'

    # generate neutrino vertices randomly
    rr = np.random.triangular(rmin, rmax, rmax, n_events)
    phiphi = np.random.uniform(0, 2 * np.pi, n_events)
    xx = rr * np.cos(phiphi)
    yy = rr * np.sin(phiphi)
    zz = np.random.uniform(zmin, zmax, n_events)

    # generate neutrino direction randomly
    azimuths = np.random.uniform(0, 360 * units.deg, n_events)
    u = np.random.uniform(-1, 1, n_events)
    zeniths = np.arccos(u)  # generates distribution that is uniform in cos(theta)

    # generate inelasticity (ported from ShelfMC)
    R1 = 0.36787944
    R2 = 0.63212056
    inelasticity = (-np.log(R1 + np.random.uniform(0., 1., n_events) * R2)) ** 2.5
    """    
    #from AraSim
    epsilon = np.log10(energies / 1e9)
    inelasticity = pickY(flavors, ccncs, epsilon)
   """ 
    if(n_events_per_file is None):
        n_events_per_file = n_events
    else:
        n_events_per_file = int(n_events_per_file)
    for iFile in range(np.int(np.ceil(n_events / n_events_per_file))):
        filename2 = filename
        if((iFile > 0) or (n_events_per_file < n_events)):
            filename2 = filename + ".part{:04}".format(iFile + 1)
        fout = h5py.File(filename2, 'w')
        fout.attrs['VERSION_MAJOR'] = VERSION_MAJOR
        fout.attrs['VERSION_MINOR'] = VERSION_MINOR
        fout.attrs['header'] = HEADER

        fout.attrs['xmin'] = rmin
        fout.attrs['xmax'] = rmax
        fout.attrs['ymin'] = rmin
        fout.attrs['ymax'] = rmax
        fout.attrs['zmin'] = zmin
        fout.attrs['zmax'] = zmax
        fout.attrs['flavors'] = flavor
        fout.attrs['Emin'] = Emin
        fout.attrs['Emax'] = Emax

        fout['event_ids'] = event_ids[iFile * n_events_per_file:(iFile + 1) * n_events_per_file]
        fout['flavors'] = flavors[iFile * n_events_per_file:(iFile + 1) * n_events_per_file]
        fout['energies'] = energies[iFile * n_events_per_file:(iFile + 1) * n_events_per_file]
        fout['ccncs'] = ccncs[iFile * n_events_per_file:(iFile + 1) * n_events_per_file]
        fout['xx'] = xx[iFile * n_events_per_file:(iFile + 1) * n_events_per_file]
        fout['yy'] = yy[iFile * n_events_per_file:(iFile + 1) * n_events_per_file]
        fout['zz'] = zz[iFile * n_events_per_file:(iFile + 1) * n_events_per_file]
        fout['zeniths'] = zeniths[iFile * n_events_per_file:(iFile + 1) * n_events_per_file]
        fout['azimuths'] = azimuths[iFile * n_events_per_file:(iFile + 1) * n_events_per_file]
        fout['inelasticity'] = inelasticity[iFile * n_events_per_file:(iFile + 1) * n_events_per_file]
        fout.close()


if __name__ == '__main__':
    # define simulation volume
    xmin = -3 * units.km
    xmax = 3 * units.km
    ymin = -3 * units.km
    ymax = 3 * units.km
    zmin = -2.7 * units.km
    zmax = 0 * units.km
    generate_eventlist_cuboid('1e19.hdf5', 1e5, 1e19 * units.eV, 1e19 * units.eV,
                       xmin, xmax, ymin, ymax, zmin, zmax)

