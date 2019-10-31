from __future__ import absolute_import, division, print_function
import numpy as np
from NuRadioReco.utilities import units

# define simulation volume
xmin = -3 * units.km
xmax = 3 * units.km
ymin = -3 * units.km
ymax = 3 * units.km
fiducial_zmin = -3 * units.km
fiducial_zmax = 0 * units.km

HEADER = """VERSION=0.2
# standard event list format of NuRadioMC (adapted from ARASim)
# the very first line defines the file version, all other lines represent the events
# each row specifies one event
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
# 8. zenith/theta angle of neutrino direction (pointing into direction of propagation)
# 9. azimuth/phi angle of neutrino direction (pointing into direction of propagation)
# 10. inelasticity (the fraction of neutrino energy that goes into the hadronic part)
#
"""


def generate_eventlist(filename, n_events, Emin, Emax,
                       start_event_id=1,
                       flavor=[12, -12, 14, -14, 16, -16]):
    n_events = int(n_events)
    event_ids = np.arange(n_events) + start_event_id

    # generate neutrino flavors randomly
    flavors = np.array([flavor[i] for i in np.random.randint(0, high=len(flavor), size=n_events)])
    # generate energies randomly
    energies = 10 ** np.random.uniform(np.log10(Emin), np.log10(Emax), n_events)

    # generate charged/neutral current randomly (ported from ShelfMC)
    rnd = np.random.uniform(0., 1., n_events)
    ccncs = np.ones(n_events, dtype='S2')
    for i, r in enumerate(rnd):
        if(r <= 0.7064):
            ccncs[i] = 'cc'
        else:
            ccncs[i] = 'nc'

    # generate neutrino vertices randomly
    xx = np.random.uniform(xmin, xmax, n_events)
    yy = np.random.uniform(ymin, ymax, n_events)
    zz = np.random.uniform(fiducial_zmin, fiducial_zmax, n_events)

    # generate neutrino direction randomly
    azimuths = np.random.uniform(0, 360 * units.deg, n_events)
    u = np.random.uniform(-1, 1, n_events)
    zeniths = np.arccos(u)  # generates distribution that is uniform in cos(theta)

    # generate inelasticity (ported from ShelfMC)
    R1 = 0.36787944
    R2 = 0.63212056
    inelasticity = (-np.log(R1 + np.random.uniform(0., 1., n_events) * R2)) ** 2.5

    with open(filename, 'w') as fout:
        fout.write(HEADER)
        for i in range(n_events):
            fout.write("{:08d} {:>+5d}  {:.5e}  {:s}  {:>10.3f}  {:>10.3f}  {:>10.3f}  {:>10.3f}  {:>10.3f}  {:>10.3f}\n".format(event_ids[i], flavors[i], energies[i], ccncs[i], xx[i], yy[i], zz[i], zeniths[i], azimuths[i], inelasticity[i]))
        fout.close()


if __name__ == '__main__':
    generate_eventlist('test.txt', 1e5, 1e18, 1e18)
