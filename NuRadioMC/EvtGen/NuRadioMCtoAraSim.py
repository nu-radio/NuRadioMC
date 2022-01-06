from __future__ import absolute_import, division, print_function
import numpy as np
from NuRadioReco.utilities import units
import argparse
import h5py

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run NuRadioMC simulation')
    parser.add_argument('hdf5input', type=str,
                        help='path to NuRadioMC hdf5 input event list')
    parser.add_argument('AraSiminput', type=str,
                        help='path to AraSim input file')
    parser.add_argument('avgDepth', type=float,
                        help='average depth of all antennas in meters (use positive numbers)')
    args = parser.parse_args()
    
    fin = h5py.File(args.hdf5input, 'r')

    #AraStationDep = 178.5598 / units.m
    AraStationDep = args.avgDepth / units.m
    event_ids = fin['event_ids']
    flavors = np.array(fin['flavors'])
    energies = np.array(fin['energies'])
    ccncs = np.array(fin['interaction_type'])
    xx = np.array(fin['xx'])
    yy = np.array(fin['yy'])
    zz = np.array(fin['zz'])
    zeniths = np.array(fin['zeniths'])
    azimuths = np.array(fin['azimuths'])
    inelasticity = np.array(fin['inelasticity'])
    n_events = len(event_ids)

    nuflavorint = (np.absolute(flavors) - 10) / 2
    nu_nubar = (-1 * np.sign(flavors) + 1) / 2
    pnu = np.log10(energies)
    currentint = np.ones(n_events)
    posnu_r = np.sqrt((xx) ** 2 + (yy) ** 2 + (-1.0 * zz - AraStationDep) ** 2)
    posnu_phi = np.ones(n_events)
    posnu_theta = np.ones(n_events)
    print("AraStationDep = " + str(AraStationDep))
    for i in range(n_events):
        if (ccncs[i] == "cc"):
            currentint[i] = 1
        else:
            currentint[i] = 0
        if (yy[i] >= 0):
            posnu_phi[i] = np.arccos(xx[i] / np.sqrt((xx[i]) ** 2 + (yy[i]) ** 2))
        else:
            posnu_phi[i] = -1.0 * np.arccos(xx[i] / np.sqrt(xx[i] ** 2 + yy[i] ** 2)) + 2.0 * np.pi
        if (-1.0 * zz[i] >= AraStationDep):
            posnu_theta[i] = -1.0 * np.arccos(np.sqrt(xx[i] ** 2 + yy[i] ** 2) / posnu_r[i])
        else:
            posnu_theta[i] = np.arccos(np.sqrt(xx[i] ** 2 + yy[i] ** 2) / posnu_r[i])

    with open(args.AraSiminput, 'w') as fout:
        fout.write("//VERSION=0.1\n//EVENT_NUM=" + str(n_events) + "\n//evid nuflavorint nu_nubar pnu currentint posnu_r posnu_theta posnu_phi nnu_theta nnu_phi elast_y\n")
    #    fout.write("//VERSION=0.1\n//EVENT_NUM=" + str(n_events) + "\n//evid nuflavorint nu_nubar pnu currentint posnu_r posnu_theta posnu_phi nnu_theta nnu_phi\n")
        for i in range(n_events):
    #        fout.write("{:08d} {:>+5d}  {:.5e}  {:s}  {:>10.3f}  {:>10.3f}  {:>10.3f}  {:>10.3f}  {:>10.3f}  {:>10.3f}\n".format(event_ids[i], flavors[i], energies[i], ccncs[i], xx[i], yy[i], zz[i], zeniths[i], azimuths[i], inelasticity[i]))
            fout.write("{:08d} {:01d} {:01d} {:.3f} {:01d} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(event_ids[i], int(nuflavorint[i]), int(nu_nubar[i]), pnu[i], int(currentint[i]), posnu_r[i], posnu_theta[i], posnu_phi[i], np.pi - zeniths[i], np.pi + azimuths[i], inelasticity[i]))
    #        fout.write("{:08d} {:01d} {:01d} {:.3f} {:01d} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(event_ids[i], int(nuflavorint[i]), int(nu_nubar[i]), pnu[i], int(currentint[i]), posnu_r[i], posnu_theta[i], posnu_phi[i], np.pi - zeniths[i], np.pi + azimuths[i]))
        fout.close()
