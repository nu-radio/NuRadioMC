from __future__ import absolute_import, division, print_function
import numpy as np
from NuRadioMC.utilities import units
from radiotools import helper as hp
from radiotools import plthelpers as php
import argparse
import h5py

parser = argparse.ArgumentParser(description='Run NuRadioMC simulation')
parser.add_argument('hdf5input', type=str,
                    help='path to NuRadioMC hdf5 input event list')
parser.add_argument('ASCIIoutput', type=str,
                    help='path to ASCII output file')
args = parser.parse_args()

fin = h5py.File(args.hdf5input, 'r')

event_ids = fin['event_ids']
flavors = fin['flavors']
energies = fin['energies']
ccncs = fin['ccncs']
xx = fin['xx']
yy = fin['yy']
zz = fin['zz']
zeniths = fin['zeniths']
azimuths = fin['azimuths']
inelasticity = fin['inelasticity']
weight = fin['weights']
shower_axis = -1.0 * hp.spherical_to_cartesian(np.array(fin['zeniths']), np.array(fin['azimuths']))
launch_vectors = np.array(fin['launch_vectors'])
viewing_angles_d = np.array([hp.get_angle(x, y) for x, y in zip(shower_axis, launch_vectors[:, 0, 0])])
viewing_angles_r = np.array([hp.get_angle(x, y) for x, y in zip(shower_axis, launch_vectors[:, 0, 1])])
launch_vectors = np.array(fin['launch_vectors'])
launch_angles_d, launch_azimuths_d = hp.cartesian_to_spherical_vectorized(launch_vectors[:, 0, 0, 0].flatten(), launch_vectors[:, 0, 0, 1].flatten(), launch_vectors[:, 0, 0, 2].flatten())
launch_angles_r, launch_azimuths_r = hp.cartesian_to_spherical_vectorized(launch_vectors[:, 0, 1, 0].flatten(), launch_vectors[:, 0, 1, 1].flatten(), launch_vectors[:, 0, 1, 2].flatten())
receive_vectors = np.array(fin['receive_vectors'])
rec_angles_d, rec_azimuths_d = hp.cartesian_to_spherical_vectorized(receive_vectors[:, 0, 0, 0].flatten(), receive_vectors[:, 0, 0, 1].flatten(), receive_vectors[:, 0, 0, 2].flatten())
rec_angles_r, rec_azimuths_r = hp.cartesian_to_spherical_vectorized(receive_vectors[:, 0, 1, 0].flatten(), receive_vectors[:, 0, 1, 1].flatten(), receive_vectors[:, 0, 1, 2].flatten())


n_events = len(event_ids)

with open(args.ASCIIoutput, 'w') as fout:
    fout.write(fin.attrs['header'])
    fout.write("event_ids flavors energies ccnc xx yy zz zeniths azimuths inelasticity weight\n")
    fout.write("viewing_angles_d viewing_angles_r launch_angles_d launch_angles_r rec_angles_d rec_angles_r\n")
    for i in range(n_events):
        fout.write("{:08d}  {:>+5d}  {:.5e}  {:s}  {:>10.3f}  {:>10.3f}  {:>10.3f}  {:>10.3f}  {:>10.3f}  {:>10.3f}  {:>10.3f}\n{:>10.3f}  {:>10.3f}  {:>10.3f}  {:>10.3f}  {:>10.3f}  {:>10.3f}\n".format(event_ids[i], flavors[i], energies[i], ccncs[i], xx[i], yy[i], zz[i], zeniths[i], azimuths[i], inelasticity[i], weight[i], viewing_angles_d[i] / units.deg, viewing_angles_r[i] / units.deg, launch_angles_d[i] / units.deg, launch_angles_r[i] / units.deg, 180.0 - rec_angles_d[i] / units.deg, 180.0 - rec_angles_r[i] / units.deg))
    fout.close()


