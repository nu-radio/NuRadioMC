from __future__ import absolute_import, division, print_function
import numpy as np
from NuRadioReco.utilities import units
from radiotools import helper as hp
from radiotools import plthelpers as php
import argparse
import h5py

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run NuRadioMC simulation')
    parser.add_argument('ASCIIoutput', type=str, help='path to ASCII output file')
    parser.add_argument('hdf5input', type=str, nargs = '+', help='path to NuRadioMC hdf5 input event list')
    args = parser.parse_args()

    fin = h5py.File(args.hdf5input[0], 'r')
    event_ids = np.array(fin['event_ids'])
    flavors = np.array(fin['flavors'])
    energies = np.array(fin['energies'])
    ccncs = np.array(fin['interaction_type'])
    xx = np.array(fin['xx'])
    yy = np.array(fin['yy'])
    zz = np.array(fin['zz'])
    zeniths = np.array(fin['zeniths'])
    azimuths = np.array(fin['azimuths'])
    inelasticity = np.array(fin['inelasticity'])
    weight = np.array(fin['weights'])
    st = np.array(fin['ray_tracing_solution_type'])
    launch_vectors = np.array(fin['launch_vectors'])
    receive_vectors = np.array(fin['receive_vectors'])

    for i in range(len(args.hdf5input) - 2):
        fin = h5py.File(args.hdf5input[i + 1], 'r')
        event_ids = np.append(event_ids, 10000 * (i + 1) + np.array(fin['event_ids']))
        flavors = np.append(flavors, np.array(fin['flavors']))
        energies = np.append(energies, np.array(fin['energies']))
        ccncs = np.append(ccncs, np.array(fin['ccncs']))
        xx = np.append(xx, np.array(fin['xx']))
        yy = np.append(yy, np.array(fin['yy']))
        zz = np.append(zz, np.array(fin['zz']))
        zeniths = np.append(zeniths, np.array(fin['zeniths']))
        azimuths = np.append(azimuths, np.array(fin['azimuths']))
        inelasticity = np.append(inelasticity, np.array(fin['inelasticity']))
        weight = np.append(weight, np.array(fin['weights']))
        st = np.append(st, np.array(fin['ray_tracing_solution_type']), axis = 0)
        launch_vectors = np.append(launch_vectors, np.array(fin['launch_vectors']), axis = 0)
        receive_vectors = np.append(receive_vectors, np.array(fin['receive_vectors']), axis = 0)

    shower_axis = -1.0 * hp.spherical_to_cartesian(zeniths, azimuths)
    viewing_angles_d = np.array([hp.get_angle(x, y) for x, y in zip(shower_axis, launch_vectors[:, 5, 0])])
    viewing_angles_r = np.array([hp.get_angle(x, y) for x, y in zip(shower_axis, launch_vectors[:, 5, 1])])
    launch_angles_d, launch_azimuths_d = hp.cartesian_to_spherical_vectorized(launch_vectors[:, 5, 0, 0].flatten(), launch_vectors[:, 5, 0, 1].flatten(), launch_vectors[:, 5, 0, 2].flatten())
    launch_angles_r, launch_azimuths_r = hp.cartesian_to_spherical_vectorized(launch_vectors[:, 5, 1, 0].flatten(), launch_vectors[:, 5, 1, 1].flatten(), launch_vectors[:, 5, 1, 2].flatten())
    rec_angles_d, rec_azimuths_d = hp.cartesian_to_spherical_vectorized(receive_vectors[:, 5, 0, 0].flatten(), receive_vectors[:, 5, 0, 1].flatten(), receive_vectors[:, 5, 0, 2].flatten())
    rec_angles_r, rec_azimuths_r = hp.cartesian_to_spherical_vectorized(receive_vectors[:, 5, 1, 0].flatten(), receive_vectors[:, 5, 1, 1].flatten(), receive_vectors[:, 5, 1, 2].flatten())
    n_events = len(event_ids)

    with open(args.ASCIIoutput, 'w') as fout:
        fout.write(fin.attrs['header'])
        fout.write("event_ids flavors energies ccnc xx yy zz zeniths azimuths inelasticity weight\n")
        fout.write("viewing_angles_0 viewing_angles_1 launch_angles_0 launch_angles_1 rec_angles_0 rec_angles_1 solution_type_0 solution_type_1\n")
        for i in range(n_events):
            print(str(event_ids[i]) + ", ", end = '')
            fout.write("{:08d}  {:>+5d}  {:.5e}  {:s}  {:>10.3f}  {:>10.3f}  {:>10.3f}  {:>10.3f}  {:>10.3f}  {:>10.3f}  {:>10.3f}\n{:>10.3f}  {:>10.3f}  {:>10.3f}  {:>10.3f}  {:>10.3f}  {:>10.3f}  {:02f}  {:02f}\n".format(int(event_ids[i]), int(flavors[i]), energies[i], ccncs[i], xx[i], yy[i], zz[i], zeniths[i] / units.deg, azimuths[i] / units.deg, inelasticity[i], weight[i], viewing_angles_d[i] / units.deg, viewing_angles_r[i] / units.deg, launch_angles_d[i] / units.deg, launch_angles_r[i] / units.deg, 180.0 - rec_angles_d[i] / units.deg, 180.0 - rec_angles_r[i] / units.deg, st[i][5][0], st[i][5][1]))
        fout.close()
