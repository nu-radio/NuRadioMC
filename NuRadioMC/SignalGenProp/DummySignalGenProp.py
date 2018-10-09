from __future__ import absolute_import, division, print_function
import numpy as np
import h5py
import argparse
import json

VERSION = 0.1

parser = argparse.ArgumentParser(description='Parse ARA event list.')
parser.add_argument('inputfilename', type=str,
                    help='path to (hdf5) input event list')
parser.add_argument('observerlist', type=str,
                    help='path to file containing the detector positions')
parser.add_argument('outputfilename', type=str,
                    help='name of output file storing the electric field traces at detector positions')
args = parser.parse_args()

# read in detector positions
with open(args.observerlist) as fobs:
    station_list = json.load(fobs)

fout = h5py.File(args.outputfilename, 'w')

fin = h5py.File(args.inputfilename, 'r')
for event in fin['eventlist']:  # loop through all events in eventlist
    evid, nuflavorint, nu_nubar, pnu, currentint, posnu_r, posnu_theta, posnu_phi, nnu_theta, nnu_phi, elast_y, = event

    evtout = fout.create_group('event_{:06d}'.format(evid))
    evtout.attrs['event_id'] = evid

    for station in station_list:
        # create dummy electric field pulse for each detector station
        n = 1024  # length of efield trace
        sampling = 1  # in ns
        efield = np.zeros((n, 4))
        efield[:, 0] = np.arange(0, sampling * n, sampling)  # the first collumn is the time array
        efield[400, 1] = 1  # add delta pulse to efield trace

        evtout[station['name']] = efield
        evtout[station['name']].attrs['position'] = station['position']  # for convenience save station position as attribute

fout.close()
