from __future__ import absolute_import, division, print_function
import numpy as np
from NuRadioReco.utilities import units
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
ccncs = fin['interaction_type']
xx = fin['xx']
yy = fin['yy']
zz = fin['zz']
zeniths = fin['zeniths']
azimuths = fin['azimuths']
inelasticity = fin['inelasticity']

n_events = len(event_ids)

with open(args.ASCIIoutput, 'w') as fout:
    fout.write(fin.attrs['header'])
    for i in range(n_events):
        fout.write("{:08d} {:>+5d}  {:.5e}  {:s}  {:>10.3f}  {:>10.3f}  {:>10.3f}  {:>10.3f}  {:>10.3f}  {:>10.3f}\n".format(event_ids[i], flavors[i], energies[i], ccncs[i], xx[i], yy[i], zz[i], zeniths[i], azimuths[i], inelasticity[i]))
    fout.close()
