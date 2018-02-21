from __future__ import absolute_import, division, print_function
import numpy as np
import h5py
import argparse
from io import BytesIO

VERSION = 0.1

parser = argparse.ArgumentParser(description='Parse ARA event list.')
parser.add_argument('filename', type=str,
                    help='path to ARA event list')
parser.add_argument('outputfilename', type=str,
                    help='name of hdf5 output filename')
args = parser.parse_args()
ara_version = 0
event_number = 0
with open(args.filename, 'r') as fin:
    lines = fin.readlines()
    data = ""
    for i, line in enumerate(lines):
        if line.startswith("VERSION"):
            ara_version = float(line.split("=")[1])
        elif line.startswith("EVENT_NUM"):
            event_number = int(line.split("=")[1])
        else:
            data += "{}".format(line)
    if(ara_version != 0.1):
        print("version != 0.1 not supported")
        import sys
        sys.exit(-1)

    data = np.genfromtxt(BytesIO(data), comments='//', skip_header=3,
                         dtype=[('eventId', int), ('nuflavorint', int),
                                ('nu_nubar', int), ('pnu', float),
                                ('currentint', float),
                                ('posnu_r', float), ('posnu_theta', float),
                                ('posnu_phi', float), ('nnu_theta', float),
                                ('nnu_phi', float), ('elast_y', float)])
    print(data)
    fout = h5py.File(args.outputfilename, 'w')
    fout['eventlist'] = data
    fout.attrs['VERSION'] = VERSION
    print(np.array(fout['eventlist']))
    fout.close()
