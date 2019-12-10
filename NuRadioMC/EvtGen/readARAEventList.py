from __future__ import absolute_import, division, print_function
import numpy as np
import argparse
from radiotools import helper as hp
from NuRadioReco.utilities import units
from io import BytesIO
import logging
logger = logging.getLogger("readARAEventList")

VERSION = 0.1


def read_ARA_eventlist(filename):
    ara_version = 0
    event_number = 0
    with open(filename, 'r') as fin:
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
            print("file version is {}. version != 0.1 not supported".format(ara_version))
            import sys
            sys.exit(-1)

        data = np.genfromtxt(BytesIO(data), comments='//', skip_header=3,
                             dtype=[('eventId', int), ('nuflavorint', int),
                                    ('nu_nubar', int), ('pnu', float),
                                    ('currentint', float),
                                    ('posnu_r', float), ('posnu_theta', float),
                                    ('posnu_phi', float), ('nnu_theta', float),
                                    ('nnu_phi', float), ('elast_y', float)])
        # convert angles into NuRadioMC coordinate convention
        for i in range(len(data)):
            data[i][3] = 10**(data[i][3] + 18.) * units.eV
            data[i][4] = data[i][4] * units.m
            data[i][6] = hp.get_normalized_angle(0.5 * np.pi - data[i][6])  # convert theta angle into NuRadioMC coordinate convention
            data[i][8] = hp.get_normalized_angle(0.5 * np.pi - data[i][8])  # convert theta angle into NuRadioMC coordinate convention
        return data


def convert_to_hdf5(araeventlist_filename, hdf5_filename):
    import h5py
    data = read_ARA_eventlist(araeventlist_filename)
    print(data)
    fout = h5py.File(hdf5_filename, 'w')
    fout['eventlist'] = data
    fout.attrs['VERSION'] = VERSION
    print(np.array(fout['eventlist']))
    fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse ARA event list.')
    parser.add_argument('filename', type=str,
                        help='path to ARA event list')
    parser.add_argument('output_filename', type=str,
                        help='name of hdf5 output filename')
    args = parser.parse_args()
    convert_to_hdf5(args.filename, args.output_filename)
