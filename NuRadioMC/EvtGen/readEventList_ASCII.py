from __future__ import absolute_import, division, print_function
import numpy as np
import argparse
from radiotools import helper as hp
from NuRadioReco.utilities import units
from io import BytesIO
import logging
logger = logging.getLogger("readEventList")

VERSION = 0.2


def read_eventlist(filename):
    version = 0
    with open(filename, 'r') as fin:
        lines = fin.readlines()
        for i, line in enumerate(lines):
            if line.startswith("VERSION"):
                version = float(line.split("=")[1])
                break
        if(version != VERSION):
            print("file version is {}. version != {} not supported".format(version, VERSION))
            raise NotImplementedError

        fin.seek(0)  # reset file to beginning
        data = np.genfromtxt(fin, comments='#', skip_header=1,
                             dtype=[('eventId', int), ('nuflavor', int),
                                    ('pnu', float),
                                    ('currentint', '|S2'),
                                    ('x', float), ('y', float),
                                    ('z', float), ('theta', float),
                                    ('phi', float), ('inelasticity', float)])
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse NuRadioMC event list.')
    parser.add_argument('filename', type=str,
                        help='path to NuRadioMC event list')
    args = parser.parse_args()
    events = read_eventlist(args.filename)
