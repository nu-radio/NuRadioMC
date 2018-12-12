#!/usr/bin/env python
"""
Merge hdf5 (big) files
"""
import os
import sys
import h5py
import hdf5
import numpy as np
from collections import OrderedDict
from parser import get_args_merge as parser
import msg
import check
from merge import get_filelist
from combine_big import load


def get_size(filelist):

    """Get total size of datasets; return size and ranges per file.

    Keyword arguments:
    filelist -- the list of input files
    """

    total_size = 0
    ranges = {}

    for f in filelist:
        data = h5py.File(f, 'r')
        size = check.get_size(data)
        ranges[f] = [total_size, total_size + size]
        total_size = total_size + size
        data.close()

    return total_size, ranges


def create_datasets(output, source, size):

    """Prepare datasets for merged file (based on one of input files).

    Keyword argument:
    output -- output merged hdf5 file
    source -- path to one of input hdf5 files
    size -- total number of entries per dataset
    """

    data = load(source)

    for key in data:
        shape = list(data[key].shape)
        shape[0] = size
        output.create_dataset(key, shape, dtype=data[key].dtype,
                              compression='gzip')

    data.close()


def add_data(source, output, range):

    """Merge dictionaries with data.

    Keyword arguments:
    source -- input hdf5 file path
    output -- output hdf5 file
    range -- where to save data in output arrays
    """

    data = h5py.File(source, 'r')

    print "\nAdding entries from %(f)s in [%(i)d:%(f)d]" \
          % {"f": source, "i": range[0], "f": range[1]}
    check.check_keys(data, output)
    check.check_shapes(data, output)
    for key in data:
        output[key][range[0]:range[1]] = data[key]

    data.close()

if __name__ == '__main__':

    msg.box("HDF5 MANIPULATOR: MERGE")

    args = parser()

    filelist = get_filelist([f.strip() for f in args.input_files.split(',')])

    if not filelist:
        msg.error("No files matching --input were found.")
        sys.exit(1)

    print "The following input files were found:\n"

    for f in filelist:
        print "\t - %s" % f

    output = h5py.File(args.output, 'w')

    size, ranges = get_size(filelist)

    create_datasets(output, filelist[0], size)

    for f in filelist:
        add_data(f, output, ranges[f])

    output.close()

    msg.info("Done")
