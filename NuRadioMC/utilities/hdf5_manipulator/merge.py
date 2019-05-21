#!/usr/bin/env python
from __future__ import print_function
"""
Merge hdf5 files
"""
import os
from NuRadioMC.utilities.hdf5_manipulator import hdf5
import numpy as np
from collections import OrderedDict
from NuRadioMC.utilities.hdf5_manipulator import msg
from NuRadioMC.utilities.hdf5_manipulator import check


def get_filelist(bases):

    """look for files which match given bases and return them as list

    Keyword arguments:
    bases -- list of 'path/basename'
    """

    filelist = []

    for base in bases:
        path, fname = os.path.dirname(base) or '.', os.path.basename(base)
        filelist.extend([path + '/' + f for f in os.listdir(path)
                        if f.startswith(fname) and f.endswith(".hdf5")])

    return sorted(filelist)


def merge_data(data_list, attrs_list, groups_list, attrs_group_list):

    """Merge dictionaries with data.

    Keyword arguments:
    data_list -- the dictionary with data dictionaries
    """

    data = None
    attrs = None
    groups = None
    attrs_group = None

    for f in data_list:
        size = check.get_size(data_list[f])
        if not data:
            print("\nThe following datasets were found in %s:\n" % f)
            msg.list_dataset(data_list[f])
            data = data_list[f]
            attrs = attrs_list[f]
            groups = groups_list[f]
            attrs_group = attrs_group_list[f]
        else:
            print("\nAdding %(n)d entries from %(f)s" % {"n": size, "f": f})

            if len(attrs['trigger_names']) == 0 and 'trigger_names' in attrs_list[f]:
                attrs['trigger_names'] = attrs_list[f]['trigger_names']
            if len(data_list[f]['triggered']) == 0:
                attrs['n_events'] += attrs_list[f]['n_events']
                continue
            check.check_keys(data, data_list[f])
            check.check_shapes(data, data_list[f])
            for key in data_list[f]:
                data[key] = np.append(data[key], data_list[f][key], axis=0)
            for key in groups_list[f]:
                for key2 in groups_list[f][key]:
                    groups[key][key2] = np.append(groups[key][key2], groups_list[f][key][key2], axis=0)
                    
            attrs['n_events'] += attrs_list[f]['n_events']

    if 'trigger_names' not in attrs:
        attrs['trigger_names'] = []

    return data, attrs, groups, attrs_group


def merge_data_filenames(filelist, outputfile):
    print("The following input files were found:\n")

    for f in filelist:
        print("\t - %s" % f)

    data = OrderedDict()
    attrs = OrderedDict()
    groups = OrderedDict()
    group_attrs = OrderedDict()

    for f in filelist:
        print(f)
        data[f], attrs[f], groups[f], group_attrs[f] = hdf5.load(f)

    hdf5.save(outputfile, *merge_data(data, attrs, groups, group_attrs))

    msg.info("Done")
