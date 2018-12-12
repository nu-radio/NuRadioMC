"""
Basic checks for data dictionaries.
"""

import sys
from NuRadioMC.utilities.hdf5_manipulator import msg


def get_size(data):

    """Check if #entries is the same for all keys and return it

    Keyword arguments:
    data -- data dictionary
    """

    sizes = [d.shape[0] for d in data.itervalues()]  # shape[0] = #entries

    if max(sizes) != min(sizes):
        msg.error("Each dataset within a file must have the "
                  "same number of entries!")
        sys.exit(1)

    return sizes[0]


def same_sizes(data1, data2):

    """Check if files have the same #entries per dataset.

    Keyword arguments:
    data1 -- first file
    data2 -- second file
    """

    if get_size(data1) != get_size(data2):
        msg.error("Files must have the same number of entries to be combined.")
        sys.exit(1)


def check_keys(data1, data2):

    """Check it both files have the same datasets.

    Keyword arguments:
    data1 -- current data dictionary
    data2 -- data dictionary to be added
    """

    if data1.keys() != data2.keys():
        msg.error("Files have different datasets.")
        sys.exit(1)


def check_shapes(data1, data2):

    """Check if shapes of datasets are the same.

    Keyword arguments:
    data1 -- current data dictionary
    data2 -- data dictionary to be added
    """

    for key in data1.keys():
        if data1[key].shape[1:] != data2[key].shape[1:]:
            msg.error("Different shapes for dataset: %s. " % key)
            sys.exit(1)


def key_exists(key, data, filename):

    """Check if given dataset is included in the file.

    Keyword arguments:
    key -- key to look for
    data -- data dictionary to check
    """

    if key not in data.keys():
        msg.error("'%(key)s' key is missing in %(file)s."
                  % {"key": key, "file": filename})
        sys.exit(1)


def different_keys(data1, data2, skip):

    """Check if given files have different (except skip) datasets.

    Keyword arguments:
    data1 -- data dictionary
    data2 -- data dictionary
    skip -- common key
    """

    for key in data1.keys():
        if key == skip:
            continue
        if key in data2.keys():
            msg.error("Duplicated dataset: %s in input files." % key)
            sys.exit(1)


def check_duplicates(keys1, keys2):

    """Check if given files have different (except skip) datasets.

    Keyword arguments:
    keys1 -- the list of keys to be copied from file1
    keys2 -- the list of keys to be copied from file2
    """

    for key in keys1:
        if key in keys2:
            msg.error("Duplicated dataset: %s in input files." % key)
            sys.exit(1)
