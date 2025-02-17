"""
IO utilities for NuRadioReco/NuRadioMC

This module provides some pickling functions to allow
for faster, numpy 2 cross-compatible pickled numpy arrays. This mostly happens
'internally', so end users normally do not need to use this module.

"""

import pickle
import numpy as np
from ._fastnumpyio import pack, unpack # these are essentially faster alternatives for np.load/save

# we overwrite the default pickling mechanism for numpy arrays
# and scalars. We store arrays using np.save / np.load,
# and scalars by explicit casting to built-in Python types
# (note that this upcasts some types, e.g. np.float32 to float)
# This allows to maintain compatibility across numpy 2.0

def _pickle_numpy_array(arr):
    return _unpickle_numpy_array, (pack(arr),)

def _unpickle_numpy_array(data):
    return unpack(data)

def _pickle_numpy_scalar(i):
    """Convert a numpy scalar to its pure Python equivalent"""
    if isinstance(i, np.floating):
        return float, (float(i),)
    elif isinstance(i, np.integer):
        return int, (int(i),)
    elif isinstance(i, np.complexfloating):
        return complex, (complex(i),)
    elif isinstance(i, np.bool_):
        return bool, (bool(i),)
    elif isinstance(i, np.str_):
        return str, (str(i),)
    elif isinstance(i, np.bytes_):
        return bytes, (bytes(i),)
    else:
        raise TypeError(f"Unsupported type of numpy scalar {i} (type {type(i)})")


def read_pickle(filename, encoding='latin1'):
    """
    Read in a pickle file and return the result
    This utility is supposed to provide compatibility for pickles created with
    different python versions. If a simple pickle.load fails, it will try to
    load the file with a specific encoding.

    Parameters
    ----------
    filename: string
        Name of the pickle file to be opened
    encoding: string
        Encoding to be used if the first attempt to open the pickle fails
    """
    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except:
        with open(filename, 'rb') as file:
            return pickle.load(file, encoding=encoding)
