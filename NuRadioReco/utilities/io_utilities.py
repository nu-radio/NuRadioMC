import pickle
import numpy as np
import io

# we overwrite the default pickling mechanism for numpy arrays
# and scalars. We store arrays using np.save / np.load,
# and scalars by explicit casting to built-in Python types
# (note that this upcasts some types, e.g. np.float32 to float)
# This allows to maintain compatibility across numpy 2.0

def _pickle_numpy_array(arr):
    dummy_file = io.BytesIO()
    np.save(dummy_file, arr)
    return _unpickle_numpy_array, (dummy_file.getvalue(),)

def _unpickle_numpy_array(data):
    dummy_file = io.BytesIO(data)
    return np.load(dummy_file)

def _pickle_numpy_scalar(i):
    if isinstance(i, np.floating):
        return float, (float(i),)
    elif isinstance(i, np.integer):
        return int, (int(i),)
    elif isinstance(i, np.complexfloating):
        return complex, (complex(i),)
    elif isinstance(i, np.bool_):
        return bool, (bool(i),)
    else:
        raise TypeError(f"Type of scalar {i} ({type(i)}) is not one of float, int, complex or bool.")

# the __reduce__ methods are overwritten by pickle.dispatch_table
# see https://docs.python.org/3/library/pickle.html#pickle.Pickler.dispatch_table
pickle.dispatch_table[np.ndarray] = _pickle_numpy_array

# there are multiple numpy scalar types (float64, float32 etc.)
# we overwrite the pickling __reduce__ for all of them
# note that this might upcast in some cases
for dtype in np.ScalarType:
    if dtype.__module__ == 'numpy':
        pickle.dispatch_table[dtype] = _pickle_numpy_scalar


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
