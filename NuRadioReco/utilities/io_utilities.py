"""
IO utilities for NuRadioReco/NuRadioMC

This module provides some pickling functions to allow
for faster, numpy 2 cross-compatible pickled numpy arrays. This mostly happens
'internally', so end users normally do not need to use this module.

"""

import base64
import json
import pickle
import copyreg
import numpy as np
import io
from ._fastnumpyio import pack, unpack # these are essentially faster alternatives for np.load/save
import logging
import datetime
import astropy.time

logger = logging.getLogger('NuRadioReco.utilities.io_utilities')

# we overwrite the default pickling mechanism for numpy arrays
# and scalars. We store arrays using custom, faster versions of np.save/np.load,
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

class _NurPickler(pickle.Pickler):
    """
    Custom pickler class that overwrites the pickling of numpy objects

    This class is used to overwrite the pickling mechanism
    of numpy arrays and scalars for better IO compatibility,
    as directly pickled numpy arrays are not read-compatible
    between numpy versions ``<2`` ``>=2``.
    """
    dispatch_table = copyreg.dispatch_table.copy()
    # the __reduce__ methods are overwritten by pickle.dispatch_table
    # see https://docs.python.org/3/library/pickle.html#pickle.Pickler.dispatch_table
    dispatch_table[np.ndarray] =  _pickle_numpy_array

    # there are multiple numpy scalar types (float64, float32 etc.)
    # we overwrite the pickling __reduce__ for all of them
    # note that this might upcast in some cases
    for dtype in np.ScalarType:
        if dtype.__module__ == 'numpy':
            dispatch_table[dtype] = _pickle_numpy_scalar

def _dumps(obj, protocol=None, *, fix_imports=True, **kwargs):
    """
    Return the pickled representation of the object as a bytes object.

    This is a copy of the standard `pickle.dumps` implementation,
    but uses the custom `_NurPickler` class for pickling instead of the
    global pickling mechanism. Currently, this affects only the pickling
    of numpy objects.
    """
    f = io.BytesIO()
    _NurPickler(f, protocol, fix_imports=fix_imports, **kwargs).dump(obj)
    res = f.getvalue()
    return res


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


def _astropy_to_dict(time):
    """
    Convert an astropy object to a dictionary.

    Parameters
    ----------
    time: astropy.time.Time
        Time object to be converted to a dictionary
    """
    if time is None:
        return None

    if not isinstance(time, astropy.time.Time):
        logger.error(f'Input is not an astropy object: {time}')
        raise ValueError(f'Input is not an astropy object: {time}')

    # Internally, astropy stores the time in the julian date (jd) fornat with a tuple of two double-precision floats.
    # The first float has an integer value and represents the number of days since the epoch (12:00 at January 1, 4713 BC)
    # and the second float gives the fraction of the day. That means we can reach a precision of (number of nanoseconds in a day) / 2^52:
    # 3600 * 24 * 1e9 / 2^52 = 0.02 ns. We choose to store the time object in its native format.

    data = {
        "val": time.jd1,
        "val2": time.jd2,
        "scale": time.scale,
        "format": "jd",
    }

    return data


def _time_object_to_astropy(time_object):
    """
    Convert a time_object to an astropy object.

    This function tries to encompases all the different possible ways
    a time object might have been stored inside a nur file.

    Parameters
    ----------
    time_object: dict or float or datetime.datetime or astropy.time.Time
        The time object to be converted to an astropy object

    Returns
    -------
    time: astropy.time.Time
        The time object
    """
    if time_object is None:
        return None

    if isinstance(time_object, (int, float)) and time_object == 0:
        # 0 was an old default value for the event time. It was replaced by None.
        return None

    if isinstance(time_object, astropy.time.Time):
        # For backward compatibility, we also keep supporting station times stored as astropy.time objects
        return time_object

    if isinstance(time_object, datetime.datetime):
        # For backward compatibility, we also keep supporting station times stored as datetime objects
        logger.warning(
            "Time object created from a `datetime` object. "
            "Nanosecond accuracy is not ensured.")

        return astropy.time.Time(time_object)

    if isinstance(time_object, dict):

        if 'value' in time_object and 'format' in time_object:
            logger.warning(
                "Time object created from a dictionary which does not store the nano second separately. "
                "Nanosecond accuracy is not ensured.")

            return astropy.time.Time(time_object['value'], format=time_object['format'])

        elif 'val' in time_object and 'val2' in time_object:
            if "format" not in time_object or time_object["format"] != "jd":
                logger.error(f"Time object is a dictionary but the format is wrong: {time_object}")
                raise ValueError(f"Time object is a dictionary but the format is wrong: {time_object}")

            return astropy.time.Time(**time_object)

        else:
            logger.error(f"Time object dictionary not recognized: {time_object}")
            raise ValueError(f"Time object dictionary not recognized: {time_object}")

    logger.error(f"Time object not recognized: {time_object}")
    raise ValueError(f"Time object not recognized: {time_object}")


# Tagged JSON: a lossless JSON representation of the (nested) python values NuRadio stores outside of
# the typed parameters, e.g. the keyword arguments of the registered modules. Values JSON can not express
# are stored as dict ``{"__type__": <tag>, ...}``; anything not covered falls back to ``repr`` (with a warning).
_JSON_TAG = "__type__"


def _to_json(obj):
    """
    Serialize `obj` to a JSON string which `_from_json` turns back into an equal object.

    Kept as is: str, int, float (incl. nan / inf, non-standard JSON), bool, None, lists and dicts with str keys.
    Tagged: tuple, dict with non-str keys (or a ``"__type__"`` key), complex, bytes, numpy arrays,
    `astropy.time.Time` and the parameter enums of `NuRadioReco.framework.parameters`.
    numpy scalars become the equivalent python scalars (as in `_NurPickler`). Other objects are stored as
    their ``repr`` string (with a warning), i.e., they are not restored.

    Parameters
    ----------
    obj: object
        The object to serialize

    Returns
    -------
    str
        The JSON string
    """
    return json.dumps(_json_encode(obj))


def _from_json(string):
    """
    Restore an object serialized with `_to_json`.

    Parameters
    ----------
    string: str
        The JSON string

    Returns
    -------
    object
        The restored object
    """
    return json.loads(string, object_hook=_json_decode)


def _json_encode(obj):
    """`obj` as structure of JSON types (see `_to_json`)."""
    if obj is None or (isinstance(obj, (str, bool, int, float)) and not isinstance(obj, np.generic)):
        return obj

    if isinstance(obj, np.generic):
        return _json_encode(obj.item())  # numpy scalar -> python scalar (np.float32 is upcast)

    if isinstance(obj, list):
        return [_json_encode(value) for value in obj]

    if isinstance(obj, tuple):
        return {_JSON_TAG: "tuple", "items": [_json_encode(value) for value in obj]}

    if isinstance(obj, dict):
        if all(isinstance(key, str) for key in obj) and _JSON_TAG not in obj:
            return {key: _json_encode(value) for key, value in obj.items()}

        return {_JSON_TAG: "dict", "items": [[_json_encode(key), _json_encode(value)] for key, value in obj.items()]}

    if isinstance(obj, complex):
        return {_JSON_TAG: "complex", "real": obj.real, "imag": obj.imag}

    if isinstance(obj, bytes):
        return {_JSON_TAG: "bytes", "data": base64.b64encode(obj).decode("ascii")}

    if isinstance(obj, np.ndarray) and obj.dtype.kind in "biufcUO":
        return {
            _JSON_TAG: "ndarray",
            "dtype": obj.dtype.str,
            "shape": list(obj.shape),
            "data": [_json_encode(value) for value in obj.ravel().tolist()],
        }

    if isinstance(obj, astropy.time.Time) and obj.isscalar:
        return {_JSON_TAG: "Time", **_astropy_to_dict(obj)}

    from NuRadioReco.framework import parameters  # not at the top: avoids an import cycle

    if isinstance(obj, parameters.TypedEnum) and getattr(parameters, type(obj).__name__, None) is type(obj):
        return {_JSON_TAG: "enum", "class": type(obj).__name__, "name": obj.name}

    logger.warning(f"Can not store an object of type {type(obj)} in JSON, storing its repr instead: {obj!r}")

    return repr(obj)


def _json_decode(dct):
    """`json.loads` object hook: restores the tagged dicts written by `_json_encode`."""
    tag = dct.get(_JSON_TAG)
    if tag is None:
        return dct

    if tag == "tuple":
        return tuple(dct["items"])
    if tag == "dict":
        return {key: value for key, value in dct["items"]}
    if tag == "complex":
        return complex(dct["real"], dct["imag"])
    if tag == "bytes":
        return base64.b64decode(dct["data"])
    if tag == "ndarray":
        dtype = np.dtype(dct["dtype"])
        if dtype.kind != "O":
            return np.array(dct["data"], dtype=dtype).reshape(dct["shape"])

        array = np.empty(len(dct["data"]), dtype=dtype)

        for i, value in enumerate(dct["data"]):  # element-wise: keeps object elements (e.g. lists) unchanged
            array[i] = value

        return array.reshape(dct["shape"])
    if tag == "Time":
        return _time_object_to_astropy({key: value for key, value in dct.items() if key != _JSON_TAG})
    if tag == "enum":
        from NuRadioReco.framework import parameters

        return getattr(parameters, dct["class"])[dct["name"]]

    raise ValueError(f"Unknown JSON tag {tag!r} in {dct}")
