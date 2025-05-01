import os
import copyreg
import numpy as np
from .utilities.io_utilities import _pickle_numpy_array, _pickle_numpy_scalar
import logging

from NuRadioReco.utilities.logging import NuRadioLogger, _setup_logger

logging.setLoggerClass(NuRadioLogger)
_setup_logger(name="NuRadioReco")

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

# Set version number
__version__ = None
# First, try to obtain version number from pyproject.toml (developer version)
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
toml_file = os.path.join(parent_dir, 'pyproject.toml')
if os.path.isfile(toml_file):
    import toml
    toml_dict = toml.load(toml_file)
    try:
        if toml_dict['tool']['poetry']['name'] == "NuRadioMC": # check this is the right pyproject.toml
            __version__ = toml_dict['tool']['poetry']['version']
    except KeyError:
        pass

# If not available, we're probably using the pip installed package
if __version__ == None:
    __version__ = importlib_metadata.version("NuRadioMC")

# Overwrite the pickling mechanism of numpy arrays for better IO compatibility.
#
# The __reduce__ methods are overwritten by the global dispatch_table.
# We modify this by using copyreg.pickle.
# see https://docs.python.org/3/library/pickle.html#dispatch-tables

copyreg.pickle(np.ndarray, _pickle_numpy_array)
# there are multiple numpy scalar types (float64, float32 etc.)
# we overwrite the pickling __reduce__ for all of them
# note that this might upcast in some cases
for dtype in np.ScalarType:
    if dtype.__module__ == 'numpy':
        copyreg.pickle(dtype, _pickle_numpy_scalar)
