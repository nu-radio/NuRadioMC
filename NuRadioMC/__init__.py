""" NuRadioMC: Simulating the radio emission of neutrinos from interaction to detector"""

import os
try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

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
