from ._deprecated import *
import sys

for module in _deprecated.__all__:
    sys.modules['NuRadioReco.utilities.' + module] = _deprecated.__dict__[module]
