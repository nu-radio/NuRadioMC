"""
This package contains many useful utility functions.

Some rely on the classes defined in the framework of the NuRadio package
but some can also be used independently to analyze radio data.

List of (most relevant) modules for users of NuRadio are:

- `NuRadioReco.utilities.signal_processing`
- `NuRadioReco.utilities.trace_utilities`
- `NuRadioReco.utilities.units`
- `NuRadioReco.utilities.geometryUtilities`
- `NuRadioReco.utilities.fft`

"""

from ._deprecated import *
import sys

for module in _deprecated.__all__:
    sys.modules['NuRadioReco.utilities.' + module] = _deprecated.__dict__[module]
