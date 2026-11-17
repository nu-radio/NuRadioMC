"""
Collection of utility functions and classes.

Most relevant modules:

- `NuRadioReco.utilities.units`: unit system (base units m, ns, GHz, eV, V, rad)
- `NuRadioReco.utilities.fft`: FFT wrappers with the NuRadio normalization (spectra in V/GHz)
- `NuRadioReco.utilities.signal_processing`: filtering, delaying, resampling and response windowing of traces
- `NuRadioReco.utilities.trace_utilities`: trace properties such as SNR, Hilbert envelope, impulsivity, energy fluence, Stokes parameters
- `NuRadioReco.utilities.geometryUtilities`: time delays, plane-wave direction fits, Fresnel factors
- `NuRadioReco.utilities.analytic_pulse`: analytic parametrization of air-shower radio pulses
- `NuRadioReco.utilities.cr_flux`: cosmic-ray flux measurements, parametrizations and event rates
- `NuRadioReco.utilities.interferometry`: beam-forming / interferometry helper functions
- `NuRadioReco.utilities.matched_filter`: matched-filter implementation
- `NuRadioReco.utilities.minimization`: wrapper around several minimizers (scipy, iminuit, ...)

Dependencies
------------

For a subset of the modules we keep the import dependencies light (numpy, scipy and matplotlib, no
NuRadioReco framework), so that they can be used in other projects without a full NuRadio installation.
Individual functions in these modules can still need additional packages, which are then imported lazily
(e.g. radiotools in `geometryUtilities`). This subset is:

`units`, `fft`, `constants`, `geometryUtilities`, `trace_utilities`, `signal_processing`, `analytic_pulse`, `cr_flux`.

It is defined in ``NuRadioReco/test/utilities/test_light_imports.py``, which checks that it stays importable.
Other modules may require additional dependencies already at import.
"""

from ._deprecated import *
import sys

for module in _deprecated.__all__:
    sys.modules['NuRadioReco.utilities.' + module] = _deprecated.__dict__[module]
