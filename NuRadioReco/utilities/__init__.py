"""
This package contains many useful utility functions. Their dependencies are
kept as small as possible (and tracked) to allow usage beyond NuRadio.

Most relevant modules that can be used independently of the NuRadio framework
(dependencies in brackets, besides numpy):

- `NuRadioReco.utilities.units` [none]
  unit system (base units m, ns, GHz, eV, V, rad)
- `NuRadioReco.utilities.fft` [none]
  FFT wrappers with the NuRadio normalization (spectra in V/GHz)
- `NuRadioReco.utilities.signal_processing` [scipy]
  filtering, delaying, resampling and response windowing of traces
- `NuRadioReco.utilities.trace_utilities` [scipy]
  trace properties such as SNR, Hilbert envelope, impulsivity, energy fluence, Stokes parameters
- `NuRadioReco.utilities.geometryUtilities` [scipy, radiotools for plane-wave fit]
  time delays, plane-wave direction fits, Fresnel factors
- `NuRadioReco.utilities.analytic_pulse` [scipy]
  analytic parametrization of air-shower radio pulses
- `NuRadioReco.utilities.cr_flux` [scipy, matplotlib for plotting]
  cosmic-ray flux measurements, parametrizations and event rates
- `NuRadioReco.utilities.interferometry` [scipy, radiotools]
  beam-forming / interferometry helper functions
- `NuRadioReco.utilities.matched_filter` [matplotlib]
  matched-filter implementation
- `NuRadioReco.utilities.minimization` [matplotlib; minimizer packages as needed]
  wrapper around several minimizers (scipy, iminuit, ...)

Dependencies
------------

Many modules can be imported without the full NuRadio dependencies
(heavy packages are imported lazily inside the functions that need them).
This is checked by ``NuRadioReco/test/utilities/test_light_imports.py``.

**numpy only**

`units`, `fft`, `logging`, `ice`, `timing`, `metaclasses`, `particle_names`, `io_utilities`
(astropy only when converting times), `templates`, `_fastnumpyio`

**numpy + scipy** (no matplotlib, radiotools, astropy or NuRadioReco framework/detector/modules)

`constants`, `geometryUtilities` (radiotools only in `analytic_plane_wave_fit`), `trace_utilities`,
`signal_processing`, `analytic_pulse`, `cr_flux`

**Other third-party packages**

- `matched_filter`, `minimization`: matplotlib at import; `minimization` additionally needs
  scipy, iminuit, noisyopt or scikit-optimize depending on the chosen method
- `interferometry`: scipy, radiotools
- `dataservers`: requests, filelock

**NuRadioReco framework / detector / modules**

`noise`, `diodeSimulator`, `framework_utilities` (these also pull in scipy, radiotools, astropy, aenum)

`version` imports the top-level `NuRadioReco` and `NuRadioMC` packages.

Matplotlib, radiotools and `NuRadioReco.detector` are imported lazily in `signal_processing`,
`trace_utilities` and `geometryUtilities`, so keep it that way.

"""

from ._deprecated import *
import sys

for module in _deprecated.__all__:
    sys.modules['NuRadioReco.utilities.' + module] = _deprecated.__dict__[module]
