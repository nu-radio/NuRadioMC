"""
Check that lightweight utilities import without heavy third-party dependencies.

Each import runs in a fresh subprocess with some packages made unimportable. Two categories exist:

* ``LIGHT_MODULES``: numpy (and the stdlib) only.
* ``SCIPY_MODULES``: additionally scipy

Add a module to the respective list if it should stay importable with these dependencies only.
"""
import subprocess
import sys

LIGHT_MODULES = ["units", "fft", "logging", "ice", "particle_names", "timing", "metaclasses", "_fastnumpyio",
                 "io_utilities"]
SCIPY_MODULES = ["constants", "geometryUtilities", "trace_utilities", "signal_processing", "analytic_pulse"]

OTHER_HEAVY = ["astropy", "radiotools", "aenum", "pyarrow", "polars", "h5py", "pandas", "numba", "toml", "requests"]
BLOCKED_LIGHT = ["scipy", "matplotlib"] + OTHER_HEAVY
BLOCKED_SCIPY = ["matplotlib"] + OTHER_HEAVY
# NuRadioReco subpackages that must not be loaded by the scipy-level modules
FORBIDDEN_LOADED = ["NuRadioReco.framework", "NuRadioReco.modules", "NuRadioReco.detector"]

CODE = """
import sys, importlib, importlib.abc
BLOCKED = {blocked!r}
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split('.')[0] in BLOCKED:
            raise ImportError('BLOCKED ' + name)
sys.meta_path.insert(0, Blocker())
importlib.import_module('NuRadioReco.utilities.{module}')
loaded = [m for m in sys.modules if any(m == f or m.startswith(f + '.') for f in {forbidden!r})]
assert not loaded, 'unexpectedly loaded: ' + ', '.join(loaded)
"""


def _import_blocked(module, blocked, forbidden=()):
    """Import ``NuRadioReco.utilities.<module>`` in a subprocess with ``blocked`` packages disabled."""
    code = CODE.format(blocked=blocked, module=module, forbidden=list(forbidden))
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_light_imports():
    """All ``LIGHT_MODULES`` import with numpy only."""
    for module in LIGHT_MODULES:
        res = _import_blocked(module, BLOCKED_LIGHT)
        assert res.returncode == 0, f"{module} needs a heavy dependency:\n{res.stderr[-1000:]}"


def test_scipy_imports():
    """All ``SCIPY_MODULES`` import with numpy and scipy only, without loading the framework."""
    for module in SCIPY_MODULES:
        res = _import_blocked(module, BLOCKED_SCIPY, FORBIDDEN_LOADED)
        assert res.returncode == 0, f"{module} needs more than scipy:\n{res.stderr[-1000:]}"


def test_deprecated_import_paths():
    """The deprecated import paths (with lazy imports) still resolve with all dependencies available."""
    code = ("import warnings\n"
            "import NuRadioReco.utilities.bandpass_filter as b\n"
            "from NuRadioReco.utilities import traceWindows\n"
            "from NuRadioReco.utilities.variableWindowSizeCorrelation import variableWindowSizeCorrelation\n"
            "from NuRadioReco.utilities.bandpass_filter import get_filter_response\n")
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-1000:]


if __name__ == "__main__":
    test_light_imports()
    test_scipy_imports()
    test_deprecated_import_paths()
    print("OK")
