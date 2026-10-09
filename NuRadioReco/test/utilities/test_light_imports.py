"""
Check that the light utilities import without the heavy NuRadio dependencies.

Each import runs in a fresh subprocess with the heavy third-party packages made unimportable
(numpy, scipy and matplotlib stay available) and must not load the NuRadioReco framework.

Add a module to ``LIGHT_MODULES`` if it should stay importable this way (and to the docstring of
``NuRadioReco/utilities/__init__.py``). All other modules are not tested.
"""
import subprocess
import sys

LIGHT_MODULES = ["units", "fft", "constants", "geometryUtilities", "trace_utilities", "signal_processing",
                 "analytic_pulse", "cr_flux"]

BLOCKED = ["astropy", "radiotools", "aenum", "pyarrow", "polars", "h5py", "pandas", "numba", "toml", "requests"]
# NuRadioReco subpackages that must not be loaded by the light modules
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


def _import_blocked(module):
    """Import ``NuRadioReco.utilities.<module>`` in a subprocess with the heavy packages disabled."""
    code = CODE.format(blocked=BLOCKED, module=module, forbidden=FORBIDDEN_LOADED)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_light_imports():
    """All ``LIGHT_MODULES`` import without the heavy dependencies and without loading the framework."""
    for module in LIGHT_MODULES:
        res = _import_blocked(module)
        assert res.returncode == 0, f"{module} needs a heavy dependency:\n{res.stderr[-1000:]}"


def test_light_modules_exist():
    """All ``LIGHT_MODULES`` exist in ``NuRadioReco.utilities``."""
    import pkgutil
    import NuRadioReco.utilities

    found = {m.name for m in pkgutil.iter_modules(NuRadioReco.utilities.__path__)}
    stale = sorted(set(LIGHT_MODULES) - found)
    assert not stale, f"Listed module(s) {stale} do not exist (anymore)"


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
    test_light_modules_exist()
    test_deprecated_import_paths()
    print("OK")
