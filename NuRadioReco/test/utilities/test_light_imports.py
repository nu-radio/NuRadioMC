"""Check that lightweight utilities import without heavy third-party dependencies."""
import subprocess
import sys

LIGHT_MODULES = ["units", "fft", "logging", "ice", "particle_names", "timing", "metaclasses", "_fastnumpyio"]
BLOCKED = ["scipy", "astropy", "matplotlib", "radiotools", "aenum", "pyarrow", "polars", "h5py", "pandas",
           "numba", "toml", "requests"]

CODE = """
import sys, importlib, importlib.abc
BLOCKED = {blocked!r}
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split('.')[0] in BLOCKED:
            raise ImportError('BLOCKED ' + name)
sys.meta_path.insert(0, Blocker())
importlib.import_module('NuRadioReco.utilities.{module}')
"""


def _import_blocked(module):
    return subprocess.run([sys.executable, "-c", CODE.format(blocked=BLOCKED, module=module)],
                          capture_output=True, text=True)


def test_light_imports():
    for module in LIGHT_MODULES:
        res = _import_blocked(module)
        assert res.returncode == 0, f"{module} needs a heavy dependency:\n{res.stderr[-1000:]}"


def test_deprecated_import_paths():
    code = ("import warnings\n"
            "import NuRadioReco.utilities.bandpass_filter as b\n"
            "from NuRadioReco.utilities import traceWindows\n"
            "from NuRadioReco.utilities.variableWindowSizeCorrelation import variableWindowSizeCorrelation\n"
            "from NuRadioReco.utilities.bandpass_filter import get_filter_response\n")
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-1000:]


if __name__ == "__main__":
    test_light_imports()
    test_deprecated_import_paths()
    print("OK")
