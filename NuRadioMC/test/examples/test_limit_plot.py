"""
Smoke test of the limit plot (``NuRadioMC/examples/Sensitivities/limit_plot.py``): the figure with all models, the
source-population figure, ``add_limit`` and the command line.
"""
import os
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import numpy as np

from NuRadioMC.examples.Sensitivities import limit_plot
from NuRadioReco.utilities import units

veff_file = os.path.join(os.path.dirname(limit_plot.__file__), "veffs", "punch_deep_pa_50m_veff.txt")


def test_figures():
    for kwargs in [dict(), dict(diffuse=False)]:
        fig, ax = limit_plot.get_limit_figure(**kwargs)
        assert len(ax.lines) > 5, f"{kwargs}: figure is (almost) empty"
        xlim = ax.get_xlim()
        assert xlim[0] < xlim[1] and np.isfinite(xlim).all()

    # the default figure with the data and models of all options switched on
    options = {name: True for name in ["show_TA", "show_TA_nominal", "show_TA_ICRC2021", "show_RNOG", "show_vanVliet_TA",
                                       "show_agn", "show_grb_afterglow", "show_blazars", "show_multimessenger"]}
    fig_all, ax_all = limit_plot.get_limit_figure(**options)
    fig, ax = limit_plot.get_limit_figure()
    assert len(ax_all.lines) > len(ax.lines)


def test_add_limit():
    data = np.loadtxt(veff_file)
    fig, ax = limit_plot.get_limit_figure()
    n_lines = len(ax.lines)
    labels = limit_plot.add_limit(ax, [], data[:, 0] * units.eV, data[:, 1] * units.km3 * units.sr, 10, "test",
                                  livetime=5 * units.year)
    assert len(labels) == 1 and len(ax.lines) == n_lines + 1
    y = ax.lines[-1].get_ydata()
    assert np.all(np.isfinite(y)) and np.all(y > 0)


def test_command_line():
    with tempfile.TemporaryDirectory() as tmp:
        for args in [[], ["--sources", "--n-stations", "35", "--livetime", "10", "--label", "test"]]:
            output = os.path.join(tmp, "limit.png")
            subprocess.run([sys.executable, limit_plot.__file__, *args, "--output", output], check=True,
                           env=dict(os.environ, MPLBACKEND="Agg"))
            assert os.path.getsize(output) > 10000
            os.remove(output)


def test_removed_module():
    try:
        from NuRadioMC.examples.Sensitivities import E2_fluxes3  # noqa: F401
    except ImportError as e:
        assert "limit_plot" in str(e) and "neutrino_fluxes" in str(e)
    else:
        raise AssertionError("importing the removed E2_fluxes3 did not raise")


if __name__ == "__main__":
    test_figures()
    test_add_limit()
    test_command_line()
    test_removed_module()
    print("OK")
