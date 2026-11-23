"""
Test of ``NuRadioMC.utilities.Veff.get_Veff_Aeff_array`` on small synthetic NuRadioMC HDF5 files: several files that
cover the same energy and zenith bin are combined instead of the last file overwriting the others
(https://github.com/nu-radio/NuRadioMC/issues/490).
"""
import os
import tempfile

import h5py
import numpy as np

from NuRadioMC.utilities.Veff import FC_limits, get_Veff_Aeff, get_Veff_Aeff_array
from NuRadioReco.utilities import units

VOLUME = 10 * units.km ** 3


def write_file(filename, n_events, n_triggered, thetamin=0., thetamax=np.pi):
    """ Write the attributes and datasets of a NuRadioMC output file that `get_Veff_Aeff` reads. """
    triggered = np.arange(n_events) < n_triggered
    with h5py.File(filename, "w") as f:
        f.attrs["n_events"] = n_events
        f.attrs["Emin"] = 1e18 * units.eV
        f.attrs["Emax"] = 1e18 * units.eV
        f.attrs["phimin"] = 0.
        f.attrs["phimax"] = 2 * np.pi
        f.attrs["thetamin"] = thetamin
        f.attrs["thetamax"] = thetamax
        f.attrs["volume"] = VOLUME
        f.attrs["Vrms"] = 1.
        f.attrs["trigger_names"] = ["trig"]
        f["weights"] = np.ones(n_events)
        f["triggered"] = triggered
        f["multiple_triggers"] = triggered[:, None]
        f["event_group_ids"] = np.arange(n_events)


def get_veffs(path):
    Veffs, energies, _, _, zenith_bins, trigger_names = get_Veff_Aeff_array(get_Veff_Aeff(path))
    return Veffs, zenith_bins, trigger_names


def check_combination(folder):
    # two files with the same energy and zenith bin: Veff 1 km^3 and 3 km^3, with 100 and 300 events
    write_file(os.path.join(folder, "a.hdf5"), 100, 10)
    write_file(os.path.join(folder, "b.hdf5"), 300, 90)

    Veffs, zenith_bins, trigger_names = get_veffs(folder)
    assert Veffs.shape == (1, 1, 2, 5), Veffs.shape
    assert list(trigger_names) == ["all_triggers", "trig"]

    # the files are one simulation with 400 events and 100 triggers
    counts = 100
    fc_low, fc_high = FC_limits(counts)
    expected = [VOLUME * counts / 400, VOLUME * counts / 400 / counts ** 0.5, counts,
                VOLUME * fc_low / 400, VOLUME * fc_high / 400]
    for iTrig in range(2):
        assert np.allclose(Veffs[0, 0, iTrig], expected, rtol=1e-12), (Veffs[0, 0, iTrig], expected)

    # a single file is unchanged
    Veffs_b, _, _ = get_veffs(os.path.join(folder, "b.hdf5"))
    fc_low, fc_high = FC_limits(90)
    expected_b = [VOLUME * 90 / 300, VOLUME * 90 / 300 / 90 ** 0.5, 90, VOLUME * fc_low / 300, VOLUME * fc_high / 300]
    assert np.allclose(Veffs_b[0, 0, 1], expected_b, rtol=1e-12), (Veffs_b[0, 0, 1], expected_b)


def check_separate_zenith_bins(folder):
    # files in different zenith bins stay separate
    write_file(os.path.join(folder, "a.hdf5"), 100, 10, 0, np.pi / 2)
    write_file(os.path.join(folder, "b.hdf5"), 100, 30, np.pi / 2, np.pi)

    Veffs, zenith_bins, _ = get_veffs(folder)
    assert np.allclose(zenith_bins, [[0, np.pi / 2], [np.pi / 2, np.pi]])
    assert np.allclose(Veffs[0, :, 1, 0], [VOLUME * 0.1, VOLUME * 0.3], rtol=1e-12), Veffs[0, :, 1, 0]


def test_Veff_combine_files():
    with tempfile.TemporaryDirectory() as tmp:
        check_combination(tmp)


def test_Veff_zenith_bins():
    with tempfile.TemporaryDirectory() as tmp:
        check_separate_zenith_bins(tmp)


if __name__ == "__main__":
    test_Veff_combine_files()
    test_Veff_zenith_bins()
    print("OK")
