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


def write_file(filename, n_events, n_triggered, thetamin=0., thetamax=np.pi,
               Emin=1e18 * units.eV, Emax=1e18 * units.eV):
    """ Write the attributes and datasets of a NuRadioMC output file that `get_Veff_Aeff` reads. """
    triggered = np.arange(n_events) < n_triggered
    with h5py.File(filename, "w") as f:
        f.attrs["n_events"] = n_events
        f.attrs["Emin"] = Emin
        f.attrs["Emax"] = Emax
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


def check_shared_zenith_edge(folder):
    # zenith bins that share only one edge are different bins
    write_file(os.path.join(folder, "a.hdf5"), 100, 10, 0, np.pi / 2)
    write_file(os.path.join(folder, "b.hdf5"), 100, 30, 0, np.pi)

    Veffs, zenith_bins, _ = get_veffs(folder)
    assert np.allclose(zenith_bins, [[0, np.pi / 2], [0, np.pi]])
    assert np.allclose(Veffs[0, :, 1, 0], [VOLUME * 0.1, VOLUME * 0.3], rtol=1e-12), Veffs[0, :, 1, 0]


def check_energy_bins(folder):
    # an energy range and a mono-energetic file with the same (log-)mean energy are different energy bins
    write_file(os.path.join(folder, "a.hdf5"), 100, 10, Emin=1e17 * units.eV, Emax=1e19 * units.eV)
    write_file(os.path.join(folder, "b.hdf5"), 100, 30)

    Veffs, energies, energies_min, energies_max, _, _ = get_Veff_Aeff_array(get_Veff_Aeff(folder))
    assert Veffs.shape[0] == len(energies) == len(energies_min) == len(energies_max) == 2, Veffs.shape
    assert np.allclose(energies, 1e18 * units.eV, rtol=1e-12), energies
    assert np.allclose(energies_min, [1e17 * units.eV, 1e18 * units.eV]), energies_min
    assert np.allclose(energies_max, [1e19 * units.eV, 1e18 * units.eV]), energies_max
    assert np.allclose(Veffs[:, 0, 1, 0], [VOLUME * 0.1, VOLUME * 0.3], rtol=1e-12), Veffs[:, 0, 1, 0]


def check_trigger_in_some_files(folder):
    # a trigger present only in the entries of some files gets the exposure of only those files
    a, b = os.path.join(folder, "a.hdf5"), os.path.join(folder, "b.hdf5")
    write_file(a, 100, 10)
    write_file(b, 300, 90)
    data = get_Veff_Aeff(a, trigger_combinations={"combA": {"triggers": ["trig"]}}) + get_Veff_Aeff(b)
    Veffs, _, _, _, _, trigger_names = get_Veff_Aeff_array(data)
    assert list(trigger_names) == ["all_triggers", "combA", "trig"]
    fc_low, fc_high = FC_limits(10)
    expected_a = [VOLUME * 0.1, VOLUME * 0.1 / 10 ** 0.5, 10, VOLUME * fc_low / 100, VOLUME * fc_high / 100]
    assert np.allclose(Veffs[0, 0, 1], expected_a, rtol=1e-12), (Veffs[0, 0, 1], expected_a)
    assert np.allclose(Veffs[0, 0, 2, :3], [VOLUME * 0.25, VOLUME * 0.025, 100], rtol=1e-12), Veffs[0, 0, 2]


def check_same_file_twice(folder):
    # the same file twice in one bin (e.g. concatenated results for different trigger combinations) is an error
    write_file(os.path.join(folder, "a.hdf5"), 100, 10)
    data = get_Veff_Aeff(folder, trigger_combinations={"combA": {"triggers": ["trig"]}}) + \
        get_Veff_Aeff(folder, trigger_combinations={"combB": {"triggers": ["trig"]}})
    try:
        get_Veff_Aeff_array(data)
    except ValueError as e:
        assert "a.hdf5" in str(e), e
    else:
        raise AssertionError("no ValueError for the same file twice in one bin")


def test_Veff_combine_files():
    with tempfile.TemporaryDirectory() as tmp:
        check_combination(tmp)


def test_Veff_zenith_bins():
    with tempfile.TemporaryDirectory() as tmp:
        check_separate_zenith_bins(tmp)
    with tempfile.TemporaryDirectory() as tmp:
        check_shared_zenith_edge(tmp)


def test_Veff_energy_bins():
    with tempfile.TemporaryDirectory() as tmp:
        check_energy_bins(tmp)


def test_Veff_trigger_in_some_files():
    with tempfile.TemporaryDirectory() as tmp:
        check_trigger_in_some_files(tmp)


def test_Veff_same_file_twice():
    with tempfile.TemporaryDirectory() as tmp:
        check_same_file_twice(tmp)


if __name__ == "__main__":
    test_Veff_combine_files()
    test_Veff_zenith_bins()
    test_Veff_energy_bins()
    test_Veff_trigger_in_some_files()
    test_Veff_same_file_twice()
    print("OK")
