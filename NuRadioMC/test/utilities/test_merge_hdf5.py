"""
Test of ``NuRadioMC.utilities.merge_hdf5.merge2``: merging files whose event group ids and shower ids overlap
(e.g. independently generated input files) must give unique ids that stay consistent between the top level and
the station groups (issue #292), and store the per-file id offsets.
"""
import os
import subprocess
import sys
import tempfile

import h5py
import numpy as np

from NuRadioMC.utilities import merge_hdf5
from NuRadioMC.utilities.merge_hdf5 import merge2


def _write_file(filename, event_group_ids, shower_ids, attrs=None, station_attrs=None, triggered=True):
    """
    Minimal NuRadioMC output file: one station that sees every shower, one event per event group. ``attrs`` and
    ``station_attrs`` are added to the file and station group attributes; ``triggered`` is the value of all
    ``triggered`` entries.
    """
    event_group_ids = np.asarray(event_group_ids)
    with h5py.File(filename, "w") as f:
        f.attrs["n_events"] = len(np.unique(event_group_ids))
        f.attrs.update(attrs or {})
        f["event_group_ids"] = event_group_ids
        f["shower_ids"] = np.asarray(shower_ids)
        f["triggered"] = np.full(len(shower_ids), triggered)
        sg = f.create_group("station_101")
        sg.attrs.update(station_attrs or {})
        sg["event_group_ids"] = np.unique(event_group_ids)
        sg["event_ids"] = np.zeros(len(np.unique(event_group_ids)), dtype=int)
        sg["shower_id"] = np.asarray(shower_ids)
        sg["event_group_id_per_shower"] = event_group_ids
        sg["event_id_per_shower"] = np.zeros(len(shower_ids), dtype=int)


DATASETS = ["event_group_ids", "shower_ids", "station_101/event_group_ids", "station_101/shower_id",
            "station_101/event_group_id_per_shower", "station_101/event_id_per_shower"]


def _merge(inputs):
    """
    Writes one file ``in<i>.hdf5`` per (event_group_ids, shower_ids) pair, merges them and returns the merged datasets
    and, under ``"attrs"``, the file attributes.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        filenames = []
        for i, (egids, sids) in enumerate(inputs):
            filenames.append(os.path.join(tmpdir, f"in{i}.hdf5"))
            _write_file(filenames[-1], egids, sids)
        output = os.path.join(tmpdir, "merged.hdf5")
        merge2(filenames, output)
        with h5py.File(output, "r") as f:
            return dict({key: f[key][...] for key in DATASETS}, attrs=dict(f.attrs))


def test_merge_shower_ids_unique():
    out = _merge([([1, 2, 2], [0, 1, 2]), ([1, 2, 2], [0, 1, 2]), ([5], [7])])
    # the second file is shifted above the first, the third lies above both and keeps its ids
    np.testing.assert_array_equal(out["shower_ids"], [0, 1, 2, 3, 4, 5, 7])
    np.testing.assert_array_equal(out["event_group_ids"], [1, 2, 2, 3, 4, 4, 5])
    # the station group references the same (shifted) showers as the top level
    np.testing.assert_array_equal(out["station_101/shower_id"], out["shower_ids"])


def test_merge_event_id_per_shower_not_renamed():
    # event ids are unique only within an event group and are kept when shower ids are shifted
    out = _merge([([1, 2, 2], [0, 1, 2]), ([1, 2, 2], [0, 1, 2])])
    assert len(np.unique(out["station_101/shower_id"])) == 6, out["station_101/shower_id"]
    np.testing.assert_array_equal(out["station_101/event_id_per_shower"], np.zeros(6, dtype=int))


def test_merge_event_group_id_zero_unique():
    # only event group id 0 overlaps
    out = _merge([([0, 1], [0, 1]), ([0, 2], [2, 3])])
    np.testing.assert_array_equal(out["event_group_ids"], [0, 1, 2, 4])
    np.testing.assert_array_equal(out["station_101/event_group_ids"], [0, 1, 2, 4])
    # the shower ids do not overlap and are kept
    np.testing.assert_array_equal(out["shower_ids"], [0, 1, 2, 3])


def test_merge_overlapping_ids_shifted():
    # all ids of an overlapping file are shifted by one offset: gaps and order are kept
    out = _merge([([10, 30, 30], [5, 9, 20]), ([10, 30, 30], [5, 9, 20])])
    np.testing.assert_array_equal(out["event_group_ids"], [10, 30, 30, 31, 51, 51])
    np.testing.assert_array_equal(out["shower_ids"], [5, 9, 20, 21, 25, 36])
    np.testing.assert_array_equal(out["attrs"]["event_group_id_offsets"], [0, 21])
    np.testing.assert_array_equal(out["attrs"]["shower_id_offsets"], [0, 16])


def test_merge_non_overlapping_ids_kept():
    # e.g. part files of one run: ids above all previous ids are kept, even with gaps
    out = _merge([([1, 2], [1, 2]), ([5, 6], [3, 4]), ([7], [9])])
    np.testing.assert_array_equal(out["event_group_ids"], [1, 2, 5, 6, 7])
    np.testing.assert_array_equal(out["shower_ids"], [1, 2, 3, 4, 9])
    np.testing.assert_array_equal(out["attrs"]["event_group_id_offsets"], [0, 0, 0])
    np.testing.assert_array_equal(out["attrs"]["shower_id_offsets"], [0, 0, 0])


def test_merge_offsets_recover_original_ids():
    inputs = [([3, 4, 4], [0, 1, 2]), ([0, 1], [0, 5]), ([20], [6]), ([2, 2], [1, 3])]
    out = _merge(inputs)
    attrs = out["attrs"]
    assert list(attrs["merged_files"]) == [f"in{i}.hdf5" for i in range(len(inputs))], attrs["merged_files"]
    i = 0
    for (egids, sids), egid_offset, sid_offset in zip(inputs, attrs["event_group_id_offsets"],
                                                      attrs["shower_id_offsets"]):
        n = len(sids)
        np.testing.assert_array_equal(out["event_group_ids"][i:i + n] - egid_offset, egids)
        np.testing.assert_array_equal(out["shower_ids"][i:i + n] - sid_offset, sids)
        i += n
    assert len(np.unique(out["shower_ids"])) == len(out["shower_ids"]), out["shower_ids"]


def test_merge_station_ids_consistent():
    out = _merge([([1, 2, 2], [0, 1, 2]), ([1, 2, 2], [0, 1, 2])])
    egid_of_shower = dict(zip(out["shower_ids"], out["event_group_ids"]))
    for sid, egid in zip(out["station_101/shower_id"], out["station_101/event_group_id_per_shower"]):
        assert egid_of_shower[sid] == egid, (sid, egid, egid_of_shower)
    np.testing.assert_array_equal(out["station_101/event_group_ids"], np.unique(out["event_group_ids"]))
    np.testing.assert_array_equal(out["station_101/event_group_id_per_shower"], out["event_group_ids"])


def test_merge_untriggered_file_adds_no_rows():
    # a file without triggered events is skipped and must not add (zero-filled) rows
    with tempfile.TemporaryDirectory() as tmpdir:
        filenames = [os.path.join(tmpdir, "triggered.hdf5"), os.path.join(tmpdir, "untriggered.hdf5")]
        _write_file(filenames[0], [1, 2, 2], [1, 2, 3])
        _write_file(filenames[1], [5, 6], [5, 6], triggered=False)
        output = os.path.join(tmpdir, "merged.hdf5")
        merge2(filenames, output)
        with h5py.File(output, "r") as f:
            np.testing.assert_array_equal(f["shower_ids"][...], [1, 2, 3])
            np.testing.assert_array_equal(f["station_101/shower_id"], [1, 2, 3])
            np.testing.assert_array_equal(f["station_101/event_group_ids"], [1, 2])
            assert f.attrs["n_events"] == 2 + 2, f.attrs["n_events"]
            # the untriggered file is listed, with offsets 0
            assert list(f.attrs["merged_files"]) == ["triggered.hdf5", "untriggered.hdf5"], f.attrs["merged_files"]
            np.testing.assert_array_equal(f.attrs["event_group_id_offsets"], [0, 0])
            np.testing.assert_array_equal(f.attrs["shower_id_offsets"], [0, 0])


def test_merge_merged_files():
    # merging merged files lists the original files, with the offsets of both merges added
    with tempfile.TemporaryDirectory() as tmpdir:
        inputs = {"a": ([1, 2], [1, 2]), "b": ([1, 2], [1, 2]), "c": ([0], [0]), "d": ([0], [0])}
        for name, (egids, sids) in inputs.items():
            _write_file(os.path.join(tmpdir, f"{name}.hdf5"), egids, sids)
        merge2([os.path.join(tmpdir, x) for x in ["a.hdf5", "b.hdf5"]], os.path.join(tmpdir, "ab.hdf5"))
        merge2([os.path.join(tmpdir, x) for x in ["c.hdf5", "d.hdf5"]], os.path.join(tmpdir, "cd.hdf5"))
        output = os.path.join(tmpdir, "merged.hdf5")
        merge2([os.path.join(tmpdir, x) for x in ["ab.hdf5", "cd.hdf5"]], output)
        with h5py.File(output, "r") as f:
            assert list(f.attrs["merged_files"]) == ["a.hdf5", "b.hdf5", "c.hdf5", "d.hdf5"], f.attrs["merged_files"]
            # ab: 1, 2, 3, 4; cd: 0, 1 shifted by 5
            np.testing.assert_array_equal(f["event_group_ids"][...], [1, 2, 3, 4, 5, 6])
            np.testing.assert_array_equal(f.attrs["event_group_id_offsets"], [0, 2, 5, 6])
            np.testing.assert_array_equal(f.attrs["shower_id_offsets"], [0, 2, 5, 6])


def _merge_attrs(attrs, station_attrs=None):
    """ Merges files with disjoint ids and the given file (station group) attributes; returns the merged attributes. """
    station_attrs = station_attrs or [{}] * len(attrs)
    with tempfile.TemporaryDirectory() as tmpdir:
        filenames = []
        for i, (a, sa) in enumerate(zip(attrs, station_attrs)):
            filenames.append(os.path.join(tmpdir, f"in{i}.hdf5"))
            _write_file(filenames[-1], [10 * i], [10 * i], attrs=a, station_attrs=sa)
        output = os.path.join(tmpdir, "merged.hdf5")
        merge2(filenames, output)
        with h5py.File(output, "r") as f:
            return dict(f.attrs), dict(f["station_101"].attrs)


def _assert_raises(exception, func, *args):
    try:
        func(*args)
    except exception:
        return
    raise AssertionError(f"{exception.__name__} not raised")


def test_merge_trigger_names_differ():
    # any differing trigger name raises, not only if all differ
    _assert_raises(AttributeError, _merge_attrs, [{"trigger_names": np.array(["a", "b"], dtype="S")},
                                                  {"trigger_names": np.array(["a", "X"], dtype="S")}])
    names = np.array(["a", "b"], dtype="S")
    attrs, _ = _merge_attrs([{"trigger_names": names}, {"trigger_names": names}])
    np.testing.assert_array_equal(attrs["trigger_names"], names)


def test_merge_station_attrs():
    # an attribute only in a later file is taken from that file
    _, station_attrs = _merge_attrs([{}, {}], [{}, {"Vrms": [1., 2.]}])
    np.testing.assert_array_equal(station_attrs["Vrms"], [1., 2.])
    # differing string attributes keep the value of the first file
    _, station_attrs = _merge_attrs([{}, {}], [{"name": "x"}, {"name": "y"}])
    assert station_attrs["name"] == "x", station_attrs


def test_merge_file_attrs():
    # string attributes (e.g. the generator's ``header``) and arrays of different length keep the value of the first file
    attrs, _ = _merge_attrs([{"header": "x", "flavors": [12, -12]}, {"header": "x", "flavors": [12, -12]}])
    assert attrs["header"] == "x", attrs
    attrs, _ = _merge_attrs([{"header": "x", "flavors": [12, -12]}, {"header": "y", "flavors": [12, -12, 14]}])
    assert attrs["header"] == "x", attrs
    np.testing.assert_array_equal(attrs["flavors"], [12, -12])


def test_merge_veff_attrs_differ():
    # attributes that enter the effective volume/area must agree, other differing attributes only warn
    attrs = {"Emin": 1e18, "Emax": 1e19, "volume": 1e9, "area": 1e6, "fiducial_zmin": -2.7e3}
    for key in ["Emin", "Emax", "volume", "area"]:
        _assert_raises(AttributeError, _merge_attrs, [attrs, dict(attrs, **{key: 2 * attrs[key]})])
    merged, _ = _merge_attrs([attrs, dict(attrs, fiducial_zmin=-1e3)])
    assert merged["fiducial_zmin"] == attrs["fiducial_zmin"], merged


def _write_part_tree(root):
    """ Generator-style tree ``root/<energy>/<name>.hdf5.partNNNN``; returns the part files per expected output file. """
    parts = {}
    for energy, name, n_parts in [("1e18", "a", 2), ("1e19", "b", 1)]:
        os.makedirs(os.path.join(root, energy))
        output = os.path.join(root, f"{name}.hdf5")
        parts[output] = []
        for i in range(n_parts):
            parts[output].append(os.path.join(root, energy, f"{name}.hdf5.part{i:04d}"))
            _write_file(parts[output][-1], [10 * i, 10 * i + 1], [10 * i, 10 * i + 1])
    # files of at most 1000 bytes are skipped
    open(os.path.join(root, "1e18", "a.hdf5.part0099"), "w").close()
    return parts


def test_find_part_files():
    with tempfile.TemporaryDirectory() as root:
        parts = _write_part_tree(root)
        found = merge_hdf5.find_part_files(root)
        assert {x["output_filename"]: list(x["filenames"]) for x in found} == parts, found


def _run_cli(*args, success=True):
    script = os.path.join(os.path.dirname(merge_hdf5.__file__), "merge_hdf5.py")
    # timeout: a pool whose workers fail to start hangs
    result = subprocess.run([sys.executable, script, *args], capture_output=True, text=True, timeout=60)
    assert (result.returncode == 0) == success, result.stderr


def test_cli_directory_mode(*args):
    """ ``args``: further command line arguments """
    with tempfile.TemporaryDirectory() as root:
        parts = _write_part_tree(root)
        _run_cli(root, *args)
        for output, inputs in parts.items():
            with h5py.File(output, "r") as f:
                assert f.attrs["n_events"] == 2 * len(inputs), (output, f.attrs["n_events"])


def test_cli_directory_mode_multiprocessing():
    test_cli_directory_mode("--cores", "2")


def test_cli_loglevel():
    # level names and numbers are accepted, anything else is an error
    with tempfile.TemporaryDirectory() as tmpdir:
        inputs = [os.path.join(tmpdir, f"in{i}.hdf5") for i in range(2)]
        for i, f in enumerate(inputs):
            _write_file(f, [i], [i])
        for i, level in enumerate(["DEBUG", "info", "10"]):
            _run_cli(os.path.join(tmpdir, f"out{i}.hdf5"), *inputs, "--loglevel", level)
        _run_cli(os.path.join(tmpdir, "out_bad.hdf5"), *inputs, "--loglevel", "print('x')", success=False)


if __name__ == "__main__":
    test_merge_shower_ids_unique()
    test_merge_event_id_per_shower_not_renamed()
    test_merge_event_group_id_zero_unique()
    test_merge_overlapping_ids_shifted()
    test_merge_non_overlapping_ids_kept()
    test_merge_offsets_recover_original_ids()
    test_merge_station_ids_consistent()
    test_merge_untriggered_file_adds_no_rows()
    test_merge_merged_files()
    test_merge_trigger_names_differ()
    test_merge_station_attrs()
    test_merge_file_attrs()
    test_merge_veff_attrs_differ()
    test_find_part_files()
    test_cli_directory_mode()
    test_cli_directory_mode_multiprocessing()
    test_cli_loglevel()
    print("OK")
