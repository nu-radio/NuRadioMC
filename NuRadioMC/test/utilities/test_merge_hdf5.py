"""
Test of ``NuRadioMC.utilities.merge_hdf5.merge2``: merging files whose event group ids and shower ids overlap
(e.g. independently generated input files) must give unique ids that stay consistent between the top level and
the station groups (issue #292).
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
        sg["event_id_per_shower"] = np.asarray(shower_ids)  # holds shower ids, see output_writer_hdf5


def _merge(inputs):
    """ Writes one file per (event_group_ids, shower_ids) pair, merges them and returns the merged datasets. """
    with tempfile.TemporaryDirectory() as tmpdir:
        filenames = []
        for i, (egids, sids) in enumerate(inputs):
            filenames.append(os.path.join(tmpdir, f"in{i}.hdf5"))
            _write_file(filenames[-1], egids, sids)
        output = os.path.join(tmpdir, "merged.hdf5")
        merge2(filenames, output)
        with h5py.File(output, "r") as f:
            return {key: f[key][...] for key in ["event_group_ids", "shower_ids", "station_101/event_group_ids",
                                                 "station_101/shower_id", "station_101/event_group_id_per_shower",
                                                 "station_101/event_id_per_shower"]}


def test_merge_shower_ids_unique():
    out = _merge([([1, 2, 2], [0, 1, 2]), ([1, 2, 2], [0, 1, 2]), ([5], [7])])
    assert len(np.unique(out["shower_ids"])) == 7, out["shower_ids"]
    # the station group references the same (renamed) showers as the top level
    np.testing.assert_array_equal(out["station_101/shower_id"], out["shower_ids"])


def test_merge_event_id_per_shower_renamed():
    # the station dataset ``event_id_per_shower`` stores shower ids and must be renamed with them
    out = _merge([([1, 2, 2], [0, 1, 2]), ([1, 2, 2], [0, 1, 2])])
    np.testing.assert_array_equal(out["station_101/event_id_per_shower"], out["station_101/shower_id"])


def test_merge_event_group_id_zero_unique():
    # only event group id 0 overlaps
    out = _merge([([0, 1], [0, 1]), ([0, 2], [2, 3])])
    assert len(np.unique(out["event_group_ids"])) == 4, out["event_group_ids"]
    np.testing.assert_array_equal(np.sort(out["station_101/event_group_ids"]), np.unique(out["event_group_ids"]))


def test_merge_station_ids_consistent():
    out = _merge([([1, 2, 2], [0, 1, 2]), ([1, 2, 2], [0, 1, 2])])
    egid_of_shower = dict(zip(out["shower_ids"], out["event_group_ids"]))
    for sid, egid in zip(out["station_101/shower_id"], out["station_101/event_group_id_per_shower"]):
        assert egid_of_shower[sid] == egid, (sid, egid, egid_of_shower)


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


def _run_cli(*args):
    script = os.path.join(os.path.dirname(merge_hdf5.__file__), "merge_hdf5.py")
    result = subprocess.run([sys.executable, script, *args], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cli_directory_mode():
    with tempfile.TemporaryDirectory() as root:
        parts = _write_part_tree(root)
        _run_cli(root)
        for output, inputs in parts.items():
            with h5py.File(output, "r") as f:
                assert f.attrs["n_events"] == 2 * len(inputs), (output, f.attrs["n_events"])


if __name__ == "__main__":
    test_merge_shower_ids_unique()
    test_merge_event_id_per_shower_renamed()
    test_merge_event_group_id_zero_unique()
    test_merge_station_ids_consistent()
    test_merge_untriggered_file_adds_no_rows()
    test_merge_trigger_names_differ()
    test_merge_station_attrs()
    test_find_part_files()
    test_cli_directory_mode()
    print("OK")
