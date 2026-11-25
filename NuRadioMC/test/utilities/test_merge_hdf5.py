"""
Tests of ``NuRadioMC.utilities.merge_hdf5``: ids made unique across files (issue #292) and their offsets, attribute
merging, empty input files, directory mode and command line.
"""
import logging
import os
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pytest

import NuRadioMC
from NuRadioReco.utilities import logging as nrr_logging
from NuRadioMC.utilities import merge_hdf5
from NuRadioMC.utilities.merge_hdf5 import merge_hdf5_files

REFERENCE = os.path.join(os.path.dirname(NuRadioMC.__file__), "test", "SingleEvents", "1e18_output_reference.hdf5")


def _write_file(filename, event_group_ids, shower_ids, attrs=None, station_attrs=None, triggered=True, libver=None):
    """
    Minimal NuRadioMC output file: one station that sees every shower, one event per event group. ``attrs`` and
    ``station_attrs`` are added to the file and station group attributes; ``triggered`` is the value of all
    ``triggered`` entries; ``libver`` is passed to `h5py.File`.
    """
    event_group_ids = np.asarray(event_group_ids)
    with h5py.File(filename, "w", libver=libver) as f:
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


def _write_files(tmp_path, inputs, **kwargs):
    """ Writes one file per (event_group_ids, shower_ids) pair; returns the file names. """
    filenames = []
    for i, (egids, sids) in enumerate(inputs):
        filenames.append(str(tmp_path / f"in{i}.hdf5"))
        _write_file(filenames[-1], egids, sids, **kwargs)
    return filenames


def _read(filename):
    """ All datasets (``group/name`` for station groups) and the file attributes. """
    out = {}
    with h5py.File(filename, "r") as f:
        f.visititems(lambda name, obj: out.__setitem__(name, obj[...]) if isinstance(obj, h5py.Dataset) else None)
        return out, dict(f.attrs)


def _merge(tmp_path, inputs):
    """ Writes one file per (event_group_ids, shower_ids) pair, merges them and returns the merged datasets. """
    output = str(tmp_path / "merged.hdf5")
    merge_hdf5_files(_write_files(tmp_path, inputs), output)
    return _read(output)[0]


def _merge_with_attrs(tmp_path, inputs):
    """ As `_merge`, but returns the merged datasets and the file attributes. """
    output = str(tmp_path / "merged.hdf5")
    merge_hdf5_files(_write_files(tmp_path, inputs), output)
    return _read(output)


def test_make_ids_unique():
    # no overlap: offsets 0, also with gaps
    assert merge_hdf5._make_ids_unique([[1, 2], [5, 9]]) == [0, 0]
    # an overlapping file is shifted above the largest id so far ([2, 6], then [7, 8, 13, 16]); the last file lies
    # above all of them
    assert merge_hdf5._make_ids_unique([[0, 1, 1], [1, 5], [0, 1, 6, 9], [20]]) == [0, 1, 7, 0]


def test_merge_shower_ids_unique(tmp_path):
    out = _merge(tmp_path, [([1, 2, 2], [0, 1, 2]), ([1, 2, 2], [0, 1, 2]), ([5], [7])])
    # the second file is shifted above the first, the third lies above both and keeps its ids
    np.testing.assert_array_equal(out["shower_ids"], [0, 1, 2, 3, 4, 5, 7])
    np.testing.assert_array_equal(out["event_group_ids"], [1, 2, 2, 3, 4, 4, 5])
    # the station group references the same (shifted) showers as the top level
    np.testing.assert_array_equal(out["station_101/shower_id"], out["shower_ids"])


def test_merge_event_id_per_shower_not_renamed(tmp_path):
    # event ids are unique only within an event group and are kept when shower ids are shifted
    out = _merge(tmp_path, [([1, 2, 2], [0, 1, 2]), ([1, 2, 2], [0, 1, 2])])
    assert len(np.unique(out["station_101/shower_id"])) == 6, out["station_101/shower_id"]
    np.testing.assert_array_equal(out["station_101/event_id_per_shower"], np.zeros(6, dtype=int))


def test_merge_event_group_id_zero_unique(tmp_path):
    # only event group id 0 overlaps
    out = _merge(tmp_path, [([0, 1], [0, 1]), ([0, 2], [2, 3])])
    np.testing.assert_array_equal(out["event_group_ids"], [0, 1, 2, 4])
    np.testing.assert_array_equal(out["station_101/event_group_ids"], [0, 1, 2, 4])
    # the shower ids do not overlap and are kept
    np.testing.assert_array_equal(out["shower_ids"], [0, 1, 2, 3])


def test_merge_overlapping_ids_shifted(tmp_path):
    # all ids of an overlapping file are shifted by one offset: gaps and order are kept
    out, attrs = _merge_with_attrs(tmp_path, [([10, 30, 30], [5, 9, 20]), ([10, 30, 30], [5, 9, 20])])
    np.testing.assert_array_equal(out["event_group_ids"], [10, 30, 30, 31, 51, 51])
    np.testing.assert_array_equal(out["shower_ids"], [5, 9, 20, 21, 25, 36])
    np.testing.assert_array_equal(attrs["event_group_id_offsets"], [0, 21])
    np.testing.assert_array_equal(attrs["shower_id_offsets"], [0, 16])


def test_merge_non_overlapping_ids_kept(tmp_path):
    # e.g. part files of one run: ids above all previous ids are kept, even with gaps
    out, attrs = _merge_with_attrs(tmp_path, [([1, 2], [1, 2]), ([5, 6], [3, 4]), ([7], [9])])
    np.testing.assert_array_equal(out["event_group_ids"], [1, 2, 5, 6, 7])
    np.testing.assert_array_equal(out["shower_ids"], [1, 2, 3, 4, 9])
    np.testing.assert_array_equal(attrs["event_group_id_offsets"], [0, 0, 0])
    np.testing.assert_array_equal(attrs["shower_id_offsets"], [0, 0, 0])


def test_merge_offsets_recover_original_ids(tmp_path):
    inputs = [([3, 4, 4], [0, 1, 2]), ([0, 1], [0, 5]), ([20], [6]), ([2, 2], [1, 3])]
    out, attrs = _merge_with_attrs(tmp_path, inputs)
    assert list(attrs["merged_files"]) == [f"in{i}.hdf5" for i in range(len(inputs))], attrs["merged_files"]
    i = 0
    for (egids, sids), egid_offset, sid_offset in zip(inputs, attrs["event_group_id_offsets"],
                                                      attrs["shower_id_offsets"]):
        n = len(sids)
        np.testing.assert_array_equal(out["event_group_ids"][i:i + n] - egid_offset, egids)
        np.testing.assert_array_equal(out["shower_ids"][i:i + n] - sid_offset, sids)
        i += n
    assert len(np.unique(out["shower_ids"])) == len(out["shower_ids"]), out["shower_ids"]


def test_merge_station_ids_consistent(tmp_path):
    out = _merge(tmp_path, [([1, 2, 2], [0, 1, 2]), ([1, 2, 2], [0, 1, 2])])
    egid_of_shower = dict(zip(out["shower_ids"], out["event_group_ids"]))
    for sid, egid in zip(out["station_101/shower_id"], out["station_101/event_group_id_per_shower"]):
        assert egid_of_shower[sid] == egid, (sid, egid, egid_of_shower)
    np.testing.assert_array_equal(out["station_101/event_group_ids"], np.unique(out["event_group_ids"]))
    np.testing.assert_array_equal(out["station_101/event_group_id_per_shower"], out["event_group_ids"])


def test_merge_untriggered_file_adds_no_rows(tmp_path):
    # a file without triggered events is skipped and must not add (zero-filled) rows, but counts for n_events
    filenames = [str(tmp_path / "triggered.hdf5"), str(tmp_path / "untriggered.hdf5")]
    _write_file(filenames[0], [1, 2, 2], [1, 2, 3])
    _write_file(filenames[1], [5, 6], [5, 6], triggered=False)
    output = str(tmp_path / "merged.hdf5")
    merge_hdf5_files(filenames, output)
    data, attrs = _read(output)
    np.testing.assert_array_equal(data["shower_ids"], [1, 2, 3])
    np.testing.assert_array_equal(data["station_101/shower_id"], [1, 2, 3])
    np.testing.assert_array_equal(data["station_101/event_group_ids"], [1, 2])
    assert attrs["n_events"] == 2 + 2, attrs["n_events"]
    # the untriggered file is listed, with offsets 0
    assert list(attrs["merged_files"]) == ["triggered.hdf5", "untriggered.hdf5"], attrs["merged_files"]
    np.testing.assert_array_equal(attrs["event_group_id_offsets"], [0, 0])
    np.testing.assert_array_equal(attrs["shower_id_offsets"], [0, 0])
    np.testing.assert_array_equal(attrs["event_group_id_starts"], [1, -1])


def test_merge_merged_files(tmp_path):
    # merging merged files lists the original files, with the offsets of both merges added
    inputs = {"a": ([1, 2], [1, 2]), "b": ([1, 2], [1, 2]), "c": ([0], [0]), "d": ([0], [0])}
    for name, (egids, sids) in inputs.items():
        _write_file(str(tmp_path / f"{name}.hdf5"), egids, sids)
    merge_hdf5_files([str(tmp_path / "a.hdf5"), str(tmp_path / "b.hdf5")], str(tmp_path / "ab.hdf5"))
    merge_hdf5_files([str(tmp_path / "c.hdf5"), str(tmp_path / "d.hdf5")], str(tmp_path / "cd.hdf5"))
    output = str(tmp_path / "merged.hdf5")
    merge_hdf5_files([str(tmp_path / "ab.hdf5"), str(tmp_path / "cd.hdf5")], output)
    data, attrs = _read(output)
    assert list(attrs["merged_files"]) == ["a.hdf5", "b.hdf5", "c.hdf5", "d.hdf5"], attrs["merged_files"]
    # ab: 1, 2, 3, 4; cd: 0, 1 shifted by 5
    np.testing.assert_array_equal(data["event_group_ids"], [1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(attrs["event_group_id_offsets"], [0, 2, 5, 6])
    np.testing.assert_array_equal(attrs["shower_id_offsets"], [0, 2, 5, 6])
    np.testing.assert_array_equal(attrs["event_group_id_starts"], [1, 3, 5, 6])


def _file_index(event_group_ids, starts):
    """ Index (in ``merged_files``) of the file of each merged event group id, as documented in HDF5_structure.rst. """
    with_rows = np.flatnonzero(starts >= 0)
    return with_rows[np.searchsorted(starts[with_rows], event_group_ids, side="right") - 1]


def test_merge_event_group_id_starts(tmp_path):
    # the file of a row follows from its event group id; files without rows have start -1
    inputs = [([3, 4, 4], [0, 1, 2]), ([7, 8], [3, 4]), ([0, 1], [0, 5]), ([20], [6])]
    filenames = [str(tmp_path / f"in{i}.hdf5") for i in range(len(inputs))]
    for i, (f, (egids, sids)) in enumerate(zip(filenames, inputs)):
        _write_file(f, egids, sids, triggered=i != 1)
    output = str(tmp_path / "merged.hdf5")
    merge_hdf5_files(filenames, output)
    data, attrs = _read(output)
    egids, sids = data["event_group_ids"], data["shower_ids"]
    # in1 has no triggered events; in2 is shifted by 5 (to 5, 6); in3 keeps its ids
    np.testing.assert_array_equal(attrs["event_group_id_starts"], [3, -1, 5, 20])
    index = _file_index(egids, attrs["event_group_id_starts"])
    np.testing.assert_array_equal(index, [0, 0, 0, 2, 2, 3])
    np.testing.assert_array_equal(egids - attrs["event_group_id_offsets"][index], [3, 4, 4, 0, 1, 20])
    np.testing.assert_array_equal(sids - attrs["shower_id_offsets"][index], [0, 1, 2, 0, 5, 6])
    # station groups too
    np.testing.assert_array_equal(_file_index(data["station_101/event_group_ids"], attrs["event_group_id_starts"]),
                                  [0, 0, 2, 2, 3])


def test_merge_incomplete_merge_history(tmp_path):
    # a merged file without event_group_id_starts (merged before it existed) is listed as one file
    for name, egids in [("a", [1, 2]), ("b", [1, 2]), ("c", [0])]:
        _write_file(str(tmp_path / f"{name}.hdf5"), egids, egids)
    merge_hdf5_files([str(tmp_path / "a.hdf5"), str(tmp_path / "b.hdf5")], str(tmp_path / "ab.hdf5"))
    with h5py.File(tmp_path / "ab.hdf5", "a") as f:
        del f.attrs["event_group_id_starts"]
    output = str(tmp_path / "merged.hdf5")
    merge_hdf5_files([str(tmp_path / "ab.hdf5"), str(tmp_path / "c.hdf5")], output)
    attrs = _read(output)[1]
    assert list(attrs["merged_files"]) == ["ab.hdf5", "c.hdf5"], attrs["merged_files"]
    np.testing.assert_array_equal(attrs["event_group_id_offsets"], [0, 5])
    np.testing.assert_array_equal(attrs["event_group_id_starts"], [1, 5])


def test_merge_duplicate_path(tmp_path, caplog):
    # a file given more than once (also via another path or a symbolic link) is merged once, with a warning
    a = str(tmp_path / "a.hdf5")
    _write_file(a, [1, 2, 2], [1, 2, 3])
    link = str(tmp_path / "link.hdf5")
    os.symlink(a, link)
    merge_hdf5_files([a], str(tmp_path / "once.hdf5"))
    with caplog.at_level(logging.WARNING, logger="NuRadioMC.HDF5-merger"):
        merge_hdf5_files([a, os.path.join(str(tmp_path), ".", "a.hdf5"), link], str(tmp_path / "twice.hdf5"))
    assert caplog.text.count("merging it once") == 2, caplog.text
    once, once_attrs = _read(str(tmp_path / "once.hdf5"))
    twice, twice_attrs = _read(str(tmp_path / "twice.hdf5"))
    assert twice_attrs["n_events"] == once_attrs["n_events"] == 2, twice_attrs["n_events"]
    for key in once:
        np.testing.assert_array_equal(twice[key], once[key])
    assert list(twice_attrs["merged_files"]) == ["a.hdf5"], twice_attrs["merged_files"]
    np.testing.assert_array_equal(twice_attrs["event_group_id_offsets"], [0])
    np.testing.assert_array_equal(twice_attrs["shower_id_offsets"], [0])
    np.testing.assert_array_equal(twice_attrs["event_group_id_starts"], [1])


def test_merge_file_without_datasets(tmp_path):
    # an empty simulation output (``write_empty_output_file``) has attributes only; it counts for n_events
    filenames = _write_files(tmp_path, [([1, 2], [1, 2])])
    with h5py.File(tmp_path / "empty.hdf5", "w") as f:
        f.attrs["n_events"] = 7
    output = str(tmp_path / "merged.hdf5")
    merge_hdf5_files([str(tmp_path / "empty.hdf5")] + filenames, output)
    data, attrs = _read(output)
    np.testing.assert_array_equal(data["shower_ids"], [1, 2])
    assert attrs["n_events"] == 7 + 2, attrs["n_events"]


def test_merge_all_files_empty(tmp_path):
    # all files empty: the first file is copied, n_events is the sum
    filenames = [str(tmp_path / "a.hdf5"), str(tmp_path / "b.hdf5")]
    _write_file(filenames[0], [1, 2], [1, 2], triggered=False, attrs={"x": 1})
    _write_file(filenames[1], [5, 6, 7], [5, 6, 7], triggered=False, attrs={"x": 2})
    output = str(tmp_path / "merged.hdf5")
    merge_hdf5_files(filenames, output)
    data, attrs = _read(output)
    np.testing.assert_array_equal(data["shower_ids"], [1, 2])
    np.testing.assert_array_equal(data["station_101/shower_id"], [1, 2])
    assert attrs["n_events"] == 2 + 3 and attrs["x"] == 1, attrs
    with h5py.File(output, "r") as f:
        assert f["shower_ids"].compression == "gzip"


def test_merge_top_level_dataset_missing(tmp_path, caplog):
    # a top-level dataset missing in one of the files is dropped with a warning
    filenames = _write_files(tmp_path, [([1], [1]), ([2], [2])])
    with h5py.File(filenames[0], "a") as f:
        f["xx"] = [1.]
    output = str(tmp_path / "merged.hdf5")
    with caplog.at_level(logging.WARNING, logger="NuRadioMC.HDF5-merger"):
        merge_hdf5_files(filenames, output)
    assert "xx" not in _read(output)[0]
    assert "not all files have the key xx" in caplog.text


def _merge_attrs(tmp_path, attrs, station_attrs=None):
    """ Merges files with disjoint ids and the given file (station group) attributes; returns the merged attributes. """
    station_attrs = station_attrs or [{}] * len(attrs)
    filenames = []
    for i, (a, sa) in enumerate(zip(attrs, station_attrs)):
        filenames.append(str(tmp_path / f"in{i}.hdf5"))
        _write_file(filenames[-1], [10 * i], [10 * i], attrs=a, station_attrs=sa)
    output = str(tmp_path / "merged.hdf5")
    if os.path.exists(output):
        os.remove(output)
    merge_hdf5_files(filenames, output)
    with h5py.File(output, "r") as f:
        return dict(f.attrs), dict(f["station_101"].attrs)


def test_merge_trigger_names_equal(tmp_path):
    names = np.array(["a", "b"], dtype="S")
    attrs, _ = _merge_attrs(tmp_path, [{"trigger_names": names}, {"trigger_names": names}])
    np.testing.assert_array_equal(attrs["trigger_names"], names)
    assert attrs["trigger_names"].dtype == names.dtype


def _write_triggers(filename, names, multiple_triggers, trigger_times):
    """ Adds ``trigger_names`` and the per-trigger datasets (one row per shower and event) to a `_write_file` file. """
    with h5py.File(filename, "a") as f:
        f.attrs["trigger_names"] = names
        for prefix, suffix in [("", ""), ("station_101/", ""), ("station_101/", "_per_event")]:
            for key, value in [("multiple_triggers", np.array(multiple_triggers, dtype=bool)),
                               ("trigger_times", np.array(trigger_times, dtype=float))]:
                if f"{prefix}{key}{suffix}" in f:
                    del f[f"{prefix}{key}{suffix}"]
                f[f"{prefix}{key}{suffix}"] = value


def test_merge_trigger_names_differ(tmp_path, caplog):
    # a trigger that never fired in a file is missing from its trigger names: merged by name, the per-trigger
    # datasets are re-indexed to the merged order, missing columns are False / NaN
    filenames = _write_files(tmp_path, [([1], [1]), ([2], [2]), ([3], [3])])
    _write_triggers(filenames[0], ["a", "b"], [[True, False]], [[1., np.nan]])
    _write_triggers(filenames[1], ["c", "a"], [[True, True]], [[3., 2.]])
    _write_triggers(filenames[2], ["a", "b"], [[False, True]], [[np.nan, 4.]])
    output = str(tmp_path / "merged.hdf5")
    with caplog.at_level(logging.WARNING, logger="NuRadioMC.HDF5-merger"):
        merge_hdf5_files(filenames, output)
    assert "trigger names of 3 files differ" in caplog.text
    data, attrs = _read(output)
    assert list(attrs["trigger_names"]) == ["a", "b", "c"], attrs["trigger_names"]
    for prefix, suffix in [("", ""), ("station_101/", ""), ("station_101/", "_per_event")]:
        np.testing.assert_array_equal(data[f"{prefix}multiple_triggers{suffix}"],
                                      [[True, False, False], [True, False, True], [False, True, False]])
        np.testing.assert_array_equal(data[f"{prefix}trigger_times{suffix}"],
                                      [[1., np.nan, np.nan], [2., np.nan, 3.], [np.nan, 4., np.nan]])
    # a per-trigger dataset that does not match the file's trigger names
    _write_triggers(filenames[1], ["c"], [[True, True]], [[3., 2.]])
    with pytest.raises(ValueError, match="trigger names"):
        merge_hdf5_files(filenames, output, overwrite=True)


def test_merge_trigger_names_of_empty_files(tmp_path, caplog):
    # files without triggered events add no rows: their trigger names do not enter the merged ones
    for empty_names in [["t2", "t1"], []]:
        filenames = _write_files(tmp_path, [([1], [1]), ([2], [2])])
        _write_file(filenames[0], [1], [1], triggered=False)
        _write_triggers(filenames[0], empty_names, [[False] * len(empty_names)], [[np.nan] * len(empty_names)])
        _write_triggers(filenames[1], ["t1", "t2"], [[True, False]], [[1., np.nan]])
        output = str(tmp_path / "merged.hdf5")
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="NuRadioMC.HDF5-merger"):
            merge_hdf5_files(filenames, output, overwrite=True)
        assert "differ" not in caplog.text, caplog.text
        data, attrs = _read(output)
        assert list(attrs["trigger_names"]) == ["t1", "t2"], attrs["trigger_names"]
        np.testing.assert_array_equal(data["multiple_triggers"], [[True, False]])
        np.testing.assert_array_equal(data["trigger_times"], [[1., np.nan]])


def test_merge_string_attrs_differ(tmp_path):
    with pytest.raises(AttributeError):
        _merge_attrs(tmp_path, [{"config": "a: 1"}, {"config": "a: 2"}])


def test_merge_station_attrs(tmp_path, caplog):
    # an attribute only in a later file is taken from that file
    _, station_attrs = _merge_attrs(tmp_path, [{}, {}], [{}, {"Vrms": [1., 2.]}])
    np.testing.assert_array_equal(station_attrs["Vrms"], [1., 2.])
    # differing string attributes keep the value of the first file
    _, station_attrs = _merge_attrs(tmp_path, [{}, {}], [{"name": "x"}, {"name": "y"}])
    assert station_attrs["name"] == "x", station_attrs
    # differing numbers warn and keep the value of the first file
    with caplog.at_level(logging.WARNING, logger="NuRadioMC.HDF5-merger"):
        _, station_attrs = _merge_attrs(tmp_path, [{}, {}], [{"Vrms": [1., 2.]}, {"Vrms": [1., 3.]}])
    np.testing.assert_array_equal(station_attrs["Vrms"], [1., 2.])
    assert "attribute Vrms of group station_101" in caplog.text


def test_merge_file_attrs(tmp_path, caplog):
    # string attributes (e.g. the generator's ``header``) and arrays of different length keep the value of the first file
    attrs, _ = _merge_attrs(tmp_path, [{"header": "x", "flavors": [12, -12]}, {"header": "x", "flavors": [12, -12]}])
    assert attrs["header"] == "x", attrs
    attrs, _ = _merge_attrs(tmp_path, [{"header": "x", "flavors": [12, -12]}, {"header": "y", "flavors": [12, -12, 14]}])
    assert attrs["header"] == "x", attrs
    np.testing.assert_array_equal(attrs["flavors"], [12, -12])
    # start_event_id is not compared
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="NuRadioMC.HDF5-merger"):
        attrs, _ = _merge_attrs(tmp_path, [{"start_event_id": 0}, {"start_event_id": 100}])
    assert attrs["start_event_id"] == 0 and "start_event_id" not in caplog.text, caplog.text
    # an attribute only in a later file is taken from that file
    attrs, _ = _merge_attrs(tmp_path, [{}, {"trigger_names": np.array(["a"], dtype="S")}])
    np.testing.assert_array_equal(attrs["trigger_names"], np.array(["a"], dtype="S"))


def test_merge_veff_attrs_differ(tmp_path):
    # attributes that enter the effective volume/area must agree, other differing attributes only warn
    attrs = {"Emin": 1e18, "Emax": 1e19, "volume": 1e9, "area": 1e6, "fiducial_zmin": -2.7e3}
    for key in ["Emin", "Emax", "volume", "area"]:
        with pytest.raises(AttributeError):
            _merge_attrs(tmp_path, [attrs, dict(attrs, **{key: 2 * attrs[key]})])
    merged, _ = _merge_attrs(tmp_path, [attrs, dict(attrs, fiducial_zmin=-1e3)])
    assert merged["fiducial_zmin"] == attrs["fiducial_zmin"], merged


def test_merge_reference_file(tmp_path):
    # two copies of a simulation output: n_events doubled, rows appended, ids unique and consistent
    filenames = [str(tmp_path / "a.hdf5"), str(tmp_path / "b.hdf5")]
    for f in filenames:
        shutil.copy(REFERENCE, f)
    output = str(tmp_path / "merged.hdf5")
    merge_hdf5_files(filenames, output)
    ref, ref_attrs = _read(REFERENCE)
    data, attrs = _read(output)
    assert attrs["n_events"] == 2 * ref_attrs["n_events"]
    assert set(data) == set(ref)
    for key in ref:
        assert len(data[key]) == 2 * len(ref[key]), key
    assert len(np.unique(data["shower_ids"])) == 2 * len(np.unique(ref["shower_ids"]))
    assert len(np.unique(data["event_group_ids"])) == 2 * len(np.unique(ref["event_group_ids"]))
    egid_of_shower = dict(zip(data["shower_ids"], data["event_group_ids"]))
    for sid, egid in zip(data["station_101/shower_id"], data["station_101/event_group_id_per_shower"]):
        assert egid_of_shower[sid] == egid
    # datasets without ids are appended unchanged
    np.testing.assert_array_equal(data["xx"], np.concatenate([ref["xx"], ref["xx"]]))
    np.testing.assert_array_equal(data["station_101/event_ids"],
                                  np.concatenate([ref["station_101/event_ids"]] * 2))
    with h5py.File(output, "r") as f:
        assert f["xx"].compression == "gzip"
        assert h5py.check_string_dtype(f["interaction_type"].dtype).encoding == "utf-8"
        assert "station_102" in f


def test_merge2_deprecated(tmp_path):
    # merge2 warns and, as before, overwrites the output
    filenames = _write_files(tmp_path, [([1], [1]), ([2], [2])])
    output = str(tmp_path / "merged.hdf5")
    open(output, "w").close()
    with pytest.warns(DeprecationWarning):
        merge_hdf5.merge2(filenames, output)
    np.testing.assert_array_equal(_read(output)[0]["shower_ids"], [1, 2])


def test_merge2_deprecation_warning_attributed_to_caller(tmp_path):
    # attributed to the calling file, the default warning filters show it in __main__ scripts
    filenames = _write_files(tmp_path, [([1], [1]), ([2], [2])])
    with pytest.warns(DeprecationWarning) as record:
        merge_hdf5.merge2(filenames, str(tmp_path / "merged.hdf5"))
    assert [w.filename for w in record] == [__file__], [w.filename for w in record]


def test_deprecated_fallback_attributed_to_caller():
    # the decorator of NuRadioReco.utilities.logging for Python < 3.13 (else warnings.deprecated)
    @nrr_logging._deprecated_fallback("f is deprecated")
    def f():
        return 1

    with pytest.warns(DeprecationWarning) as record:
        assert f() == 1
    assert [w.filename for w in record] == [__file__], [w.filename for w in record]


def test_merge_output_checks(tmp_path):
    filenames = _write_files(tmp_path, [([1], [1]), ([2], [2])])
    output = str(tmp_path / "merged.hdf5")
    with pytest.raises(ValueError):
        merge_hdf5_files([], output)
    with pytest.raises(ValueError):
        merge_hdf5_files(filenames, filenames[0])
    merge_hdf5_files(filenames, output)
    # an existing output is overwritten only on request
    with pytest.raises(FileExistsError):
        merge_hdf5_files(filenames[:1], output)
    merge_hdf5_files(filenames[:1], output, overwrite=True)
    np.testing.assert_array_equal(_read(output)[0]["shower_ids"], [1])
    # a failed merge leaves neither the output nor the temporary file
    with h5py.File(filenames[1], "a") as f:
        f.attrs["config"] = "x"
    with h5py.File(filenames[0], "a") as f:
        f.attrs["config"] = "y"
    os.remove(output)
    with pytest.raises(AttributeError):
        merge_hdf5_files(filenames, output)
    assert not os.path.exists(output) and not os.path.exists(output + ".tmp")


def test_merge_station_groups_union(tmp_path):
    # a station group only in a later file is merged as well, with its attributes
    filenames = _write_files(tmp_path, [([1], [1]), ([2], [2])])
    with h5py.File(filenames[1], "a") as f:
        f["station_102/shower_id"] = [2]
        f["station_102"].attrs["Vrms"] = [1.]
    output = str(tmp_path / "merged.hdf5")
    merge_hdf5_files(filenames, output)
    data, _ = _read(output)
    np.testing.assert_array_equal(data["station_101/shower_id"], [1, 2])
    np.testing.assert_array_equal(data["station_102/shower_id"], [2])
    with h5py.File(output, "r") as f:
        np.testing.assert_array_equal(f["station_102"].attrs["Vrms"], [1.])


def test_merge_incomplete_station(tmp_path, caplog, restore_loglevel):
    # a station dataset missing in one file raises by default, or that file's rows of the station are left out
    filenames = _write_files(tmp_path, [([1, 2], [1, 2]), ([3], [3]), ([4], [4])])
    for f in filenames[::2]:
        with h5py.File(f, "a") as fin:
            fin["station_101/maximum_amplitudes"] = np.ones((len(fin["station_101/event_ids"]), 4))
    output = str(tmp_path / "merged.hdf5")
    with pytest.raises(ValueError, match="maximum_amplitudes"):
        merge_hdf5_files(filenames, output)
    with caplog.at_level(logging.WARNING, logger="NuRadioMC.HDF5-merger"):
        merge_hdf5_files(filenames, output, skip_incomplete_stations=True)
    assert "in1.hdf5 misses the datasets ['maximum_amplitudes'] of station_101" in caplog.text
    data, attrs = _read(output)
    # the top level keeps all files
    np.testing.assert_array_equal(data["shower_ids"], [1, 2, 3, 4])
    assert attrs["n_events"] == 4
    # all datasets of the station group stay aligned
    np.testing.assert_array_equal(data["station_101/shower_id"], [1, 2, 4])
    np.testing.assert_array_equal(data["station_101/event_group_id_per_shower"], [1, 2, 4])
    np.testing.assert_array_equal(data["station_101/event_group_ids"], [1, 2, 4])
    assert data["station_101/maximum_amplitudes"].shape == (3, 4)
    # command line
    merge_hdf5.main([output, *filenames, "--overwrite", "--skip-incomplete-stations"])
    np.testing.assert_array_equal(_read(output)[0]["station_101/shower_id"], [1, 2, 4])


def test_merge_empty_station_group(tmp_path):
    # the writer leaves the group of a station without events empty; it adds no rows and is not incomplete
    filenames = _write_files(tmp_path, [([1], [1]), ([2], [2])])
    with h5py.File(filenames[0], "a") as f:
        f.create_group("station_102")
    with h5py.File(filenames[1], "a") as f:
        f["station_102/shower_id"] = [2]
        f["station_102/event_ids"] = [0]
    output = str(tmp_path / "merged.hdf5")
    merge_hdf5_files(filenames, output)
    np.testing.assert_array_equal(_read(output)[0]["station_102/shower_id"], [2])


def test_merge_trailing_shape_mismatch(tmp_path):
    filenames = _write_files(tmp_path, [([1], [1]), ([2], [2])])
    for f, n in zip(filenames, [2, 3]):
        with h5py.File(f, "a") as fin:
            fin["station_101/maximum_amplitudes"] = np.zeros((1, n))
    with pytest.raises(ValueError, match="maximum_amplitudes"):
        merge_hdf5_files(filenames, str(tmp_path / "merged.hdf5"))
    assert not os.path.exists(str(tmp_path / "merged.hdf5.tmp"))


def _write_part_tree(root):
    """ Generator-style tree ``root/<energy>/<name>.hdf5.partNNNN``; returns the part files per expected output file. """
    parts = {}
    for energy, name, part_numbers in [("1e18", "a", [0, 1, 10000]), ("1e18", "c", [0]), ("1e19", "b", [0])]:
        os.makedirs(os.path.join(root, energy), exist_ok=True)
        output = os.path.join(root, f"{name}.hdf5")
        parts[output] = []
        for i in part_numbers:
            parts[output].append(os.path.join(root, energy, f"{name}.hdf5.part{i:04d}"))
            _write_file(parts[output][-1], [10 * i, 10 * i + 1], [10 * i, 10 * i + 1])
    # files of at most 1000 bytes are skipped
    open(os.path.join(root, "1e18", "a.hdf5.part0099"), "w").close()
    return parts


def test_find_part_files(tmp_path, caplog):
    parts = _write_part_tree(str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="NuRadioMC.HDF5-merger"):
        found = merge_hdf5.find_part_files(str(tmp_path))
    assert {x["output_filename"]: list(x["filenames"]) for x in found} == parts, found
    assert "a.hdf5.part0099" in caplog.text
    # existing outputs are skipped unless requested
    existing = str(tmp_path / "a.hdf5")
    open(existing, "w").close()
    assert existing not in [x["output_filename"] for x in merge_hdf5.find_part_files(str(tmp_path))]
    assert existing in [x["output_filename"] for x in merge_hdf5.find_part_files(str(tmp_path), skip_existing=False)]


def test_find_part_files_same_name_in_two_folders(tmp_path, caplog):
    # the same name in two subfolders maps to one output file: warned about, the groups are returned as before
    parts = _write_part_tree(str(tmp_path))
    duplicate = os.path.join(str(tmp_path), "1e19", "a.hdf5.part0000")
    _write_file(duplicate, [5], [5])
    with caplog.at_level(logging.WARNING, logger="NuRadioMC.HDF5-merger"):
        found = merge_hdf5.find_part_files(str(tmp_path))
    assert "a.hdf5" in caplog.text and os.path.join(str(tmp_path), "1e18") in caplog.text \
        and os.path.join(str(tmp_path), "1e19") in caplog.text, caplog.text
    output = os.path.join(str(tmp_path), "a.hdf5")
    assert [x["filenames"] for x in found if x["output_filename"] == output] == [parts[output], [duplicate]]
    # no warning without duplicates
    caplog.clear()
    os.remove(duplicate)
    with caplog.at_level(logging.WARNING, logger="NuRadioMC.HDF5-merger"):
        merge_hdf5.find_part_files(str(tmp_path))
    assert "subfolders" not in caplog.text, caplog.text


def test_find_part_files_all_parts_small(tmp_path, caplog):
    # a name whose part files are all too small gives no group
    parts = _write_part_tree(str(tmp_path))
    open(os.path.join(str(tmp_path), "1e19", "d.hdf5.part0000"), "w").close()
    with caplog.at_level(logging.ERROR, logger="NuRadioMC.HDF5-merger"):
        found = merge_hdf5.find_part_files(str(tmp_path))
    assert {x["output_filename"]: list(x["filenames"]) for x in found} == parts, found
    assert os.path.join(str(tmp_path), "d.hdf5") in caplog.text, caplog.text


@pytest.fixture
def restore_loglevel():
    """ `merge_hdf5.main` sets the level of the merger's logger """
    logger = logging.getLogger("NuRadioMC.HDF5-merger")
    level = logger.level
    yield
    logger.setLevel(level)


def test_main(tmp_path, caplog, restore_loglevel):
    inputs = _write_files(tmp_path, [([0], [0]), ([1], [1])])
    output = str(tmp_path / "merged.hdf5")
    merge_hdf5.main([output, *inputs])
    np.testing.assert_array_equal(_read(output)[0]["shower_ids"], [0, 1])
    # an existing output is skipped with an error message, or overwritten with --overwrite
    with caplog.at_level(logging.ERROR, logger="NuRadioMC.HDF5-merger"):
        merge_hdf5.main([output, inputs[0]])
    assert "already exists" in caplog.text
    np.testing.assert_array_equal(_read(output)[0]["shower_ids"], [0, 1])
    merge_hdf5.main([output, inputs[0], "--overwrite", "--loglevel", "warning"])
    np.testing.assert_array_equal(_read(output)[0]["shower_ids"], [0])
    with pytest.raises(SystemExit):
        merge_hdf5.main([output, *inputs, "--loglevel", "print('x')"])


def test_main_directory_overwrite(tmp_path, restore_loglevel):
    parts = _write_part_tree(str(tmp_path))
    for output in parts:
        with h5py.File(output, "w") as f:
            f.attrs["n_events"] = -1
    merge_hdf5.main([str(tmp_path)])
    for output in parts:
        with h5py.File(output, "r") as f:
            assert f.attrs["n_events"] == -1
    merge_hdf5.main([str(tmp_path), "--overwrite"])
    for output, inputs in parts.items():
        with h5py.File(output, "r") as f:
            assert f.attrs["n_events"] == 2 * len(inputs)


def _run_cli(*args, success=True):
    script = os.path.join(os.path.dirname(merge_hdf5.__file__), "merge_hdf5.py")
    # timeout: a pool whose workers fail to start hangs
    result = subprocess.run([sys.executable, script, *args], capture_output=True, text=True, timeout=60)
    assert (result.returncode == 0) == success, result.stderr


@pytest.mark.parametrize("args", [[], ["--cores", "2"]])
def test_cli_directory_mode(tmp_path, args):
    """ ``args``: further command line arguments """
    parts = _write_part_tree(str(tmp_path))
    _run_cli(str(tmp_path), *args)
    for output, inputs in parts.items():
        with h5py.File(output, "r") as f:
            assert f.attrs["n_events"] == 2 * len(inputs), (output, f.attrs["n_events"])


def test_cli_loglevel(tmp_path):
    # level names and numbers are accepted, anything else is an error
    inputs = _write_files(tmp_path, [([0], [0]), ([1], [1])])
    for i, level in enumerate(["DEBUG", "info", "10"]):
        _run_cli(str(tmp_path / f"out{i}.hdf5"), *inputs, "--loglevel", level)
    _run_cli(str(tmp_path / "out_bad.hdf5"), *inputs, "--loglevel", "print('x')", success=False)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
