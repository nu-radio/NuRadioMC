"""
Test of ``NuRadioMC.utilities.merge_hdf5.merge2``: merging files whose event group ids and shower ids overlap
(e.g. independently generated input files) must give unique ids that stay consistent between the top level and
the station groups (issue #292).
"""
import os
import tempfile

import h5py
import numpy as np

from NuRadioMC.utilities.merge_hdf5 import merge2


def _write_file(filename, event_group_ids, shower_ids):
    """ Minimal NuRadioMC output file: one station that sees every shower, one event per event group. """
    event_group_ids = np.asarray(event_group_ids)
    with h5py.File(filename, "w") as f:
        f.attrs["n_events"] = len(np.unique(event_group_ids))
        f["event_group_ids"] = event_group_ids
        f["shower_ids"] = np.asarray(shower_ids)
        f["triggered"] = np.ones(len(shower_ids), dtype=bool)
        sg = f.create_group("station_101")
        sg["event_group_ids"] = np.unique(event_group_ids)
        sg["event_ids"] = np.zeros(len(np.unique(event_group_ids)), dtype=int)
        sg["shower_id"] = np.asarray(shower_ids)
        sg["event_group_id_per_shower"] = event_group_ids


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
                                                 "station_101/shower_id", "station_101/event_group_id_per_shower"]}


def test_merge_shower_ids_unique():
    out = _merge([([1, 2, 2], [0, 1, 2]), ([1, 2, 2], [0, 1, 2]), ([5], [7])])
    assert len(np.unique(out["shower_ids"])) == 7, out["shower_ids"]
    # the station group references the same (renamed) showers as the top level
    np.testing.assert_array_equal(out["station_101/shower_id"], out["shower_ids"])


def test_merge_event_group_id_zero_unique():
    # only event group id 0 overlaps
    out = _merge([([0, 1], [0, 1]), ([0, 2], [2, 3])])
    assert len(np.unique(out["event_group_ids"])) == 4, out["event_group_ids"]
    np.testing.assert_array_equal(np.sort(out["station_101/event_group_ids"]), np.unique(out["event_group_ids"]))


if __name__ == "__main__":
    test_merge_shower_ids_unique()
    test_merge_event_group_id_zero_unique()
    print("OK")
