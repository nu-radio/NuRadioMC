"""
Merges NuRadioMC hdf5 output files into one file and keeps track of the total number of simulated events
(attribute ``n_events``), which the effective volume calculation needs. See `merge_hdf5_files` for the rules.

Command line::

    python merge_hdf5.py <root directory>
    python merge_hdf5.py <output file> <input file 1> [<input file 2> ...]

With a single argument, the script expects the part files of the event generator in subfolders, e.g. one per energy,
``<root directory>/<subfolder>/<name>.hdf5.partNNNN``, and writes ``<root directory>/<name>.hdf5`` per ``<name>``
(see `find_part_files`). With several arguments, the first one is the output file, the others are the input files.

Existing output files are skipped with an error message unless ``--overwrite`` is given. ``--cores N`` merges N output
files in parallel (directory mode); ``--loglevel`` sets the log level (e.g. ``DEBUG``, ``INFO``, ``WARNING``).
"""
import argparse
import glob
import logging
import os
import sys

import h5py
import numpy as np

from NuRadioReco.utilities.logging import deprecated

logger = logging.getLogger("NuRadioMC.HDF5-merger")

str_type_attrs = ['trigger_names', 'NuRadioMC_EvtGen_version', 'NuRadioMC_EvtGen_version_hash', 'NuRadioMC_version',
                  'NuRadioMC_version_hash', 'config']
# attributes the effective volume/area calculation reads from the merged file: they must agree between files
veff_attrs = ['Emin', 'Emax', 'volume', 'area']
# datasets (top level and station groups) holding ids that the merge shifts: dataset name -> top-level id dataset
# whose offsets apply
_ID_DATASETS = {'event_group_ids': 'event_group_ids', 'event_group_id_per_shower': 'event_group_ids',
                'shower_ids': 'shower_ids', 'shower_id': 'shower_ids'}


# attributes recording the files merged into a file, their id offsets (see `_make_ids_unique`) and the smallest
# merged event group id of each file (-1: no rows)
merge_history_attrs = ['merged_files', 'event_group_id_offsets', 'shower_id_offsets', 'event_group_id_starts']


def _make_ids_unique(id_arrays):
    """
    Finds the offsets that make the ids unique across files.

    In merge order, all ids of a file whose smallest id is not above the largest id of the previous files are shifted
    by one offset, so that its smallest id becomes that largest id + 1. Gaps and order of the ids are kept; a file whose
    ids lie above those of all previous files (e.g. the part files of one run) keeps them (offset 0).

    Parameters
    ----------
    id_arrays : list of array
        ids (e.g. ``event_group_ids``) per file, in merge order

    Returns
    -------
    list of int
        offset per file (merged id = original id + offset)
    """
    offsets = []
    max_id = None  # largest id of the previous files, after shifting
    for ids in id_arrays:
        ids = np.asarray(ids)
        offsets.append(0)
        if len(ids):
            if max_id is not None and ids.min() <= max_id:
                offsets[-1] = int(max_id + 1 - ids.min())
            max_id = ids.max() + offsets[-1]
    return offsets


def _merge_history(attrs, filename, start):
    """
    Files merged into the file ``filename`` (with attributes ``attrs``), their id offsets and event group id starts,
    as in `merge_history_attrs`: the file itself with offsets 0 and start ``start`` (its smallest event group id, -1
    if it has no rows) if it is not the output of a merge.
    """
    if 'merged_files' in attrs:
        if all(key in attrs for key in merge_history_attrs):
            return [list(attrs[key]) for key in merge_history_attrs]
        logger.warning(f"{filename} lacks some of the attributes {merge_history_attrs}, listing it as one file")
    return [[os.path.basename(filename)], [0], [0], [start]]


def _attrs_equal(a, b):
    """ Compares two attribute values: numbers with `numpy.allclose` (NaNs equal), anything else (e.g. strings) exactly. """
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
        return np.allclose(a, b, equal_nan=True)
    return np.array_equal(a, b)


def _merge_attributes(attrs_per_file):
    """
    Merges the file attributes.

    The first file that has a key sets its value. Later files must agree on the string attributes
    (``str_type_attrs``, e.g. ``config``) and on ``veff_attrs`` (else `AttributeError`); other differing values warn.
    ``start_event_id`` is not compared, ``trigger_names`` missing or empty so far are taken from a later file.

    Parameters
    ----------
    attrs_per_file : dict
        file attributes (dict) per file name, in merge order

    Returns
    -------
    dict
        merged attributes (``n_events`` not yet summed)
    """
    filenames = list(attrs_per_file)
    attrs = {}
    for f, file_attrs in attrs_per_file.items():
        for key, value in file_attrs.items():
            if key not in attrs:
                attrs[key] = value
            elif key in str_type_attrs:
                if not np.array_equal(attrs[key], value):
                    message = f"attribute {key} of file {filenames[0]} and {f} are different ({attrs[key]} vs. {value}. "
                    logger.error(message)
                    raise AttributeError(message)
            elif key == "start_event_id":
                continue
            elif not _attrs_equal(attrs[key], value):
                if key == "n_events":
                    logger.warning(
                        f"number of events in file {filenames[0]} and {f} are different ({attrs[key]} vs. {value}. We "
                        "keep track of the total number of events, but in case the simulation was performed with "
                        "different settings per file (e.g. different zenith angle bins), the averaging might be effected.")
                elif key in veff_attrs:
                    raise AttributeError(f"attribute {key} of file {filenames[0]} and {f} are different "
                                         f"({attrs[key]} vs. {value}), the merged effective volume would be wrong")
                else:
                    logger.warning(f"attribute {key} of file {filenames[0]} and {f} are different ({attrs[key]} vs. "
                                   f"{value}. Using attribute value of first file, but you have been warned!")

            if (len(attrs.get('trigger_names', [])) == 0) and 'trigger_names' in file_attrs:
                attrs['trigger_names'] = file_attrs['trigger_names']
    return attrs


def _merge_group_attributes(group_attrs_per_file):
    """
    Merges the attributes of each station group: union of the keys, the first file that has a key sets its value,
    differing values of later files warn.

    Parameters
    ----------
    group_attrs_per_file : dict
        per file name (in merge order), the attributes (dict) per station group

    Returns
    -------
    dict
        merged attributes per station group
    """
    filenames = list(group_attrs_per_file)
    group_attrs = {}
    for f, file_group_attrs in group_attrs_per_file.items():
        for group, attrs in file_group_attrs.items():
            if group not in group_attrs:
                group_attrs[group] = dict(attrs)
                continue
            for key, value in attrs.items():
                if key not in group_attrs[group]:
                    logger.warning(f"attribute {key} of group {group} is missing in previous files, using the value of {f}")
                    group_attrs[group][key] = value
                elif not _attrs_equal(group_attrs[group][key], value):
                    logger.warning(
                        f"attribute {key} of group {group} of file {filenames[0]} and {f} are different "
                        f"({group_attrs[group][key]} vs. {value}. Using attribute value of first file, but you have "
                        "been warned!")
    return group_attrs


def find_part_files(root_directory, skip_existing=True):
    """
    Groups the files ``root_directory/<subfolder>/<name>.hdf5.partNNNN`` by ``<name>.hdf5``.

    Files of at most 1000 bytes (e.g. truncated by a failed job) are left out with a warning; their events are then
    missing from ``n_events`` of the merged file.

    Parameters
    ----------
    root_directory : str
        directory containing one subfolder (e.g. per energy) with the part files
    skip_existing : bool, default: True
        leave out (with an error message) the groups whose output file already exists

    Returns
    -------
    list of dict
        per output file ``root_directory/<name>.hdf5``, the keyword arguments of `merge_hdf5_files`
        (``filenames``: sorted part files, ``output_filename``)
    """
    # strip the .partNNNN suffix
    filenames = np.unique([os.path.splitext(f)[0] for f in glob.glob(f"{root_directory}/*/*.hdf5.part*")])
    input_args = []
    for filename in sorted(filenames):
        if os.path.splitext(filename)[1] != '.hdf5':
            continue
        folder, name = os.path.split(filename)
        output_filename = os.path.join(os.path.dirname(folder), name)  # remove subfolder from filename
        if skip_existing and os.path.exists(output_filename):
            logger.error(f'file {output_filename} already exists, skipping')
            continue
        input_files = sorted(glob.glob(filename + '.part*'))
        small = [f for f in input_files if os.path.getsize(f) <= 1000]
        if small:
            logger.warning(f"{len(small)} files were deselected because their filesize was too small, the merged "
                           f"{output_filename} misses their events in n_events: {small}")
        input_args.append({'filenames': [f for f in input_files if f not in small], 'output_filename': output_filename})
    return input_args


def _parse_loglevel(value):
    """ Command line log level: a level name (case-insensitive) or number. """
    if value.isdigit():
        return int(value)
    level = logging.getLevelName(value.upper())
    if not isinstance(level, int):
        raise argparse.ArgumentTypeError(f"unknown log level {value}")
    return level


class _FileInfo:
    """
    What the first pass of `merge_hdf5_files` reads from one input file: everything except the dataset contents
    (apart from the ids in ``ids``).

    Attributes
    ----------
    filename : str
    attrs : dict
        file attributes
    is_empty : bool
        True if the file has no ``triggered`` dataset or no triggered entry; empty files add no rows
    datasets : dict
        (shape, dtype) per top-level dataset
    groups : dict
        per station group, (shape, dtype) per dataset
    group_attrs : dict
        attributes per station group
    ids : dict
        top-level id datasets that the merge may shift (``event_group_ids``, ``shower_ids``; non-empty files only)
    """

    def __init__(self, filename):
        self.filename = filename
        self.attrs = {}
        self.is_empty = True
        self.datasets = {}
        self.groups = {}
        self.group_attrs = {}
        self.ids = {}


def _read_file_info(filename):
    """ First pass of `merge_hdf5_files`: returns the `_FileInfo` of the file. """
    info = _FileInfo(filename)
    with h5py.File(filename, 'r') as fin:
        info.attrs = dict(fin.attrs)
        # empty files might not have the triggered object at all
        info.is_empty = 'triggered' not in fin or np.sum(fin['triggered'][...]) == 0
        for key, obj in fin.items():
            if isinstance(obj, h5py.Group):  # station groups
                info.groups[key] = {key2: (dset.shape, dset.dtype) for key2, dset in obj.items()}
                info.group_attrs[key] = dict(obj.attrs)
            else:
                info.datasets[key] = (obj.shape, obj.dtype)
        if not info.is_empty:
            for key in set(_ID_DATASETS.values()):
                if key in fin:
                    info.ids[key] = fin[key][...]
    return info


def _dataset_layout(name, shapes_dtypes):
    """
    Shape and dtype of the merged dataset ``name`` from the (filename, shape, dtype) of each file that contributes rows:
    rows summed, trailing shape and dtype of the first file.
    """
    _, shape, dtype = shapes_dtypes[0]
    for filename, other_shape, other_dtype in shapes_dtypes[1:]:
        if tuple(other_shape[1:]) != tuple(shape[1:]):
            raise ValueError(f"dataset {name} of {filename} has shape {other_shape}, incompatible with {shape} of "
                             f"{shapes_dtypes[0][0]}")
        if other_dtype != dtype:
            logger.warning(f"dataset {name} of {filename} has dtype {other_dtype}, converting it to {dtype} of "
                           f"{shapes_dtypes[0][0]}")
    return (sum(s[0] for _, s, _ in shapes_dtypes),) + tuple(shape[1:]), dtype


def _merge_datasets(fout, infos, skip_incomplete_stations=False):
    """
    Second pass of `merge_hdf5_files` for the non-empty input files ``infos``: creates the merged datasets and station
    groups in ``fout`` and copies the rows file by file, shifting overlapping ids (see `_make_ids_unique`). Returns the
    offsets per id dataset (``event_group_ids``, ``shower_ids`` if shifted) and file name. ``skip_incomplete_stations``:
    see `merge_hdf5_files`.
    """
    # unique event group ids: the effective volume/area calculation uses them to find multi-station coincidences
    id_offsets = {'event_group_ids': _make_ids_unique([info.ids['event_group_ids'] for info in infos])}
    # shower ids are unique only within the files of one event generation run (issue #292)
    if all('shower_ids' in info.ids for info in infos):
        id_offsets['shower_ids'] = _make_ids_unique([info.ids['shower_ids'] for info in infos])
    # event ids (``event_ids``, ``event_id_per_shower``) are unique only within an event group, so the unique
    # event group ids keep them unique
    for key, offsets in id_offsets.items():
        for info, offset in zip(infos, offsets):
            if offset:
                logger.warning(f"{key} of {info.filename} overlap with previous files; shifted by {offset}")

    # output datasets: path -> (open dataset, offset per index of a file that adds rows). The datasets stay open
    # while the files are copied, so that a chunk filled by several files is compressed and written once (from the
    # chunk cache); reopening them per file grows the output by rewritten chunks.
    outputs = {}

    def create(path, files):
        shape, dtype = _dataset_layout(path, [(infos[i].filename, *shapes_dtypes) for i, shapes_dtypes in files])
        dset = fout.create_dataset(path, shape, dtype=dtype, compression='gzip')
        offsets = np.cumsum([0] + [shapes_dtypes[0][0] for _, shapes_dtypes in files])
        outputs[path] = (dset, dict(zip([i for i, _ in files], offsets)))

    first = infos[0]
    for key in first.datasets:
        logger.info(f"merging key {key}")
        if not all(key in info.datasets for info in infos):
            logger.warning(f"not all files have the key {key}. This key will not be present in the merged file.")
            continue
        create(key, [(i, info.datasets[key]) for i, info in enumerate(infos)])
    for info in infos[1:]:
        for key in info.datasets:
            if key not in first.datasets:
                logger.warning(f"{info.filename} has the key {key}, but {first.filename} has not. This key will not be "
                               "present in the merged file.")

    groups = []  # union of the station groups, in order of first appearance
    for info in infos:
        groups += [group for group in info.groups if group not in groups]
    for group in groups:
        logger.info(f"writing group {group}")
        fout.create_group(group)
        # files with datasets in this group (the writer leaves the group of a station without events empty)
        members = [i for i, info in enumerate(infos) if info.groups.get(group)]
        keys = []  # union of the datasets of the group, in order of first appearance
        for i in members:
            keys += [key for key in infos[i].groups[group] if key not in keys]
        # the datasets of a group are row-aligned, so a file must have all of them or add no rows
        missing = {i: [key for key in keys if key not in infos[i].groups[group]] for i in members}
        incomplete = [i for i in members if missing[i]]
        if incomplete and not skip_incomplete_stations:
            details = "; ".join(f"{infos[i].filename} misses {missing[i]}" for i in incomplete)
            raise ValueError(f"not all files have the same datasets in {group} ({details}). Use "
                             "skip_incomplete_stations=True (command line: --skip-incomplete-stations) to leave out "
                             "the rows of these files for this station.")
        for i in incomplete:
            logger.warning(f"{infos[i].filename} misses the datasets {missing[i]} of {group}: leaving out its rows "
                           "for this station")
        members = [i for i in members if not missing[i]]
        if not members:
            keys = []
        for key in keys:
            create(f"{group}/{key}", [(i, infos[i].groups[group][key]) for i in members])

    # copy the rows file by file; one dataset of one file is in memory at a time
    for i, info in enumerate(infos):
        logger.info(f"copying {info.filename}")
        with h5py.File(info.filename, 'r') as fin:
            for path, (dset, offsets) in outputs.items():
                if i not in offsets:
                    continue
                data = fin[path][...]
                kind = _ID_DATASETS.get(path.split('/')[-1])
                if kind in id_offsets:
                    data = data + id_offsets[kind][i]
                if len(data):
                    dset[offsets[i]:offsets[i] + len(data)] = data
    return {key: {info.filename: offset for info, offset in zip(infos, offsets)} for key, offsets in id_offsets.items()}


def _copy_first_file(fout, filename, n_events_total):
    """ Copies the file ``filename`` into ``fout`` (datasets gzip compressed), with ``n_events`` set to ``n_events_total``. """
    with h5py.File(filename, 'r') as fin:
        for key in fin.attrs:
            if key not in merge_history_attrs:
                fout.attrs[key] = n_events_total if key == "n_events" else fin.attrs[key]
        for key, obj in fin.items():
            if isinstance(obj, h5py.Group):
                g = fout.create_group(key)
                for key2, dset in obj.items():
                    g.create_dataset(key2, dset.shape, dtype=dset.dtype, compression='gzip')[...] = dset[...]
                for key2 in obj.attrs:
                    g.attrs[key2] = obj.attrs[key2]
            else:
                fout.create_dataset(key, obj.shape, dtype=obj.dtype, compression='gzip')[...] = obj[...]


def merge_hdf5_files(filenames, output_filename, overwrite=False, skip_incomplete_stations=False):
    """
    Merges NuRadioMC HDF5 output files into one file.

    Datasets are appended in the order of ``filenames``; ``n_events`` is the sum over all files, which the effective
    volume calculation needs. Files without triggered events count for ``n_events`` but add no rows; if all files are
    empty, the first file is copied.

    Event group ids and shower ids of a file that overlap those of the previous files are shifted (see
    `_make_ids_unique`), in the top-level datasets and in the station groups. The attributes ``merged_files``,
    ``event_group_id_offsets`` and ``shower_id_offsets`` list the original files and their offsets (merged id =
    original id + offset), ``event_group_id_starts`` the smallest merged event group id of each file (-1: no rows);
    for an input file that is itself a merged file, its entries are listed, with the offsets of both merges added.

    Attributes: the first file that has an attribute sets its value. ``config``, ``trigger_names`` and the NuRadioMC
    versions must be equal in all files, as must ``Emin``, ``Emax``, ``volume`` and ``area`` (else `AttributeError`);
    other differing attributes warn. A top-level dataset missing in one of the non-empty files is left out with a
    warning. The merged file has the station groups of all files.

    The merge is written to ``output_filename + '.tmp'``, which is renamed to ``output_filename`` when done.
    The first pass reads all attributes and the ids, the second copies one dataset of one file at a time, so the
    memory use does not grow with the size of the files.

    Parameters
    ----------
    filenames : list of str
        input files; a file given more than once (also via another path) is merged once
    output_filename : str
        merged file
    overwrite : bool, default: False
        overwrite an existing ``output_filename``; if False, an existing file raises `FileExistsError`
    skip_incomplete_stations : bool, default: False
        if a file lacks some of the datasets that other files have in a station group, leave out the rows of that file
        for that station (with a warning); if False, raise `ValueError`. Simulations with the same config write the
        same datasets.
    """
    output_filename = str(output_filename)
    unique_filenames = {}  # real path -> file name
    for f in map(str, filenames):
        if os.path.realpath(f) in unique_filenames:
            logger.warning(f"{f} is the same file as {unique_filenames[os.path.realpath(f)]}, merging it once")
        else:
            unique_filenames[os.path.realpath(f)] = f
    filenames = list(unique_filenames.values())
    if not filenames:
        raise ValueError("no input files given")
    if os.path.realpath(output_filename) in unique_filenames:
        raise ValueError(f"the output file {output_filename} is one of the input files")
    if os.path.exists(output_filename) and not overwrite:
        raise FileExistsError(f"{output_filename} already exists")

    logger.warning(f"merging {len(filenames)} files into {os.path.basename(output_filename)}")
    infos = []
    n_events_total = 0
    for f in filenames:
        logger.info(f"adding file {f}")
        infos.append(_read_file_info(f))
        n_events_total += infos[-1].attrs['n_events']
        logger.debug(f"increasing total number of events by {infos[-1].attrs['n_events']} to {n_events_total}")
        if infos[-1].is_empty:
            logger.info(f"file {f} contains no events")
    attrs = _merge_attributes({info.filename: {key: value for key, value in info.attrs.items()
                                               if key not in merge_history_attrs} for info in infos})
    attrs['n_events'] = n_events_total
    group_attrs = _merge_group_attributes({info.filename: info.group_attrs for info in infos})
    non_empty = [info for info in infos if not info.is_empty]

    tmp_filename = output_filename + '.tmp'
    try:
        with h5py.File(tmp_filename, 'w') as fout:
            id_offsets = {}
            if non_empty:
                id_offsets = _merge_datasets(fout, non_empty, skip_incomplete_stations)
                for group in fout:
                    if isinstance(fout[group], h5py.Group):
                        for key, value in group_attrs[group].items():
                            fout[group].attrs[key] = value
                for key, value in attrs.items():
                    fout.attrs[key] = value
            else:
                logger.warning("All files are empty. Copying content of first file to output file and keeping track "
                               "of total number of simulated events.")
                _copy_first_file(fout, filenames[0], n_events_total)
            # the original files and their offsets; the offsets of a file merged before add up
            merged_files, egid_offsets, sid_offsets, egid_starts = [], [], [], []
            for info in infos:
                start = -1 if info.is_empty else int(np.min(info.ids['event_group_ids']))
                names, egid_offsets_f, sid_offsets_f, egid_starts_f = _merge_history(info.attrs, info.filename, start)
                merged_files += names
                egid_offset = id_offsets.get('event_group_ids', {}).get(info.filename, 0)
                sid_offset = id_offsets.get('shower_ids', {}).get(info.filename, 0)
                egid_offsets += [x + egid_offset for x in egid_offsets_f]
                sid_offsets += [x + sid_offset for x in sid_offsets_f]
                egid_starts += [x + egid_offset if x >= 0 else -1 for x in egid_starts_f]
            fout.attrs['merged_files'] = merged_files
            fout.attrs['event_group_id_offsets'] = np.array(egid_offsets, dtype=np.int64)
            fout.attrs['shower_id_offsets'] = np.array(sid_offsets, dtype=np.int64)
            fout.attrs['event_group_id_starts'] = np.array(egid_starts, dtype=np.int64)
        os.replace(tmp_filename, output_filename)
    except BaseException:
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)
        raise


@deprecated("merge2 is deprecated, use merge_hdf5_files (which does not overwrite an existing output by default)")
def merge2(filenames, output_filename):
    """ Deprecated: `merge_hdf5_files` with ``overwrite=True``. """
    merge_hdf5_files(filenames, output_filename, overwrite=True)


def _merge_job(kwargs):
    """ `merge_hdf5_files` with keyword arguments; module level, so that a multiprocessing pool can pickle it. """
    merge_hdf5_files(**kwargs)


def main(argv=None):
    """ Command line interface, see the module docstring; ``argv``: arguments (default: `sys.argv`). """
    parser = argparse.ArgumentParser(description='Merge NuRadioMC hdf5 output files.')
    parser.add_argument('files', nargs='+',
                        help='root directory of the part files, or the output file followed by the input files')
    parser.add_argument('--loglevel', metavar='level', type=_parse_loglevel,
                        help='loglevel set to either DEBUG, INFO, or WARNING (or another level name or number)')
    parser.add_argument('--cores', default=1, type=int,
                        help='number of output files to merge in parallel (directory mode)')
    parser.add_argument('--overwrite', action='store_true',
                        help='overwrite existing output files (default: skip them with an error message)')
    parser.add_argument('--skip-incomplete-stations', action='store_true',
                        help='leave out the rows of a file for a station whose group lacks datasets that other files '
                             'have (default: error)')
    args = parser.parse_args(argv)

    if args.loglevel is not None:
        logger.setLevel(args.loglevel)

    if len(args.files) == 1:
        root_directory = args.files[0]
        if not os.path.isdir(root_directory):
            sys.exit(f"{root_directory} is not a directory.")

        input_args = find_part_files(root_directory, skip_existing=not args.overwrite)
        for kwargs in input_args:
            kwargs.update(overwrite=args.overwrite, skip_incomplete_stations=args.skip_incomplete_stations)

        if args.cores == 1:
            for kwargs in input_args:
                _merge_job(kwargs)
        else:
            from multiprocessing import Pool
            logger.warning(f"running {len(input_args)} job on {args.cores} cores")
            # a failed job raises here and stops the pool
            with Pool(args.cores, initializer=logger.setLevel, initargs=(logger.level,)) as p:
                p.map(_merge_job, input_args)

    else:
        output_filename = args.files[0]
        if os.path.exists(output_filename) and not args.overwrite:
            logger.error(f'file {output_filename} already exists, skipping')
        else:
            merge_hdf5_files(args.files[1:], output_filename, overwrite=args.overwrite,
                             skip_incomplete_stations=args.skip_incomplete_stations)


if __name__ == "__main__":
    main()
