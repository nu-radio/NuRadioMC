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
import dataclasses
import functools
import glob
import logging
import os
import sys

import h5py
import numpy as np

from NuRadioReco.utilities.logging import deprecated

logger = logging.getLogger("NuRadioMC.HDF5-merger")

# attributes that must be equal in all files: the NuRadioMC versions and the config
str_type_attrs = ['NuRadioMC_EvtGen_version', 'NuRadioMC_EvtGen_version_hash', 'NuRadioMC_version',
                  'NuRadioMC_version_hash', 'config']
# attributes the effective volume/area calculation reads from the merged file: they must agree between files
veff_attrs = ['Emin', 'Emax', 'volume', 'area']
# datasets (top level and station groups) with one column per trigger in ``trigger_names``; mirrors
# ``output_writer_hdf5.py`` (``multiple_triggers``/``trigger_times`` extended per new trigger around :143-155, the per
# event ones written around :239-240 and listed in ``keys_per_event`` around :496)
_PER_TRIGGER_DATASETS = ['multiple_triggers', 'trigger_times', 'multiple_triggers_per_event', 'trigger_times_per_event']
# datasets (top level and station groups) holding ids that the merge shifts: dataset name -> top-level id dataset
# whose offsets apply; mirrors the ids written by ``output_writer_hdf5.py`` around :171-172, :220 and :256-257
_ID_DATASETS = {'event_group_ids': 'event_group_ids', 'event_group_id_per_shower': 'event_group_ids',
                'shower_ids': 'shower_ids', 'shower_id': 'shower_ids'}


# attributes recording the files merged into a file, their id offsets (see `_make_ids_unique`) and the smallest
# merged event group id of each file (-1: no rows)
merge_history_attrs = ['merged_files', 'event_group_id_offsets', 'shower_id_offsets', 'event_group_id_starts']


def _make_ids_unique(id_arrays):
    """
    Finds the offset of the ids of each file (rule: see `merge_hdf5_files`).

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


def _ordered_union(sequences):
    """ Union of the items of ``sequences``, in order of first appearance. """
    return list(dict.fromkeys(item for sequence in sequences for item in sequence))


def _attrs_equal(a, b):
    """ Compares two attribute values: numbers with `numpy.allclose` (NaNs equal), anything else (e.g. strings) exactly. """
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False

    if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
        return np.allclose(a, b, equal_nan=True)

    return np.array_equal(a, b)


def _merge_attrs(attrs_per_file, label, must_match=(), skip=()):
    """
    Merges attributes with the rules of `merge_hdf5_files`.

    Parameters
    ----------
    attrs_per_file : dict
        attributes (dict) per file name, in merge order
    label : str
        what the attributes belong to, for messages (e.g. ``"of group station_101"``)
    must_match : sequence of str
        keys whose values must agree (else `AttributeError`); other differing values warn
    skip : sequence of str
        keys that are not compared (the first value is kept)

    Returns
    -------
    dict
        merged attributes
    """
    attrs, source = {}, {}  # value and file name per key

    for f, file_attrs in attrs_per_file.items():
        for key, value in file_attrs.items():
            if key not in attrs:
                if source:
                    logger.info(f"attribute {key} {label} is missing in previous files, using the value of {f}")
                attrs[key], source[key] = value, f
            elif key not in skip and not _attrs_equal(attrs[key], value):
                message = f"attribute {key} {label} differs between {source[key]} and {f} ({attrs[key]} vs. {value})"
                if key in must_match:
                    raise AttributeError(message)

                logger.warning(f"{message}, using the value of {source[key]}")

    return attrs


def _merge_trigger_names(attrs_per_file):
    """
    Merges the ``trigger_names`` of the files by name (rule: see `merge_hdf5_files`).

    Parameters
    ----------
    attrs_per_file : dict
        file attributes (dict) per file name, in merge order

    Returns
    -------
    trigger_names : array, list of str or None
        merged ``trigger_names``: the value of the first file if all files agree, else the union; None if no file has
        ``trigger_names``
    columns : dict
        per file name whose ``trigger_names`` differ from the merged ones, the merged column of each of its triggers
    """
    names_per_file = {}

    for f, attrs in attrs_per_file.items():
        if 'trigger_names' in attrs:
            names_per_file[f] = [n.decode() if isinstance(n, bytes) else str(n)
                                 for n in np.atleast_1d(attrs['trigger_names'])]

    if not names_per_file:
        return None, {}

    union = _ordered_union(names_per_file.values())
    columns = {f: np.array([union.index(n) for n in names], dtype=int)
               for f, names in names_per_file.items() if names != union}
    if not columns:
        return attrs_per_file[next(iter(names_per_file))]['trigger_names'], columns

    logger.warning(f"the trigger names of {len(columns)} files differ from the merged trigger names {union}; "
                   "their per-trigger datasets are re-indexed, columns of triggers missing in a file are False / NaN")

    for f in columns:
        logger.info(f"trigger names of {f}: {names_per_file[f]}")

    return union, columns


def find_part_files(root_directory, skip_existing=True):
    """
    Groups the files ``root_directory/<subfolder>/<name>.hdf5.partNNNN`` by ``<name>.hdf5``.

    Files of at most 1000 bytes (e.g. truncated by a failed job) are left out with a warning; their events are then
    missing from ``n_events`` of the merged file. A name whose part files are all that small is left out with an
    error message. A name in several subfolders gives one entry per subfolder, all with the same output file, and a
    warning.

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
    parts = {}  # part files per name with the .partNNNN suffix stripped

    for f in sorted(glob.glob(f"{root_directory}/*/*.hdf5.part*")):
        if os.path.splitext(os.path.splitext(f)[0])[1] == '.hdf5':
            parts.setdefault(os.path.splitext(f)[0], []).append(f)

    folders_per_name = {}

    for filename in parts:
        folder, name = os.path.split(filename)
        folders_per_name.setdefault(name, []).append(folder)

    for name, folders in folders_per_name.items():
        if len(folders) > 1:
            logger.warning(f"the part files of {name} are in several subfolders ({', '.join(sorted(folders))}), "
                           f"which all map to the same output file {os.path.join(root_directory, name)}")

    input_args = []

    for filename, input_files in sorted(parts.items()):
        folder, name = os.path.split(filename)
        output_filename = os.path.join(os.path.dirname(folder), name)  # remove subfolder from filename
        if skip_existing and os.path.exists(output_filename):
            logger.error(f'file {output_filename} already exists, skipping')
            continue

        small = {f for f in input_files if os.path.getsize(f) <= 1000}
        if len(small) == len(input_files):
            logger.error(f"all part files of {output_filename} are too small (at most 1000 bytes), skipping")
            continue

        if small:
            logger.warning(f"{len(small)} files were deselected because their filesize was too small, the merged "
                           f"{output_filename} misses their events in n_events: {sorted(small)}")
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


@dataclasses.dataclass
class _FileInfo:
    """
    What the first pass of `merge_hdf5_files` reads from one input file: everything except the dataset contents
    (apart from the id ranges in ``ids``).

    Attributes
    ----------
    filename : str
    attrs : dict
        file attributes
    is_empty : bool
        empty file (see `merge_hdf5_files`)
    datasets : dict
        (shape, dtype) per top-level dataset
    groups : dict
        per station group, (shape, dtype) per dataset
    group_attrs : dict
        attributes per station group
    ids : dict
        smallest and largest value of the top-level id datasets that the merge may shift (``event_group_ids``,
        ``shower_ids``; non-empty files only), all that `_make_ids_unique` needs
    """

    filename: str
    attrs: dict
    is_empty: bool
    datasets: dict
    groups: dict
    group_attrs: dict
    ids: dict


def _min_max(ids):
    """ Smallest and largest of ``ids`` (empty: empty array). """
    return np.array([ids.min(), ids.max()]) if len(ids) else ids


def _read_file_info(filename):
    """ First pass of `merge_hdf5_files`: returns the `_FileInfo` of the file. """
    with h5py.File(filename, 'r') as fin:
        # empty files might not have the triggered object at all
        is_empty = 'triggered' not in fin or not np.any(fin['triggered'][...])
        groups = {key: obj for key, obj in fin.items() if isinstance(obj, h5py.Group)}  # station groups

        return _FileInfo(
            filename=filename,
            attrs=dict(fin.attrs),
            is_empty=is_empty,
            datasets={key: (obj.shape, obj.dtype) for key, obj in fin.items() if key not in groups},
            groups={key: {key2: (dset.shape, dset.dtype) for key2, dset in obj.items()} for key, obj in groups.items()},
            group_attrs={key: dict(obj.attrs) for key, obj in groups.items()},
            ids={} if is_empty else {key: _min_max(fin[key][...]) for key in set(_ID_DATASETS.values()) if key in fin})


def _reindex_triggers(data, columns, n_triggers):
    """ Per-trigger dataset ``data`` with its columns moved to ``columns`` of ``n_triggers``; others False / NaN. """
    reindexed = np.full((len(data), n_triggers), np.nan if data.dtype.kind == 'f' else 0, data.dtype)
    reindexed[:, columns] = data

    return reindexed


def _merge_datasets(fout, infos, trigger_columns, n_triggers):
    """
    Second pass of `merge_hdf5_files` (rules there) for the non-empty input files ``infos``: creates the merged
    datasets and station groups in ``fout`` and copies the rows file by file. ``trigger_columns`` and ``n_triggers``:
    re-indexing of the per-trigger datasets (see `_merge_trigger_names`). Returns the id offsets per id dataset
    (``event_group_ids``, ``shower_ids`` if shifted) and file name.
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

    # output datasets: path -> (open dataset, {index of a file that adds rows: (offset, row transform or None)})
    outputs = {}

    def create(path, files):
        """ Creates the merged dataset ``path`` from the (file index, (shape, dtype)) of the files that add rows. """
        name = path.split('/')[-1]
        transforms = {}

        for i, (shape, _) in files:
            id_offset = id_offsets[_ID_DATASETS[name]][i] if _ID_DATASETS.get(name) in id_offsets else 0
            if id_offset:
                transforms[i] = functools.partial(np.add, id_offset)
            elif name in _PER_TRIGGER_DATASETS and infos[i].filename in trigger_columns:
                columns = trigger_columns[infos[i].filename]
                if len(shape) != 2 or shape[1] != len(columns):
                    raise ValueError(f"dataset {path} of {infos[i].filename} has shape {shape}, but the file has "
                                     f"{len(columns)} trigger names")

                transforms[i] = functools.partial(_reindex_triggers, columns=columns, n_triggers=n_triggers)

        # trailing shape after re-indexing the trigger columns
        trailing = [(n_triggers,) if i in transforms and name in _PER_TRIGGER_DATASETS else tuple(shape[1:])
                    for i, (shape, _) in files]
        (i0, (_, dtype)) = files[0]

        for (i, (shape, other_dtype)), other_trailing in zip(files[1:], trailing[1:]):
            if other_trailing != trailing[0]:
                raise ValueError(f"dataset {path} of {infos[i].filename} has shape {shape}, incompatible with "
                                 f"{files[0][1][0]} of {infos[i0].filename}")

            if other_dtype != dtype:
                logger.warning(f"dataset {path} of {infos[i].filename} has dtype {other_dtype}, converting it to "
                               f"{dtype} of {infos[i0].filename}")

        rows = [shape[0] for _, (shape, _) in files]
        dset = fout.create_dataset(path, (sum(rows),) + trailing[0], dtype=dtype, compression='gzip')
        offsets = np.cumsum([0] + rows)
        outputs[path] = (dset, {i: (offset, transforms.get(i)) for (i, _), offset in zip(files, offsets)})

    for key in _ordered_union(info.datasets for info in infos):
        logger.info(f"merging key {key}")
        if not all(key in info.datasets for info in infos):
            logger.warning(f"not all files have the key {key}. This key will not be present in the merged file.")
            continue

        create(key, [(i, info.datasets[key]) for i, info in enumerate(infos)])

    for group in _ordered_union(info.groups for info in infos):
        logger.info(f"writing group {group}")
        fout.create_group(group)
        # files with datasets in this group (the writer leaves the group of a station without events empty)
        members = [i for i, info in enumerate(infos) if info.groups.get(group)]
        keys = _ordered_union(infos[i].groups[group] for i in members)
        # the datasets of a group are row-aligned, so a file must have all of them or add no rows
        missing = {i: [key for key in keys if key not in infos[i].groups[group]] for i in members}
        incomplete = [i for i in members if missing[i]]
        if incomplete:
            details = "; ".join(f"{infos[i].filename} misses {missing[i]}" for i in incomplete)
            raise ValueError(f"not all files have the same datasets in {group} ({details}); files simulated with the "
                             "same config write the same datasets")

        for key in keys:
            create(f"{group}/{key}", [(i, infos[i].groups[group][key]) for i in members])

    # copy the rows file by file; one dataset of one file is in memory at a time
    for i, info in enumerate(infos):
        logger.info(f"copying {info.filename}")
        with h5py.File(info.filename, 'r') as fin:
            for path, (dset, per_file) in outputs.items():
                if i not in per_file:
                    continue

                offset, transform = per_file[i]
                data = fin[path][...]
                if transform is not None:
                    data = transform(data)
                if len(data):
                    dset[offset:offset + len(data)] = data

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


def merge_hdf5_files(filenames, output_filename, overwrite=False):
    """
    Merges NuRadioMC HDF5 output files into one file.

    Rules:

    * Rows: datasets are appended in the order of ``filenames``. Files without triggered events (no ``triggered``
      dataset or none True) are empty: they count for ``n_events`` but add no rows. If all files are empty, the first
      file is copied (with the summed ``n_events``).
    * Datasets: a top-level dataset missing in one of the non-empty files is left out with a warning. The merged file
      has the station groups of all non-empty files. A file without a station's group, or with an empty one (the
      writer's output for a station without events), adds no rows to it; a file with rows for a station must have
      all datasets the other files have in that group (else `ValueError`), as files simulated with the same config
      do.
    * Ids: all event group ids (shower ids) of a file whose smallest id is not above the largest id of the previous
      files are shifted by one offset, so that its smallest id becomes that largest id + 1, in the top-level datasets
      and in the station groups. Gaps and order are kept; a file whose ids lie above those of all previous files
      (e.g. the part files of one run) keeps them. Event ids are unique only within an event group and are kept.
    * Merged files: the attributes ``merged_files`` (file names without directory), ``event_group_id_offsets`` and
      ``shower_id_offsets`` list the input files in merge order, including empty ones (offset 0), and their offsets
      (merged id = original id + offset); ``event_group_id_starts`` the smallest merged event group id of each file
      (-1: no rows). An input file that is itself a merged file is replaced by its entries, with the offsets (and the
      starts) shifted by those of this merge, so that they refer to the original files.
    * Attributes (file and station groups): the first file that has an attribute sets its value, differing values of
      later files warn. ``config`` and the NuRadioMC versions must be equal in all files, as must ``Emin``, ``Emax``,
      ``volume`` and ``area`` (else `AttributeError`). ``n_events`` is the sum over all files, which the effective
      volume calculation needs; ``start_event_id`` is not compared.
    * ``trigger_names`` are merged by name: the writer adds a trigger only once it fires, so a trigger that never
      fired in a file is missing from its names. The merged names are the union, in order of first appearance, over
      the non-empty files (over all files if all are empty); the per-trigger datasets (e.g. ``multiple_triggers``) of
      a file with other names are re-indexed to the merged columns, with False / NaN for its missing triggers.

    The merge is written to ``output_filename + '.tmp'``, which is renamed to ``output_filename`` when done.
    The first pass reads all attributes and the ids, the second copies one dataset of one file at a time (instead of
    holding all files in memory).

    Parameters
    ----------
    filenames : list of str
        input files; a file given more than once (also via another path) is merged once
    output_filename : str
        merged file
    overwrite : bool, default: False
        overwrite an existing ``output_filename``; if False, an existing file raises `FileExistsError`
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

    attrs_per_file = {info.filename: info.attrs for info in infos}
    attrs = _merge_attrs({f: {key: value for key, value in file_attrs.items() if key not in merge_history_attrs}
                          for f, file_attrs in attrs_per_file.items()}, "of the files",
                         must_match=str_type_attrs + veff_attrs, skip=['n_events', 'start_event_id', 'trigger_names'])
    attrs['n_events'] = n_events_total
    non_empty = [info for info in infos if not info.is_empty]
    trigger_names, trigger_columns = _merge_trigger_names(
        {info.filename: info.attrs for info in non_empty or infos})
    attrs.pop('trigger_names', None)
    if trigger_names is not None:
        attrs['trigger_names'] = trigger_names
    group_attrs = {group: _merge_attrs({info.filename: info.group_attrs[group] for info in infos
                                        if group in info.group_attrs}, f"of group {group}")
                   for group in _ordered_union(info.group_attrs for info in infos)}

    tmp_filename = output_filename + '.tmp'
    try:
        # no chunk cache: a chunk that several files fill is then compressed and written per file, which grows
        # the output slightly (+0.1 % for 80 files of 1.4 MB, more for small files), but the cache made the memory
        # use grow with the number of files (measured: 80 files, peak RSS 835 MB with the default 1 MiB per
        # dataset, 430 MB without).
        # HDF5 keeps attributes in the object header, limited to 64 kB per attribute in the default (earliest) file
        # format: the merge history attributes (one entry per original file) would fail beyond about 4000 files. The
        # 1.8 file format (readable by HDF5 >= 1.8) stores large attributes outside the header.
        with h5py.File(tmp_filename, 'w', rdcc_nbytes=0, libver=('v108', 'latest')) as fout:
            id_offsets = {}
            if non_empty:
                n_triggers = 0 if trigger_names is None else len(trigger_names)
                id_offsets = _merge_datasets(fout, non_empty, trigger_columns, n_triggers)

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
    """ Command line interface, see the module docstring; ``argv``: arguments (default: ``sys.argv``). """
    parser = argparse.ArgumentParser(description='Merge NuRadioMC hdf5 output files.')
    parser.add_argument('files', nargs='+',
                        help='root directory of the part files, or the output file followed by the input files')

    parser.add_argument('--loglevel', metavar='level', type=_parse_loglevel,
                        help='loglevel set to either DEBUG, INFO, or WARNING (or another level name or number)')

    parser.add_argument('--cores', default=1, type=int,
                        help='number of output files to merge in parallel (directory mode)')

    parser.add_argument('--overwrite', action='store_true',
                        help='overwrite existing output files (default: skip them with an error message)')

    args = parser.parse_args(argv)

    if args.loglevel is not None:
        logger.setLevel(args.loglevel)

    if len(args.files) == 1:
        root_directory = args.files[0]
        if not os.path.isdir(root_directory):
            sys.exit(f"{root_directory} is not a directory.")

        input_args = find_part_files(root_directory, skip_existing=not args.overwrite)

        for kwargs in input_args:
            kwargs.update(overwrite=args.overwrite)

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
        try:
            merge_hdf5_files(args.files[1:], args.files[0], overwrite=args.overwrite)
        except FileExistsError as e:
            logger.error(f'{e}, skipping')


if __name__ == "__main__":
    main()
