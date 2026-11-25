import glob
import os
import sys
import numpy as np
from collections import OrderedDict
import h5py
import argparse
import os
import logging
import math
logger = logging.getLogger("NuRadioMC.HDF5-merger")

str_type_attrs = ['trigger_names', 'NuRadioMC_EvtGen_version', 'NuRadioMC_EvtGen_version_hash', 'NuRadioMC_version', 'NuRadioMC_version_hash', 'config']
# attributes the effective volume/area calculation reads from the merged file: they must agree between files
veff_attrs = ['Emin', 'Emax', 'volume', 'area']


# attributes recording the files merged into a file and their id offsets (see `_make_ids_unique`)
merge_history_attrs = ['merged_files', 'event_group_id_offsets', 'shower_id_offsets']


def _make_ids_unique(data, groups, filenames, key, group_keys):
    """
    Shifts the ids of files that overlap the ids of previous files, in place, so that the ids are unique across files.

    In merge order, all ids of a file whose smallest id is not above the largest id of the previous files are shifted
    by one offset, so that its smallest id becomes that largest id + 1. Gaps and order of the ids are kept; a file whose
    ids lie above those of all previous files (e.g. the part files of one run) keeps them. The top-level dataset
    ``key`` and the datasets ``group_keys`` of all station groups of the file are shifted alike.

    Parameters
    ----------
    data : dict
        top-level datasets per file name
    groups : dict
        station groups (dict of datasets) per file name
    filenames : list of str
        files to process, in merge order
    key : str
        name of the top-level dataset holding the ids
    group_keys : list of str
        names of the station-group datasets holding the same ids

    Returns
    -------
    dict
        offset per file name (merged id = original id + offset)
    """
    offsets = {}
    max_id = None
    for f in filenames:
        ids = data[f][key]
        offsets[f] = 0
        if max_id is not None and ids.min() <= max_id:
            offsets[f] = int(max_id + 1 - ids.min())
            for array in [ids] + [groups[f][g][k] for g in groups[f] for k in group_keys if k in groups[f][g]]:
                array += offsets[f]
            logger.warning(f"{key} of {f} overlap with previous files; shifted by {offsets[f]}")
        max_id = ids.max()
    return offsets


def _merge_history(fin, filename):
    """
    Files merged into the open file ``fin`` (named ``filename``) and their id offsets, as in `merge_history_attrs`:
    ``fin`` itself with offsets 0 if it is not the output of a merge.
    """
    if 'merged_files' in fin.attrs:
        return [list(fin.attrs[key]) for key in merge_history_attrs]
    return [[os.path.basename(filename)], [0], [0]]


def _attrs_equal(a, b):
    """ Compares two attribute values: numbers with `numpy.allclose` (NaNs equal), anything else (e.g. strings) exactly. """
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
        return np.allclose(a, b, equal_nan=True)
    return np.array_equal(a, b)


def find_part_files(root_directory):
    """
    Groups the files ``root_directory/<subfolder>/<name>.hdf5.part*`` by ``<name>.hdf5``.

    Files of at most 1000 bytes are skipped, as are groups whose output file already exists.

    Parameters
    ----------
    root_directory : str
        directory containing one subfolder (e.g. per energy) with the part files

    Returns
    -------
    list of dict
        per output file ``root_directory/<name>.hdf5``, the keyword arguments of `merge2`
        (``filenames``: sorted part files, ``output_filename``)
    """
    # strip the .partNNNN suffix
    filenames = np.unique([os.path.splitext(f)[0] for f in glob.glob(f"{root_directory}/*/*.hdf5.part*")])
    input_args = []
    for filename in sorted(filenames):
        if(os.path.splitext(filename)[1] == '.hdf5'):
            d = os.path.split(filename)
            a, b = os.path.split(d[0])
            output_filename = os.path.join(a, d[1])  # remove subfolder from filename
            if(os.path.exists(output_filename)):
                logger.error('file {} already exists, skipping'.format(output_filename))
            else:
                input_files = np.array(sorted(glob.glob(filename + '.part*')))
                mask = np.array([os.path.getsize(x) > 1000 for x in input_files], dtype=bool)
                if(np.sum(~mask)):
                    logger.warning("{:d} files were deselected because their filesize was to small".format(np.sum(~mask)))
                input_args.append({'filenames': input_files[mask], 'output_filename': output_filename})
    return input_args


def _parse_loglevel(value):
    """ Command line log level: a level name (case-insensitive) or number. """
    if value.isdigit():
        return int(value)
    level = logging.getLevelName(value.upper())
    if not isinstance(level, int):
        raise argparse.ArgumentTypeError(f"unknown log level {value}")
    return level


def merge2(filenames, output_filename):
    unique_filenames = {}  # real path -> first file name with it
    for f in filenames:
        if os.path.realpath(f) in unique_filenames:
            logger.warning(f"{f} is the same file as {unique_filenames[os.path.realpath(f)]}, merging it once")
        else:
            unique_filenames[os.path.realpath(f)] = f
    filenames = list(unique_filenames.values())
    logger.warning(f"merging {len(filenames)} files into {os.path.basename(output_filename)}")
    data = OrderedDict()
    attrs = OrderedDict()
    groups = OrderedDict()
    group_attrs = OrderedDict()
    history = OrderedDict()
    non_empty_filenames = []
    n_events_total = 0
    event_group_id_offsets = {}
    shower_id_offsets = {}

    for f in filenames:
        logger.info("adding file {}".format(f))
        fin = h5py.File(f, 'r')
        n_events_total += fin.attrs['n_events']
        logger.debug(f"increasing total number of events by {fin.attrs['n_events']:d} to {n_events_total:d} ")

        # empty files might not have the triggered object at all
        if('triggered' not in fin.keys() or  ('triggered' in fin.keys() and np.sum(np.array(fin['triggered'])) == 0)):
            logger.info(f"file {f} contains no events")
        else:
            non_empty_filenames.append(f)
            logger.debug(f"file {f} contains {np.sum(np.array(fin['triggered']))} triggered events.")

        history[f] = _merge_history(fin, f)
        data[f] = {}
        groups[f] = {}

        for key in fin:
            if isinstance(fin[key], h5py._hl.group.Group):  # loop through station groups
                groups[f][key] = {}
                for key2 in fin[key]:
                    groups[f][key][key2] = fin[key][key2][...]
                if(key not in group_attrs):
                    group_attrs[key] = {}
                    for key2 in fin[key].attrs:
                        group_attrs[key][key2] = fin[key].attrs[key2]
                else:
                    for key2 in fin[key].attrs:
                        if key2 not in group_attrs[key]:
                            logger.warning(f"attribute {key2} of group {key} is missing in previous files, using the value of {f}")
                            group_attrs[key][key2] = fin[key].attrs[key2]
                        elif not _attrs_equal(group_attrs[key][key2], fin[key].attrs[key2]):
                            logger.warning(
                                f"attribute {key2} of group {key} of file {filenames[0]} and {f} are different "
                                f"({group_attrs[key][key2]} vs. {fin[key].attrs[key2]}. Using attribute value of "
                                "first file, but you have been warned!")
            else:
                data[f][key] = fin[key][...]

        for key in fin.attrs:
            if key in merge_history_attrs:  # combined per file below
                continue
            if(key not in attrs):
                attrs[key] = fin.attrs[key]
            else:
                if(key not in str_type_attrs):
                    if not _attrs_equal(attrs[key], fin.attrs[key]):
                        if(key == "n_events"):
                            logger.warning(f"number of events in file {filenames[0]} and {f} are different ({attrs[key]} vs. "
                            f"{fin.attrs[key]}. We keep track of the total number of events, but in case the simulation was "
                            "performed with different settings per file (e.g. different zenith angle bins), the averaging might be effected.")
                        elif(key == "start_event_id"):
                            continue
                        elif key in veff_attrs:
                            raise AttributeError(f"attribute {key} of file {filenames[0]} and {f} are different "
                                                 f"({attrs[key]} vs. {fin.attrs[key]}), the merged effective volume would be wrong")
                        else:
                            logger.warning(f"attribute {key} of file {filenames[0]} and {f} are different ({attrs[key]} vs. "
                            f"{fin.attrs[key]}. Using attribute value of first file, but you have been warned!")
                else:
                    if not np.array_equal(attrs[key], fin.attrs[key]):
                        logger.error(f"attribute {key} of file {filenames[0]} and {f} are different ({attrs[key]} vs. {fin.attrs[key]}. ")
                        raise AttributeError(f"attribute {key} of file {filenames[0]} and {f} are different ({attrs[key]} vs. {fin.attrs[key]}. ")

            if((('trigger_names' not in attrs) or (len(attrs['trigger_names']) == 0)) and 'trigger_names' in fin.attrs):
                attrs['trigger_names'] = fin.attrs['trigger_names']

        fin.close()

    # create data sets
    logger.info("creating data sets")
    fout = h5py.File(output_filename, 'w')
    if(len(non_empty_filenames)):
        # unique event group ids: the effective volume/area calculation uses them to find multi-station coincidences
        # (shifting: see `_make_ids_unique`)
        event_group_id_offsets = _make_ids_unique(data, groups, non_empty_filenames, 'event_group_ids',
                                                  ['event_group_ids', 'event_group_id_per_shower'])
        # shower ids are unique only within the files of one event generation run (issue #292)
        if all('shower_ids' in data[f] for f in non_empty_filenames):
            shower_id_offsets = _make_ids_unique(data, groups, non_empty_filenames, 'shower_ids', ['shower_id'])
        # event ids (``event_ids``, ``event_id_per_shower``) are unique only within an event group, so the unique
        # event group ids keep them unique

        keys = data[non_empty_filenames[0]]
        for key in keys:
            logger.info(f"merging key {key}")
            all_files_have_key = True
            for f in non_empty_filenames:
                if(not key in data[f]):
                    logger.debug(f"key {key} not in {f}")
                    all_files_have_key = False
            if(not all_files_have_key):
                logger.warning(f"not all files have the key {key}. This key will not be present in the merged file.")
                continue
            shape = list(data[non_empty_filenames[0]][key].shape)
            shape[0] = sum(len(data[f][key]) for f in non_empty_filenames)  # skipped files add no rows

            tmp = np.zeros(shape, dtype=data[non_empty_filenames[0]][key].dtype)

            i = 0
            for f in non_empty_filenames:
                tmp[i:(i + len(data[f][key]))] = data[f][key]
                i += len(data[f][key])

            fout.create_dataset(key, tmp.shape, dtype=tmp.dtype,
                                compression='gzip')[...] = tmp

        keys = groups[non_empty_filenames[0]]
        for key in keys:  # loop through all groups
            logger.info("writing group {}".format(key))
            # first loop through all keys of this group(station) to find all available entries (necessary because some
            # of the files might be empty
            list_of_keys = list(groups[non_empty_filenames[0]][key].keys())
            list_of_dtypes = {}
            list_of_shapes = {}
            for f in non_empty_filenames:
                for key2 in groups[f][key]:  # loop through all datasets of this group
                    if(key2 not in list_of_dtypes):
                        list_of_dtypes[key2] = groups[f][key][key2].dtype
                        list_of_shapes[key2] = list(groups[f][key][key2].shape)
                    groups[f][key][key2].dtype
                    if(key2 not in list_of_keys):
                        list_of_keys.append(key2)

            g = fout.create_group(key)
            for key2 in list_of_keys:  # loop through all datasets of this group
                shape = list_of_shapes[key2]
                shape[0] = sum(len(groups[f][key][key2]) for f in non_empty_filenames if key2 in groups[f][key])

                tmp = np.zeros(shape, dtype=list_of_dtypes[key2])
                i = 0
                for f in non_empty_filenames:
                    if(key2 in groups[f][key]):
                        tmp[i:(i + len(groups[f][key][key2]))] = groups[f][key][key2]
                        i += len(groups[f][key][key2])
                    else:
                        logger.info(f"data set {key2} not in file {f} of station {key}")

                g.create_dataset(key2, shape, dtype=list_of_dtypes[key2],
                                 compression='gzip')[...] = tmp
            # save group attributes
            for key2 in group_attrs[key]:
                fout[key].attrs[key2] = group_attrs[key][key2]
        # save all atrributes
        attrs['n_events'] = n_events_total
        for key in attrs:
            fout.attrs[key] = attrs[key]
    else:  # now handle the case
        logger.warning("All files are empty. Copying content of first file to output file and keeping track of total number of simulated events.")
        # all files are empty, so just copy the content of the first file (attributes and empyt data sets) to the output file
        # update n_events attribute with the total number of events
        fin = h5py.File(filenames[0], 'r')
        for key in fin.attrs:
            if(key == "n_events"):
                fout.attrs[key] = n_events_total
            elif key not in merge_history_attrs:
                fout.attrs[key] = fin.attrs[key]
        for key in fin:
            if isinstance(fin[key], h5py._hl.group.Group):
                g = fout.create_group(key)
                for key2 in fin[key]:
                    g.create_dataset(key2, fin[key][key2].shape, dtype=fin[key][key2].dtype,
                                     compression='gzip')[...] = fin[key][key2]
                for key2 in fin[key].attrs:
                    g.attrs[key2] = fin[key].attrs[key2]
            else:
                fout.create_dataset(key, fin[key].shape, dtype=fin[key].dtype,
                                    compression='gzip')[...] = fin[key]

    # the original files and their offsets; the offsets of a file merged before add up
    merged_files, egid_offsets, sid_offsets = [], [], []
    for f in filenames:
        names, egid_offsets_f, sid_offsets_f = history[f]
        merged_files += names
        egid_offsets += [x + event_group_id_offsets.get(f, 0) for x in egid_offsets_f]
        sid_offsets += [x + shower_id_offsets.get(f, 0) for x in sid_offsets_f]
    fout.attrs['merged_files'] = merged_files
    fout.attrs['event_group_id_offsets'] = np.array(egid_offsets, dtype=np.int64)
    fout.attrs['shower_id_offsets'] = np.array(sid_offsets, dtype=np.int64)
    fout.close()


if __name__ == "__main__":
    """
    Merges multiple hdf5 files into one single file. Keeps automatically track of the total number
    of simulated events across all input files (which is necessary to correctly calculate the effective volume).

    If a single path is passed as argument the script interprets it as root directory and expects the following folder structure:
    ../output/energy/*.hdf5.part????
    The output file path is automatically determined.

    If multiple paths are passed, the first one is taken as output file path. The following paths are taken as input.

    Optional log level setting to either set DEBUG, INFO, or WARNING to the readout. Example: add --loglevel DEBUG when calling script to set loglevel to DEBUG.
    """
    parser = argparse.ArgumentParser(description='Merge hdf5 files')
    parser.add_argument('files', nargs='+', help='input file or files')
    parser.add_argument('--loglevel', metavar='level', type=_parse_loglevel,
                        help='loglevel set to either DEBUG, INFO, or WARNING (or another level name or number)')
    parser.add_argument('--cores', default=1, type=int, help='number of cores to use')
    args = parser.parse_args()

    if args.loglevel is not None:
        logger.setLevel(args.loglevel)

    if len(args.files) == 1:
        root_directory = args.files[0]
        if not os.path.isdir(root_directory):
            sys.exit(f"{root_directory} is not a directory.")

        input_args = find_part_files(root_directory)

        if(args.cores == 1):
            for i in range(len(input_args)):
                merge2(input_args[i]['filenames'], input_args[i]['output_filename'])
        else:
            from multiprocessing import Pool
            logger.warning(f"running {len(input_args)} job on {args.cores} cores")
            # merge2 is picklable as a module-level function (a local one fails under the "spawn" start method)
            with Pool(args.cores) as p:
                p.starmap(merge2, [(x['filenames'], x['output_filename']) for x in input_args])

    else:
        output_filename = args.files[0]
        if(os.path.exists(output_filename)):
            logger.error('file {} already exists, skipping'.format(output_filename))
        else:
            input_files = args.files[1:]
            merge2(input_files, output_filename)
