#!/usr/bin/env python3
"""
Extend the event files written by `NuRadioReco.modules.io.eventWriterParquet` with information from the mattak
``headers.root`` of the same run(s) which is not part of the NuRadio event structure (e.g. the DiDAQ beam mask).

The rows are matched via ``(station_id, run, event_id)`` = ``(station_number, run_number, event_number)``. New columns
are named ``<prefix><name>`` (default prefix ``mattak.``). Files with several runs read the header of each run.
The existing columns, the file metadata and the row group size are kept, the electric field / sim channel files are
not touched.

Only the columns of the digitizer which recorded the data are added (DiDAQ or RADIANT / low-threshold board, plus the
columns of both). The digitizer is taken from ``stationParametersRNOG.digitizer`` and from the trigger flags of
the header (DiDAQ if ``didaq_trigger`` is set for an event, else RADIANT): a file fails if they disagree.

A file fails (and is not written) if a run directory is missing or an event is not in the header. Branches missing in
the header (older mattak versions) give null columns with a warning.

The header is read column-wise with uproot (no ROOT / PyROOT needed, no per-event loop).

Examples
--------
List the available columns::

    add_mattak_columns.py --list-columns

Add the default columns, the run directory is searched below ``--run-path`` (``station11/run1000``, ``st11_run1000``, ...)::

    add_mattak_columns.py out/station11_run001000.parquet --run-path $RNO_G_ROOT_DATA

Whole directory / glob, only some columns, write to another directory, check first with ``--dry-run``::

    add_mattak_columns.py out/ --run-path /data/rootified -c didaq_beam_mask didaq_channel_mask -o out_ext/ --dry-run
"""
import argparse
import datetime
import glob
import json
import logging
import os
import re
import sys
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from NuRadioReco.utilities.parquet_utilities import _FILE_PATTERN

logger = logging.getLogger("NuRadioReco.add_mattak_columns")

_KEYS = ["station_id", "run", "event_id"]
_HEADER_KEYS = {"station_id": "station_number", "run": "run_number", "event_id": "event_number"}

# Tree names of the header in headers.root, as tried by mattak
_HEADER_TREES = ["hdr", "header", "hd", "hds", "headers"]

# Values of `stationParametersRNOG.digitizer`, see `mattak.Dataset.Digitizer`
DIGITIZER_COLUMN = "stationParametersRNOG.digitizer"
DIGITIZERS = {1: "RADIANT", 2: "DIDAQ"}

# column name -> (header branch, arrow type, digitizer (None: both), description).
# The branch is matched by its (array-stripped) leaf name, see mattak::TriggerInfo / mattak::Header.
COLUMNS = {
    "didaq_beam_mask": ("trigger_info.didaq_info.beam_mask", pa.int64(), "DIDAQ", "DiDAQ: beams which triggered (deep phased trigger)"),
    "didaq_channel_mask": ("trigger_info.didaq_info.channel_mask", pa.int64(), "DIDAQ", "DiDAQ: channels which triggered (other RF triggers)"),
    "didaq_type": ("trigger_info.didaq_info.type", pa.int64(), "DIDAQ", "DiDAQ: raw trigger type bits"),
    "didaq_trigger": ("trigger_info.didaq_trigger", pa.bool_(), "DIDAQ", "True for a DiDAQ RF trigger"),
    "didaq_start_offsets": ("trigger_info.didaq_info.start_offsets", pa.list_(pa.uint16()), "DIDAQ", "DiDAQ: start offset per channel"),
    "lt_beam_mask": ("trigger_info.lt_info.beam_mask", pa.int64(), "RADIANT", "Low-threshold board: phased trigger beam mask"),
    "lt_channel_mask": ("trigger_info.lt_info.channel_mask", pa.int64(), "RADIANT", "Low-threshold board: channel mask"),
    "which_lt_trigger": ("trigger_info.which_lt_trigger", pa.int64(), "RADIANT", "LT trigger used (0: hi-lo, 1: phased, -1: n/a)"),
    "which_radiant_trigger": ("trigger_info.which_radiant_trigger", pa.int64(), "RADIANT", "RADIANT trigger used (-1: unknown)"),
    "radiant_channel_mask": ("trigger_info.radiant_info.channel_mask", pa.int64(), "RADIANT", "RADIANT: channel mask"),
    "trigger_number": ("trigger_number", pa.int64(), None, "Trigger number of the DAQ"),
    "sysclk": ("sysclk", pa.int64(), None, "System clock at the trigger"),
    "pps_num": ("pps_num", pa.int64(), None, "PPS counter"),
    "readout_time": ("readout_time", pa.float64(), None, "Readout time (s)"),
}
# Only those of the digitizer of the data are added
DEFAULT_COLUMNS = ["didaq_beam_mask", "didaq_channel_mask", "didaq_type", "didaq_start_offsets",
                   "lt_beam_mask", "lt_channel_mask", "readout_time"]


def find_run_dir(path, station, run):
    """
    Locate the directory with ``headers.root`` of a run.

    `path` can be the run directory itself (``headers.root`` inside) or a base directory which is searched for
    ``station<S>/run<R>``, ``st<S>_run<R>`` or ``station<S>_run<R>`` (directly, then recursively).
    """
    if os.path.isfile(os.path.join(path, "headers.root")):
        return path

    names = [f"station{station}/run{run}", f"st{station}_run{run}", f"station{station}_run{run}"]
    for name in names:
        candidate = os.path.join(path, name)
        if os.path.isfile(os.path.join(candidate, "headers.root")):
            return candidate

    for name in names:
        hits = sorted(glob.glob(os.path.join(glob.escape(path), "**", name, "headers.root"), recursive=True))
        if hits:
            if len(hits) > 1:
                logger.warning(f"Several run directories for station {station}, run {run}, using {hits[0]}")
            return os.path.dirname(hits[0])

    raise FileNotFoundError(f"No headers.root for station {station}, run {run} below {path}")


def _tree(file, names, what):
    """ The first tree of `names` in the uproot `file`. """
    for name in names:
        if name in file:
            return file[name]
    raise KeyError(f"{file.file_path}: no {what} tree (tried {names}), found {file.keys()}")


def digitizer_of_run(run_dir):
    """ The digitizer of a run: DiDAQ if ``didaq_trigger`` is set for an event of its header, else RADIANT. """
    import uproot  # imported here: only needed for reading mattak files

    with uproot.open(os.path.join(run_dir, "headers.root")) as file:
        tree = _tree(file, _HEADER_TREES, "header")
        branches = {re.sub(r"\[.*\]$", "", key.split("/")[-1]): key for key in tree.keys(recursive=True)}
        branch = branches.get("trigger_info.didaq_trigger")  # missing in data from before the DiDAQ
        is_didaq = branch is not None and bool(tree[branch].array(library="np").any())

    return "DIDAQ" if is_didaq else "RADIANT"


def get_digitizer(table, run_dirs):
    """
    The digitizer of the events in `table` from the headers (`digitizer_of_run`), checked against
    `DIGITIZER_COLUMN` if present. Raises a `ValueError` if they disagree.
    """
    from_headers = {digitizer_of_run(run_dir) for run_dir in run_dirs}
    if len(from_headers) > 1:
        raise ValueError(f"The runs {run_dirs} were recorded with different digitizers {sorted(from_headers)}.")
    digitizer = from_headers.pop()

    if DIGITIZER_COLUMN in table.column_names and table[DIGITIZER_COLUMN].null_count < table.num_rows:
        values = set(pc.unique(table[DIGITIZER_COLUMN].drop_null()).to_pylist())
        if len(values) > 1 or not values <= set(DIGITIZERS):
            raise ValueError(f"Expected one known digitizer {DIGITIZERS} in {DIGITIZER_COLUMN}, found {sorted(values)}.")
        from_table = DIGITIZERS[values.pop()]
        if from_table != digitizer:
            raise ValueError(f"{DIGITIZER_COLUMN} gives {from_table}, but didaq_trigger of the header(s) {digitizer}.")

    return digitizer


def read_header(run_dir, columns):
    """
    Read the keys and requested columns of ``headers.root``.

    Returns
    -------
    keys : dict of np.ndarray
        ``station_id``, ``run``, ``event_id`` (int64).
    data : dict of pyarrow.Array
        Requested columns; missing branches are not included (a warning is logged).
    """
    import uproot  # imported here: only needed for reading mattak files

    with uproot.open(os.path.join(run_dir, "headers.root")) as file:
        tree = _tree(file, _HEADER_TREES, "header")
        # leaf name without the array extent ("start_offsets[24]" -> "start_offsets") -> full key
        branches = {re.sub(r"\[.*\]$", "", key.split("/")[-1]): key for key in tree.keys(recursive=True)}

        keys = {name: tree[branches[branch]].array(library="np").astype(np.int64) for name, branch in _HEADER_KEYS.items()}

        data, missing = {}, []
        for name in columns:
            branch = branches.get(COLUMNS[name][0])
            if branch is None:
                missing.append(name)
                continue

            values = tree[branch].array(library="np")
            data[name] = pa.array(list(values) if values.ndim > 1 else values, type=COLUMNS[name][1])

    if missing:  # e.g. headers written by an older mattak version
        logger.warning(f"{run_dir}: not in the header (null): {missing}")

    return keys, data


def read_headers(run_dirs, columns):
    """ `read_header` of several runs, concatenated. Columns missing in some runs are null for these runs. """
    keys, data, sizes = {name: [] for name in _KEYS}, {name: [] for name in columns}, []
    for run_dir in run_dirs:
        run_keys, run_data = read_header(run_dir, columns)
        size = len(run_keys["event_id"])
        sizes.append(size)
        for name in _KEYS:
            keys[name].append(run_keys[name])
        for name in columns:
            data[name].append(run_data.get(name, pa.nulls(size, COLUMNS[name][1])))

    keys = {name: np.concatenate(values) for name, values in keys.items()}
    # Columns which are missing in all runs stay missing (filled with null later)
    missing = {name for name in columns if all(array.null_count == len(array) for array in data[name])}
    data = {name: pa.concat_arrays(arrays) for name, arrays in data.items() if name not in missing}
    return keys, data


def _row_indices(table, header_keys):
    """ Index into the header for each row of `table` (null if the event is not in the header). """
    import polars as pl

    rows = pl.from_arrow(table.select(_KEYS)).cast(pl.Int64)
    header = pl.DataFrame({**header_keys, "_header_idx": np.arange(len(header_keys["event_id"]))})
    if header.select(_KEYS).is_duplicated().any():
        logger.warning("Duplicated (station, run, event) in the header: using the first occurrence.")
        header = header.unique(subset=_KEYS, keep="first", maintain_order=True)

    return rows.join(header, on=_KEYS, how="left", maintain_order="left")["_header_idx"].to_arrow()


def extend_file(file, run_path, columns=None, prefix="mattak.", output=None, overwrite=False, dry_run=False):
    """
    Add the mattak `columns` to one event file.

    Parameters
    ----------
    file : str
        Event parquet file (``station<S>_run<R>[-<last run>][_<suffix>].parquet``).
    run_path : str
        Run directory or base directory, see `find_run_dir`.
    columns : list of str, optional
        Keys of `COLUMNS`, default: `DEFAULT_COLUMNS`. Columns of the other digitizer are skipped.
    prefix : str, default="mattak."
        Prefix of the new column names.
    output : str, optional
        Output file. Default: modify `file` in place (atomically).
    overwrite : bool, default=False
        Replace existing columns of the same name (they are skipped otherwise).
    dry_run : bool, default=False
        Only report what would be done.

    Returns
    -------
    list of str
        Names of the added columns.

    Raises
    ------
    FileNotFoundError
        If the run directory of a run in `file` is missing.
    ValueError
        If events of `file` are not in the header(s), or the digitizer is ambiguous (see `get_digitizer`).
    """
    pf = pq.ParquetFile(file)
    table = pf.read()
    metadata = pf.metadata

    if not table.num_rows:
        logger.info(f"{file}: no events, nothing to add.")
        return []

    # The stations and runs are taken from the table (a file can hold several runs)
    stations = pc.unique(table["station_id"]).to_pylist()
    runs = sorted(pc.unique(table["run"]).to_pylist())
    run_dirs = [find_run_dir(run_path, station, run) for station in stations for run in runs
                if pc.any(pc.and_(pc.equal(table["station_id"], station), pc.equal(table["run"], run))).as_py()]

    digitizer = get_digitizer(table, run_dirs)
    requested = list(columns or DEFAULT_COLUMNS)
    columns = [c for c in requested if COLUMNS[c][2] in (None, digitizer)]
    skipped = [c for c in requested if c not in columns]
    if skipped:
        logger.info(f"{file}: {digitizer} data, skipping the columns of the other digitizer {skipped}.")

    names = {column: prefix + column for column in columns}
    existing = [column for column, name in names.items() if name in table.column_names]
    if existing and not overwrite:
        logger.info(f"{file}: skipping existing columns {[names[c] for c in existing]} (use --overwrite to replace).")
        columns = [c for c in columns if c not in existing]
        if not columns:
            return []

    header_keys, data = read_headers(run_dirs, columns)
    indices = _row_indices(table, header_keys)
    if indices.null_count:  # events without header information: do not write a partially extended file
        missing = table.select(_KEYS).filter(pc.is_null(indices))
        example = {key: missing[key][0].as_py() for key in _KEYS}
        raise ValueError(f"{indices.null_count} of {table.num_rows} events are not in the header(s) of {run_dirs} "
                         f"(e.g. {example}). Is the mattak data incomplete?")
    logger.debug(f"{os.path.basename(file)}: all {table.num_rows} events found in {len(run_dirs)} header(s).")

    added = []
    for column in columns:
        name = names[column]
        new = pc.take(data[column], indices) if column in data else pa.nulls(table.num_rows, COLUMNS[column][1])
        if name in table.column_names:
            table = table.set_column(table.column_names.index(name), name, new)
        else:
            table = table.append_column(name, new)
        added.append(name)

    if dry_run:
        logger.info(f"[dry run] {file}: would add {added}")
        return added

    info = {"run_dirs": [os.path.abspath(run_dir) for run_dir in run_dirs], "digitizer": digitizer,
            "columns": added, "date": datetime.datetime.now().isoformat(timespec="seconds")}
    schema_metadata = dict(table.schema.metadata or {})
    schema_metadata[b"mattak_columns"] = json.dumps(info).encode()
    table = table.replace_schema_metadata(schema_metadata)

    # Keep the row group size and the compression of the original file
    row_group_size = max(metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)) if metadata.num_row_groups else None
    compression = metadata.row_group(0).column(0).compression.lower() if metadata.num_row_groups else "zstd"

    output = output or file
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".parquet", dir=os.path.dirname(os.path.abspath(output)))
    os.close(fd)
    try:
        pq.write_table(table, tmp, row_group_size=row_group_size, compression=compression)
        os.replace(tmp, output)  # atomic: never leaves a half written file
    except BaseException:
        os.remove(tmp)
        raise

    logger.info(f"{output}: added {added}")
    return added


def collect_event_files(inputs):
    """ Expand files / directories (recursive) / globs to event files (electric field and sim channel files are dropped). """
    files = []
    for path in inputs:
        pattern = os.path.join(glob.escape(path), "**", "*.parquet") if os.path.isdir(path) else path
        for file in sorted(glob.glob(pattern, recursive=True)):
            match = _FILE_PATTERN.match(os.path.basename(file))
            if match is None:
                logger.warning(f"Skipping {file}: the file name does not match "
                               "station<id>_run<run>[-<last run>][_<suffix>].parquet")
            elif match.group(4) is None:  # group 4: "efield" / "simchannel"
                files.append(file)

    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("Examples")[0].strip(),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help="Event parquet files, directories (searched recursively) or glob patterns.")
    parser.add_argument("-r", "--run-path", help="Run directory (with headers.root) or base directory of the mattak runs "
                        "(default: $RNO_G_ROOT_DATA or $RNO_G_DATA).")
    parser.add_argument("-c", "--columns", nargs="+", metavar="NAME",
                        help="Columns to add, or 'all' (default: see --list-columns). "
                        "Columns of the other digitizer are skipped.")
    parser.add_argument("-o", "--output-dir", help="Write the extended files here instead of modifying them in place.")
    parser.add_argument("-p", "--prefix", default="mattak.", help="Prefix of the new column names (default: %(default)s).")
    parser.add_argument("--overwrite", action="store_true", help="Replace columns which already exist.")
    parser.add_argument("-n", "--dry-run", action="store_true", help="Only report what would be done.")
    parser.add_argument("--list-columns", action="store_true", help="List the available columns and exit.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(format="%(levelname)s: %(message)s")
    # Set on the logger itself: the NuRadioReco logging setup (parent logger) ignores basicConfig
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    if args.list_columns:
        for name, (branch, dtype, digitizer, description) in COLUMNS.items():
            print(f"{name:24s}{'*' if name in DEFAULT_COLUMNS else ' '} {digitizer or 'both':8s} {str(dtype):16s} {description}")
        print("\n* default")
        return 0

    columns = list(COLUMNS) if args.columns == ["all"] else args.columns
    unknown = [c for c in columns or [] if c not in COLUMNS]
    if unknown:
        parser.error(f"Unknown columns {unknown}. See --list-columns.")

    run_path = args.run_path or os.environ.get("RNO_G_ROOT_DATA") or os.environ.get("RNO_G_DATA")
    if not args.files or not run_path:
        parser.error("Need input files and --run-path (or $RNO_G_ROOT_DATA / $RNO_G_DATA).")

    files = collect_event_files(args.files)
    if not files:
        parser.error("No event files found.")

    n_failed = 0
    for i, file in enumerate(files, 1):
        logger.info(f"[{i}/{len(files)}] {file}")
        output = os.path.join(args.output_dir, os.path.basename(file)) if args.output_dir else None
        try:
            extend_file(file, run_path, columns, args.prefix, output, args.overwrite, args.dry_run)
        except Exception as e:  # keep going with the other runs
            n_failed += 1
            logger.error(f"{file}: {type(e).__name__}: {e}")
            if args.verbose:
                logger.exception(e)

    if n_failed:
        logger.error(f"{n_failed}/{len(files)} files failed (not written).")
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
