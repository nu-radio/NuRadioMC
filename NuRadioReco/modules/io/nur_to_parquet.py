"""
Convert ``.nur`` files to the parquet format of `NuRadioReco.modules.io.eventWriter`.

Use `convert_nur`.

The events are read with `eventReaderNur` and written with `eventWriter`, with all typed station and channel
parameter enums (incl. the RNO-G ones; ``ARIANNAParameters`` are untyped and not stored). A detector description
stored in a ``.nur`` file is not converted. The file metadata gets the keys ``source_files`` (JSON list of the input
file names) and ``nur_file_version`` (JSON: input file name -> version of the ``.nur`` format).
"""

import glob
import json
import logging
import os
import time
from timeit import default_timer as timer

from NuRadioReco.framework import parameters
from NuRadioReco.modules.io.eventReaderNur import eventReaderNur
from NuRadioReco.modules.io.eventWriter import eventWriter
from NuRadioReco.utilities.logging import LOGGING_STATUS

logger = logging.getLogger("NuRadioReco.nur_to_parquet")

STATION_ENUMS = [parameters.stationParameters, parameters.stationParametersRNOG]
CHANNEL_ENUMS = [parameters.channelParameters, parameters.channelParametersRNOG]


def _expand(inputs):
    """Files matching `inputs` (files or glob patterns), in the given order, without duplicates."""
    if isinstance(inputs, (str, os.PathLike)):
        inputs = [inputs]
    files = []

    for pattern in map(os.fspath, inputs):
        matches = sorted(glob.glob(pattern)) if glob.has_magic(pattern) else [pattern]
        matches = [match for match in matches if os.path.isfile(match)]
        if not matches:
            raise FileNotFoundError(f"No file matches {pattern}")

        files += [match for match in matches if match not in files]

    return files


def _nur_version(filename):
    """Version ``major.minor`` of the ``.nur`` format, from the 12 byte file header."""
    with open(filename, "rb") as f:
        header = f.read(12)
    return f"{int.from_bytes(header[:6], 'little')}.{int.from_bytes(header[6:], 'little')}"


def _output_files(stem, t_start):
    """Parquet files written for `stem` (incl. event files and parts) since `t_start` (`time.time`)."""
    candidates = glob.glob(glob.escape(stem) + ".parquet") + glob.glob(glob.escape(stem) + "_*.parquet")
    return [f for f in candidates if os.path.getmtime(f) >= t_start - 1]  # - 1 s: coarse file system times


def _mb(n_bytes):
    return n_bytes / 1024**2


def convert_nur(
    inputs,
    output=None,
    output_dir=None,
    traces=True,
    event_file=True,
    max_file_size=1024,
    compression="snappy",
    overwrite=False,
):
    """
    Convert ``.nur`` files to parquet files.

    Parameters
    ----------
    inputs : str or list of str
        ``.nur`` files or glob patterns.
    output : str, optional
        Write all inputs into this output (``<output>.parquet``, the extension may be omitted; relative to
        `output_dir` if given). Default: one output per input, same name with ``.parquet``.
    output_dir : str, optional
        Directory of the outputs. Default: next to each input.
    traces : bool, default=True
        Store the traces (channels, sim channels, electric fields). If False, only parameters etc.
    event_file : bool, default=True
        Write separate event files (see `eventWriter.begin`).
    max_file_size : float, default=1024
        Maximum size of an output file pair in MB, larger outputs are split into parts.
    compression : str, default="snappy"
        Parquet compression codec, e.g. "zstd".
    overwrite : bool, default=False
        Overwrite existing output files (else a `FileExistsError` is raised).

    Returns
    -------
    list of str
        The parquet files written.
    """
    files = _expand(inputs)
    if output is not None:
        output = output[: -len(".parquet")] if output.endswith(".parquet") else output
        groups = {os.path.join(output_dir or "", output): files}
    else:
        groups = {}

        for f in files:
            stem = os.path.join(output_dir or os.path.dirname(f), os.path.splitext(os.path.basename(f))[0])
            if stem in groups:
                raise ValueError(
                    f"{f} and {groups[stem][0]} would be written to the same output {stem}.parquet, "
                    "use --output or different output directories."
                )

            groups[stem] = [f]

    logger.warning("A detector description stored in the .nur files is not converted.")

    written = []

    for stem, sources in groups.items():
        metadata = {
            "source_files": json.dumps([os.path.basename(f) for f in sources]),
            "nur_file_version": json.dumps({os.path.basename(f): _nur_version(f) for f in sources}),
        }
        writer = eventWriter()
        writer.begin(
            stem,
            event_file=event_file,
            max_file_size=max_file_size,
            compression=compression,
            station_enums=STATION_ENUMS,
            channel_enums=CHANNEL_ENUMS,
            overwrite=overwrite,
            comment="converted from .nur",
            metadata=metadata,
        )
        t_start, size_in, n_total = time.time(), 0, 0

        for f in sources:
            t0 = timer()
            reader = eventReaderNur()
            reader.begin(f)
            n_events = 0

            for evt in reader.run():
                writer.run(evt, mode=None if traces else False)
                n_events += 1

            reader.end()
            size_in += os.path.getsize(f)
            n_total += n_events
            logger.log(
                LOGGING_STATUS, f"{f}: {n_events} events ({_mb(os.path.getsize(f)):.1f} MB) " f"in {timer() - t0:.1f} s"
            )

        writer.end()

        outputs = sorted(_output_files(stem, t_start))
        size_out = sum(os.path.getsize(f) for f in outputs)
        logger.log(
            LOGGING_STATUS,
            f"{stem}.parquet: {n_total} events, {_mb(size_in):.1f} MB .nur -> "
            f"{_mb(size_out):.1f} MB parquet ({len(outputs)} file(s)) "
            f"in {time.time() - t_start:.1f} s",
        )
        written += outputs

    return written
