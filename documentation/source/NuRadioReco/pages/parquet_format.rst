Event Files (parquet)
=====================

NuRadioReco stores complete events (parameters, traces, simulated quantities, triggers, module settings) in
`parquet <https://parquet.apache.org/>`_ files: typed, columnar and compressed tables. They replace the pickle based
``.nur`` files. This page covers how to use them, the file layout, why parquet, and the rules for developers.
Details of the options are in the API documentation of
:mod:`eventWriter <NuRadioReco.modules.io.eventWriter>`, :mod:`eventReader <NuRadioReco.modules.io.eventReader>` and
:mod:`parquet_utilities <NuRadioReco.utilities.parquet_utilities>`.

Getting started
---------------

Write and read events
_____________________

.. code-block:: Python

  from NuRadioReco.modules.io.eventReader import eventReader
  from NuRadioReco.modules.io.eventWriter import eventWriter

  reader = eventReader()
  reader.begin("input.parquet")      # files, directories (searched recursively) or glob patterns
  writer = eventWriter()
  writer.begin("processed")          # writes processed.parquet + processed_event.parquet
  for evt in reader.run():
      # ... run your modules on evt ...
      writer.run(evt)                # all stations of the event; mode=False stores no traces
  writer.end()
  reader.end()

The reader returns the events in the order they were written. ``reader.get_event_i(i)`` gives random access to
single events. The files store no detector description: load the detector separately.

Existing files are only replaced with ``begin(..., overwrite=True)``. The writer writes to hidden temporary files in
the target directory (``.<name>.<random>.tmp``) and moves them into place when it is closed (``end`` or ``close``).
Therefore a script may rewrite the file it reads from (same file name, ``overwrite=True``), and a script that stops
with an exception leaves existing files untouched. A replaced file keeps its permissions, a new file gets the default
permissions of new files (umask).

Read less
_________

Reading the traces dominates the time. Without them, and only for some stations or runs:

.. code-block:: Python

  reader = eventReader()
  reader.begin("processed.parquet", stations=[101], runs=[7, 8], read_traces=False)

Events without a selected station are skipped. Files whose row group statistics exclude the selection are not read.

Column analyses with polars
___________________________

Every parameter is a column, so quantities of many events are read without building ``Event`` objects (and without
NuRadio, with any parquet library).
:class:`AnalysisReader <NuRadioReco.utilities.analysis_reader.AnalysisReader>` finds the files, selects stations
and runs, and returns a `polars <https://pola.rs>`_ ``DataFrame`` with one row per event and station; the event
level columns have the prefix ``event.``. It needs polars >= 2 (``pip install -e .[analysis]``).

.. code-block:: Python

  import polars as pl
  from NuRadioReco.utilities.analysis_reader import AnalysisReader

  df = AnalysisReader("processed.parquet", stations=[101]).read()
  snr = (
      df.select("run", "event_id", "channel_ids", "channelParameters.SNR")
      .explode("channel_ids", "channelParameters.SNR")  # one row per channel
      .with_columns(pl.col("channelParameters.SNR").struct.field("definitions").map.get("peak_amplitude").alias("snr"))
  )
  energies = df["event.generatorAttributes.Emin"].to_numpy()

Traces are only read on request (``AnalysisReader(..., traces=True)``). See also
``NuRadioReco/examples/parquet_io_example.py``.

Command line
____________

``nuradio-analyse`` (installed with NuRadioMC; also ``python -m NuRadioReco.utilities.analysis_reader``) inspects
and converts files (``nuradio-analyse <command> --help``):

.. code-block:: bash

  nuradio-analyse columns processed.parquet                         # columns with data
  nuradio-analyse columns processed.parquet --class stationParameters --stats
  nuradio-analyse columns processed.parquet --triggers              # events per (first) trigger
  nuradio-analyse hist SNR processed.parquet --key peak_amplitude --channels 0 1 -o snr.png
  nuradio-analyse convert old/*.nur --output-dir new/               # .nur -> parquet

``convert`` writes one output per input (``--output NAME`` merges them) and does not convert a detector stored in
the ``.nur`` file. From python: :func:`convert_nur <NuRadioReco.modules.io.nur_to_parquet.convert_nur>`.

``.nur`` files
______________

``eventReader`` still reads ``.nur`` files (deprecated, with a ``FutureWarning``, via
:mod:`eventReaderNur <NuRadioReco.modules.io.eventReaderNur>`); ``.nur`` and parquet files can not be read together.
They can no longer be written. These tools still read only ``.nur`` files (through
:class:`NuRadioRecoio <NuRadioReco.modules.io.NuRadioRecoio.NuRadioRecoio>`) and have not been migrated yet:
:mod:`channelMeasuredNoiseAdder <NuRadioReco.modules.measured_noise.channelMeasuredNoiseAdder>`, the detector browser
(``NuRadioReco/detector/detector_browser``) and the example
``NuRadioReco/examples/neutrinoDirectionReconstructor/runNeutrinoReconstruction.py``.

Event browser
_____________

The :doc:`event browser </NuRadioReco/pages/event_display>` opens parquet files:
``python NuRadioReco/eventbrowser/index.py <folder or file>``. Select the station file; an ``_event.parquet``
file is opened via its station file.


File layout
-----------

Files
_____

.. list-table::
  :header-rows: 1
  :widths: 30 70

  * - File
    - Content
  * - ``<name>.parquet`` (station file)
    - One row per (event, station): station, sim station, channels, trigger channels, sim channels, electric fields,
      sim electric fields, triggers. Keys ``run``, ``event_id``, ``station_id``.
  * - ``<name>_event.parquet`` (event file)
    - One row per event: event parameters, generator attributes, event time, showers, sim showers, hybrid showers,
      particles, sim emitters. Keys ``run``, ``event_id``.

- **Single file layout** (``eventWriter.begin(..., event_file=False)``): no event file; its columns are added to every
  station row with the prefix ``event.`` (e.g. ``event.generatorAttributes.Emin``, ``event.modules``). Events
  without stations are not written; count events with ``unique(["run", "event_id"])``.
- **Splitting**: when a file pair reaches ``max_file_size`` (default 1024 MB, estimated) or ``events_per_file``, the
  output continues in ``<name>_part02.parquet`` + ``<name>_part02_event.parquet``, ``_part03``, ... A station file and
  its event file hold the same events; a new pair is only started between events.
- **Row groups**: at most ``row_group_size`` rows (default 100), closed earlier once the buffered traces exceed
  ``max_row_group_bytes`` (default 256 MiB; a row group must stay below 2 GiB). Readers decode row groups in parallel
  and select stations and runs by their statistics.
- Stations and runs may be mixed freely in one file; the writer imposes no file per station or run.

Columns
_______

Objects of one kind are stored as **list columns with one entry per object**, aligned with the id column of their
kind (struct of lists). For example, ``channelParameters.SNR[i]`` belongs to channel ``channel_ids[i]``.

.. list-table::
  :header-rows: 1
  :widths: 18 30 52

  * - Kind
    - Id columns
    - Parameter columns / attributes
  * - station
    - ``station_id`` (one per row)
    - ``stationParameters.<name>``, ``station.<attribute>``
  * - sim station
    - ``sim_station.station_id``
    - ``sim_station.stationParameters.<name>``, ``sim_station.<attribute>``
  * - channel
    - ``channel_ids`` (sorted)
    - ``channelParameters.<name>``, ``channel.<attribute>``
  * - trigger channel
    - ``trigger_channel.id`` (aligned with ``channel_ids``, null: none)
    - ``trigger_channel.channelParameters.<name>``, ...
  * - sim channel
    - ``sim_channel.channel_id``, ``.shower_id``, ``.ray_tracing_id``
    - ``sim_channel.channelParameters.<name>``, ...
  * - electric field
    - ``efield.channel_ids`` (list per field), ``.shower_id``, ``.ray_tracing_id``
    - ``efield.electricFieldParameters.<name>``, ``efield.position``, ...
  * - sim electric field
    - as ``efield``, prefix ``sim_efield.``
    - ``sim_efield.electricFieldParameters.<name>``, ...
  * - trigger
    - ``trigger.name``, ``trigger.type``
    - ``trigger.triggerParameters.<name>``
  * - event (event file)
    - ``run``, ``event_id``
    - ``eventParameters.<name>``, ``generatorAttributes.<name>``, ``event_time_*``, ``modules``
  * - shower, sim shower
    - ``shower.id``, ``shower.station_ids`` (``sim_shower.``)
    - ``shower.showerParameters.<name>``, ...
  * - hybrid shower
    - ``hybrid_shower.id``, ``hybrid_shower.name``
    - ``hybrid_shower.showerParameters.<name>``
  * - particle
    - ``particle.id``
    - ``particle.particleParameters.<name>``
  * - sim emitter
    - ``sim_emitter.id``, ``sim_emitter.station_ids``
    - ``sim_emitter.emitterParameters.<name>``

Naming: parameter columns are ``[<kind>.]<enum class>.<member>``, other attributes ``<kind>.<attribute>``. Every kind
has the columns ``<kind>.parameter_types`` (names of its parameter enums), ``<kind>.covariances`` and
``<kind>.none_parameters`` (see below). The attributes of the event itself have no prefix in the event file
(``modules``, ``parameter_types``, ``covariances``, ``none_parameters``); like all event columns, they get the prefix
``event.`` in the station rows (single file layout, ``AnalysisReader``), e.g. ``event.modules``. Further station columns: ``station_time_jd1/_jd2/_scale`` (and
``sim_station.station_time_*``), ``trigger`` (name of the earliest fired trigger, null if none fired),
``station.modules``, ``station.particle_type``, ``station.triggered``, ``station.reference_reconstruction``,
``sim_station.magnetic_field_vector``, ``sim_station.simulation_weight``, ``<kind>.group_id`` of channels.

Which station and channel enums get columns is set by ``station_enums`` / ``channel_enums`` of ``eventWriter.begin``
(default: ``stationParameters`` + ``stationParametersRNOG``, ``channelParameters`` + ``channelParametersRNOG``);
the reader takes them from the file metadata.

Traces
______

Traces are stored in the time domain as float64, as nested lists: ``<kind>.trace`` is ``list<list<double>>`` for
channels (one list of samples per channel) and ``list<list<list<double>>>`` for electric fields (three components),
with ``<kind>.sampling_rate`` and ``<kind>.trace_start_time`` per object (relative to the station time). Trace
lengths may differ between objects. An object without stored trace has a null entry. Trace columns are neither
dictionary encoded nor have statistics. Read them with pyarrow (``pyarrow.parquet.read_table``); polars' native
reader is much slower for nested float lists (``AnalysisReader`` does this for you).

Parameter types
_______________

The column type follows the ``dtype`` declared in the enum member (``name = id, dtype``, see
:class:`TypedEnum <NuRadioReco.framework.parameters.TypedEnum>`; mapping in
:func:`to_arrow_type <NuRadioReco.utilities.parquet_utilities.to_arrow_type>`). The reader converts the values back
to the declared type.

.. list-table::
  :header-rows: 1
  :widths: 35 30 35

  * - Declared type
    - Arrow type
    - Read back as
  * - ``bool``, ``int``, ``float``, ``str``
    - ``bool``, ``int64``, ``double``, ``string``
    - python scalar
  * - ``np.ndarray``
    - ``list<double>`` (1-D only)
    - float64 array
  * - ``complex``
    - ``struct<real, imag>``
    - ``complex``
  * - ``list[T]``, ``tuple[T, ...]``
    - ``list<T>``
    - ``list`` / ``tuple``
  * - ``dict[K, V]``
    - ``map<K, V>``
    - ``dict``
  * - ``TypedDict`` subclass
    - ``struct`` (one field per key)
    - ``dict`` (keys that were missing are left out)
  * - ``OneOf(name=type, ...)``
    - ``struct`` (one field per alternative; only the field of the stored alternative is set)
    - the value of the set field
  * - ``astropy.time.Time``
    - no parameter column; the event time, station time and sim station time are stored as ``<name>_jd1``,
      ``<name>_jd2`` (float64, about 10 ps precision; as in ``.nur``) and ``<name>_scale``
    - ``astropy.time.Time``
  * - ``Any``, bare ``dict`` / ``list``, other classes
    - no column (warning when the schema is built)
    - not stored

- **Null means "not set".** Parameters explicitly set to None are listed by name in ``<kind>.none_parameters``
  (e.g. ``["particleParameters.parent_id"]``) and restored as None.
- **Covariances** (incl. errors, stored as variances by ``set_parameter_error``): ``<kind>.covariances`` is a list of
  ``struct<a, b, value, values>`` with the two parameter names, the scalar value, or the flattened values for
  array valued parameters.
- **Triggers**: name and type (selects the class) plus all settings and results as ``triggerParameters`` columns.
  Triggers of sim stations are not stored.
- **Module registry** and other free-form data are tagged JSON strings (``NuRadioReco.utilities.io_utilities._to_json``):
  ``modules`` (event file) and ``station.modules`` hold the names and keyword arguments of the modules run on the event
  (not the module instances). Objects JSON can not express are stored as their ``repr`` (with a warning).

File metadata
_____________

Key-value pairs in the parquet footer (``pyarrow.parquet.read_schema(file).metadata``):

.. list-table::
  :header-rows: 1
  :widths: 25 75

  * - Key
    - Value
  * - ``kind``
    - ``station`` or ``event``; files without it are not read
  * - ``layout``
    - ``dual`` (station + event file) or ``single`` (``event_file=False``)
  * - ``parameter_enums``
    - JSON: names of the station and channel enums with columns
  * - ``sim_config``
    - JSON: ``eventParameters.sim_config`` of the first event of the file (simulations)
  * - ``nuradio_version``, ``nuradio_commit``
    - version and git commit of the writer
  * - ``comment``
    - free text (``eventWriter.begin(comment=...)``)
  * - further keys
    - ``eventWriter.begin(metadata=...)``; ``nuradio-analyse convert`` adds ``source_files`` and ``nur_file_version``

Not stored: the detector description, the hybrid detector object of hybrid showers, triggers of sim stations.


Why parquet
-----------

- Typed columns that any parquet library (pyarrow, polars, pandas, DuckDB, ...) reads without NuRadio.
- No pickle: reading a file does not run code, and files do not depend on the class layout at the time of writing.
- Column selection: an analysis reads one quantity of many events without decoding the traces; row group
  statistics skip files and row groups outside a station or run selection.
- Compression and size: the files are compressed per column (snappy by default, ``compression="zstd"`` for smaller
  files).
- Row groups are decoded in parallel when reading.
- Written with pyarrow: it appends row groups incrementally and writes custom file metadata with stable API
  (polars encodes faster in parallel, but incremental writing needs unstable API).
- No detector is stored: it is large and duplicated in every file; keep track of the detector (JSON file or
  database time) yourself.


Developer rules
---------------

- **Every parameter has a concrete type.** A new member of a parameter enum (also ``triggerParameters``) is
  declared as ``name = id, dtype`` with a type from the table above, never ``Any`` or a bare ``dict`` / ``list``
  (it would get no column and not be stored). Structured dicts with fixed keys get a ``TypedDict`` (see
  ``RaytracingSolution`` in :mod:`NuRadioReco.framework.parameters`), values that are one of several types a
  ``OneOf`` (e.g. a threshold for all channels or per channel id:
  ``OneOf(value=float, per_channel=dict[int, float])``). Add members at the bottom, never reuse ids,
  and never change the type of an existing member: add a new member instead.
- **Values must match the declared type.** The writer converts what pyarrow accepts for the column type, and the
  reader returns the declared type, not the original one: an ``int`` in a ``float`` column comes back as float, a
  numpy array in a ``list`` column as list, a whole-number float (``14.0``) in an ``int`` column as int.
  Values that can not be converted (e.g. a 1-element array for a ``float``, a key that is no field of the
  ``TypedDict``, for an ``int`` a float with a fractional part, NaN, inf or a number outside the int64 range) are
  written as null, with a warning once per column. Such an invalid ``int`` as element of a list, dict or struct
  raises an ``ElementTypeError`` instead (null would lose the whole value).
  Ids (run number, event, station, channel and object ids, also in id lists) must be ints in the int64 range, and
  run number, event, station and channel ids must not be None: anything else raises a ``TypeError``. An event that
  raises is not written at all (none of its rows); events written before are kept.
- **No pickling.** New framework classes and attributes need columns: extend the groups in
  :mod:`eventWriter <NuRadioReco.modules.io.eventWriter>` and restore them in
  :mod:`eventReader <NuRadioReco.modules.io.eventReader>` through public API of the framework classes (constructors,
  getters, setters), not through ``serialize`` / ``deserialize``.
- **Extend the round-trip test.** ``NuRadioReco/test/io/test_parquet_roundtrip.py`` writes events, reads them back
  and compares them object by object. Add new objects or attributes to ``_synthetic_events`` and to the comparison
  (``_Comparison``). Differences are failures unless listed in ``KNOWN_DIFFERENCES`` (input file, regular
  expression, cause); prefer fixing the type mismatch over adding an entry. Run it and ``NuRadioReco/test/io/test_event_writer_parquet.py`` after
  changes to the framework or the I/O.
- **The** ``.nur`` **reader is deprecated.** :mod:`eventReaderNur <NuRadioReco.modules.io.eventReaderNur>` is kept
  only to read and convert old files, for one or two minor releases. Do not add features to it.
