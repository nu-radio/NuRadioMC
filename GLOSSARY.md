# GLOSSARY.md

Domain language of NuRadioMC / NuRadioReco: what the terms mean and which ones must not be mixed up.
How to work in the repository is in `AGENTS.md`; this file only defines words.

Rules for editing: one term per entry, the code name in backticks, only verified meanings. Mark what is unknown
as unknown (see [Open questions](#open-questions)) instead of guessing. Add a term when it caused a misunderstanding.

## Detector and antennas

**Antenna type** (`ant_type`)
: Key of a channel in the detector description naming its antenna model. Before v3.2 the detector appended
  `_InfAir`/`_InfFirn` to it depending on the antenna depth (`antenna_by_depth`); with #1190 (planned for v3.2)
  `ant_type` is the full antenna model name. `get_antenna_model()` returns the name that is actually loaded.

**Antenna model**
: A simulated antenna response stored as a pickle file, identified by its name (the key in
  `NuRadioReco/detector/antenna_models_hash.json`, without `.pkl`), loaded via `AntennaPatternProvider`.
  Analytic models (`analytic_*`) need no file. *Not to be confused with* the antenna type of older detector
  files, which lacked the medium suffix.

**Antenna pattern**
: The Python object (`AntennaPattern`, `AntennaPatternAnalytic`) created from an antenna model; it returns the
  vector effective length for given frequencies and directions.

**Vector effective length (VEL)**
: Complex (magnitude and phase), realized vector effective length of an antenna, split into the `theta` and `phi`
  polarization components (eTheta, ePhi). Defined in (A.13) of the NuRadioReco paper (arXiv:1903.07023).

**Infinite medium** (`_InfAir`, `_InfFirn`, `_inf_n<n>`)
: Antenna model simulated in a homogeneous medium without a boundary. "Firn" means n = 1.3-1.4; newer models
  state the refractive index (`_inf_n1.4`).

**Near-boundary model** (`_z<d>_InAir`, `_z<d>_InFirn`)
: Antenna model simulated *in* air or firn at distance `d` from the air-firn boundary, which is part of the
  simulation. *Not to be confused with* the infinite-medium models (`Inf` vs `In`). Documented with a theta range
  of [0, 90] deg only.

**Reference station / reference channel** (`reference_station`, `reference_channel`)
: Entry in a detector description that makes `Detector(...)` return a `GenericDetector`: missing properties of a
  station/channel are taken from the referenced one. Replaces the deprecated `default_station`/`default_channel`.

**Channel group id**
: Returned by `get_channel_group_id()`; equals the channel id if no group id is set.

## Signals and propagation

**Ray-tracing solution** (`solution_types`)
: Path of a signal from the source to an antenna: `1` = direct, `2` = refracted, `3` = reflected
  (`NuRadioMC/SignalProp/propagation.py`). A channel can receive one electric field per solution.

**Cosmic-ray vs. neutrino station** (`station.is_cosmic_ray()`)
: Flag set by `eventTypeIdentifier` (`'cr'` or `'nu'`). It decides whether a signal arriving from above is refracted
  into the ice (with Fresnel factors) before the antenna response is applied (`efieldToVoltageConverter`).

**Shadow zone**
: Region that no ray path from the source reaches (e.g. shallow receivers far from the source in an exponential
  firn profile). Zero ray-tracing solutions there are physical, see the pitfall in `AGENTS.md`.

## Simulation

**Trigger channel** (two unrelated meanings)
: 1. `trigger_channels` (simulation argument): ids of the channels used for the trigger decision. The other channels
  are only simulated if the event triggered.
  2. `Channel.get_trigger_channel()`: a separate `Channel` object with the same id holding the trace as seen by the
  trigger path; returns the channel itself if none is set.

**Event group vs. event** (`event_group_id`, `event_id`)
: An event group contains all showers that follow the same first interaction. Showers that interfere are combined
  into one event (`event_id` counting from 0); `group_into_events` splits a group wherever the gap between signal
  arrival times exceeds `split_event_time_diff`. See `HDF5_structure.rst`.

**Interaction type** (`ccnc`, `interaction_type`)
: `'cc'` = charged current, `'nc'` = neutral current. The generator option `interaction_type="ccnc"` picks one of
  them at random.

## Data and units

**Spectrum**
: `utilities.fft.time2freq` returns an *energy* spectral density (V/GHz; `|spec|**2` integrates to the energy),
  not a power spectral density. Matters e.g. for noise temperatures; see the docstring of `utilities/fft.py`.

**Samples vs. time**
: Lags, windows and offsets are either a number of samples (integers) or a time (in `units`). Name the unit in
  identifiers and docstrings (e.g. `lag_range_samples`) and convert at one place, the module boundary.

## Open questions

Terms used in the code or model names whose meaning is not documented. Fill in when known.

- Antenna model suffixes `_RG`, `_NoRG`, `_Backlobe`, `_BoresightToBoundary` (createLPDA / dip7cm models).
