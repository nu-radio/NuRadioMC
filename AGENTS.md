# AGENTS.md

Guidance for AI coding agents (and new developers) working in this repository.
Full documentation: https://nu-radio.github.io/NuRadioMC/main.html (sources: `documentation/source`, reST + Sphinx + numpydoc).

## What this repository is

One repository, two interdependent Python packages:

- **NuRadioReco** – event data framework, detector description, processing modules, I/O. Used for both simulated and measured data (RNO-G, ARIANNA, ARA, LOFAR, ...).
- **NuRadioMC** – Monte Carlo simulation of in-ice radio neutrino (and emitter) detection: event generation, signal generation, ray tracing, end-to-end detector simulation. Builds on NuRadioReco.

NuRadioMC imports NuRadioReco, never the other way round.

## Setup and tests

```bash
pip install -e .[dev]          # optional extras: proposal, galacticnoise, muon-flux, cr_interpolator, ALL
pre-commit install             # blocks commits to master/develop, files > 500 kB, notebooks with outputs; runs flake8 (E9, F63, F7, F82)
./run_all_tests.sh             # shell-driven integration tests under NuRadioMC/test and NuRadioReco/test
python -m pytest NuRadioReco/detector/test/test_antennapattern.py -q   # single pytest file
python documentation/make_docs.py                                     # build docs into documentation/build/html
```

- CI: `.github/workflows/run_tests.yaml` (flake8 + test jobs: single events, signal processing, Veff/Aeff with fixed seeds, RNO-G, examples, module structure). Coverage is combined across jobs and commented on PRs.
- Many tests are scripts compared against reference output (fixed seeds). A change in physics output breaks them intentionally; update references only with a justification.
- Antenna models (up to ~700 MB each) are downloaded on first use into `NuRadioReco/detector/AntennaModels/` (sha1-checked, see `antenna_models_hash.json`, `utilities/dataservers.py`). They are not in git.

## Code map

| Path | Content |
|---|---|
| `NuRadioReco/framework/` | Event data structure (see below) |
| `NuRadioReco/modules/` | Processing modules. Generic ones at top level (`channelBandPassFilter`, `efieldToVoltageConverter`, `channelGenericNoiseAdder`, ...); experiment-specific in `RNO_G/`, `ARA/`, `ARIANNA/`, `LOFAR/`; triggers in `trigger/`; readers/writers in `io/`; `base/module.py` (`register_run`) |
| `NuRadioReco/detector/` | Detector description: `detector.py` (JSON/SQL), `generic_detector.py` (simulation), `RNO_G/rnog_detector.py` (MongoDB), `antennapattern.py`, `response.py` (signal-chain responses), `amp.py`/`filterresponse.py`, JSON detector files per experiment |
| `NuRadioReco/utilities/` | `units`, `fft`, `signal_processing`, `trace_utilities`, `geometryUtilities`, `logging`, `noise`, `dataservers`, ... |
| `NuRadioReco/eventbrowser/` | Dash web app to inspect `.nur` files |
| `NuRadioMC/EvtGen/` | Neutrino/secondary event generation (`generator.py`, PROPOSAL interface) → input HDF5 |
| `NuRadioMC/SignalGen/` | Askaryan emission (`askaryan.py`, `parametrizations.py`, `ARZ/`), emitters |
| `NuRadioMC/SignalProp/` | Propagation: `analyticraytracing.py` (C++ backend by default, numba/python fallback), `radioproparaytracing.py` |
| `NuRadioMC/utilities/` | Ice models (`medium.py`), attenuation, cross sections, fluxes, `Veff.py`, HDF5 merge/split |
| `NuRadioMC/simulation/` | `simulation.py` (end-to-end driver), `output_writer_hdf5.py`, `config_default.yaml` |
| `*/examples/`, `*/test/` | Examples (also run in CI) and tests |

## Data structure (`NuRadioReco/framework`)

Hierarchical; children are reached via `get_*` / `iter_*` on the parent.

```
Event (run number, event id, event_time)
├── Station                      reconstructed / measured, per station id
│   ├── Channel(s)               voltage traces; optional separate trigger channel (get_trigger_channel) (Trace class)
│   ├── ElectricField(s)         reconstructed E-fields (Trace class)
│   ├── Trigger(s)               type, threshold, has_triggered, trigger time
│   └── SimStation               MC truth (absent in measured data)
│       ├── SimChannel(s)        per channel × shower × ray-tracing solution (Trace class)
│       └── ElectricField(s)     simulated E-fields at the antenna (Trace class)
├── Shower(s) / SimShower(s)     RadioShower: reconstructed; SimShower: MC truth
├── SimEmitter(s)                for emitter (pulser) simulations
├── Particle(s)                  primary + interaction products
└── HybridInformation            showers from complementary detectors
```

- **Traces** (`base_trace.BaseTrace`, parent of `Channel`/`ElectricField`): Store waveforms in either the time or frequency domain; `get_frequency_spectrum()` uses `utilities.fft`. `get_times()` / `trace_start_time` are relative to the station time. `+` sums traces (resampling/padding as needed).
- **ElectricField**: has a position and associated channel ids; several per channel (one per ray solution). Components are on-sky (r, θ, φ) polarisations.
- **Parameter storage**: `obj[stnp.zenith] = 45 * units.deg`, `get_parameter_error(...)`. Enums live in `framework/parameters.py` (`stationParameters`, `channelParameters`, `electricFieldParameters`, `showerParameters`, `particleParameters`, `eventParameters`, ...). Add new enum members **at the bottom**, never reuse values, document with a `#:` comment.
- **Files**: `.nur` is the full event format (every class implements `serialize`/`deserialize`; read with `modules/io/eventReader` or `NuRadioRecoio`, write with `eventWriter`). NuRadioMC additionally writes a summary **HDF5** file (see `documentation/.../HDF5_structure.rst`).
- **Times**: absolute times are `astropy.time.Time` (event/station time); everything inside a station is a float relative to it. See `documentation/source/NuRadioReco/pages/times.rst`.

Docs: `NuRadioReco/pages/event_structure.rst`, `detector/detector.rst`, `detector/rnog.rst`.

## Modules

```python
class myModule:
    def begin(self, ...): ...                       # settings that don't change per event
    @register_run()                                 # from NuRadioReco.modules.base.module
    def run(self, evt, station, det, ...): ...      # per event/station
    def end(self): ...
```

- Signature order matters: `run(evt, station, det, ...)`; `register_run` records module name + kwargs on the event.
- Module registration is stored **per `Event` instance** (`evt.iter_modules(station_id)`). A module run against a temporary/dummy event is invisible on the real one; code that inspects which filters were applied (e.g. to color noise) relies on this.
- Logging: `logger = logging.getLogger('NuRadioReco.<module>')` (or `NuRadioMC.<...>`). Do not set a level inside the module; a custom `STATUS` level (25) exists. See `nur_modules.rst`.
- New modules must pass `NuRadioReco/test/check_modules.py` (CI "module structure" job).

## NuRadioMC simulation flow

Users subclass `NuRadioMC.simulation.simulation.simulation` and implement `_detector_simulation_filter_amp(evt, station, det)` and `_detector_simulation_trigger(evt, station, det)`; the YAML config is merged over `config_default.yaml`.

Per event group (`run()`), per station:

- Read input HDF5 → ray tracing per shower × channel → `calculate_sim_efield`, for the trigger channels only.
- **Event splitting**: `group_into_events` sorts the sim channels by `trace_start_time` (signal arrival) and splits the group into separate events wherever the gap exceeds `split_event_time_diff` (config).
- Per event: detector response (`apply_det_response`: antenna via `efieldToVoltageConverter`, noise, filter/amp) → trigger → `channelReadoutWindowCutter` if triggered.
- **Non-trigger channels** (all channels not in `trigger_channels`) are simulated only if at least one event of the group triggered: efields → `apply_det_response_sim` (per efield, noiseless) → sim channels summed into the readout window of each triggered event → noise added afterwards (`add_filtered_noise_to_channels`).
- Write HDF5 (+ optional `.nur`) for triggered events. `trigger_channels=None` means all channels are trigger channels.

Manuals: `documentation/source/NuRadioMC/pages/Manuals/` (config, event generation, signal generation/propagation, ice models, Veff tutorial, clusters).

## Conventions

- **Units**: always use `NuRadioReco.utilities.units`. Base units: m, ns, GHz, eV, V, rad. Multiply on input (`10 * units.m`), divide on output (`x / units.MHz`).
- **FFT**: use `NuRadioReco.utilities.fft` (`freqs`, `time2freq`, `freq2time`), never bare `numpy.fft`. Convention: `rfft / sampling_rate * sqrt(2)` → spectra in V/GHz, energy-conserving (`sum(trace**2) * dt ≈ sum(|spec|**2) * df`, up to DC/Nyquist bins).
- **Coordinates**: x = East, y = North, z = up; origin at the surface. Zenith 0° = up, 180° = down; azimuth counted from East towards North. Directions point to where the signal *came from* (exception: `launch_vector`). See `Introduction/pages/conventions.rst`.
- **Particles**: PDG codes (12/14/16 = νe/νμ/ντ, negative = anti).
- **Style**: PEP-8, space after commas, numpydoc docstrings (blank line before lists). Don't restyle untouched legacy code. In docstrings, the default Sphinx role resolves single backticks as Python references; use double backticks for anything that isn't an importable object (module lists, file paths, package names), or the docs build fails.
- **API changes**: deprecate instead of removing public API (`deprecated` decorator in `NuRadioReco/utilities/logging.py`, or a property that warns).
- **Dependencies**: do not add new ones without discussing with maintainers.

## Contributing

- Branch from and open PRs against **`develop`** (`main` = releases, published to PyPI).
- Fill in `pull_request_template.md`; add a line to `changelog.txt` (current `-dev` version, "new features" or "bugfixes").
- Merge only after approval and a 24 h waiting period (see `CONTRIBUTING.md`, `Introduction/pages/contributing.rst`).

## Pitfalls

- **Ray tracing – 0 solutions can be physical**: shallow receivers far from the source lie in the shadow zone of the exponential firn profile. Cross-check the C++ backend against python (`use_cpp=False`) before calling it a bug.
- **Detector responses**: evaluate `Response` objects (e.g. `rnog_detector.Detector.get_signal_chain_response`) on the rfft grid from `fft.freqs(n, sampling_rate)`; arbitrary `linspace` grids break the time-domain windowing (or pass `window_response=False`).
- **RNO-G database**: `rnog_detector.Detector(database_connection='RNOG_public')` needs network access to the MongoDB; an SSL/`ServerSelectionTimeoutError` is a network/firewall issue. The legacy `RNO_season_*.json` files are for `detector.Detector`, not `rnog_detector.Detector`.
- **Always `det.update(astropy.time.Time(...))`** before querying a time-dependent detector.
- **FFT filtering is circular**: multiplying spectra wraps pulses around short traces; pad or use `channelStopFilter`.
- **Filter-then-sum ≠ sum-then-filter** on short windows; changes to the order of detector-response steps can alter trigger rates / Veff.
- `AntennaPattern.get_antenna_response_vectorized` returns 0-d values for a single frequency.

## Working as an agent

- Keep scratch scripts and outputs outside the repo; never commit large files, antenna models, or notebook outputs.
- Keep diffs scoped to the task.
- For changes that can affect physics output, show before/after numbers (e.g. a Veff or trigger-rate test with fixed seed, or bit-identical HDF5 output for pure refactors).
- Run the relevant tests from `run_all_tests.sh` / CI and `flake8` on touched files before proposing a PR.
