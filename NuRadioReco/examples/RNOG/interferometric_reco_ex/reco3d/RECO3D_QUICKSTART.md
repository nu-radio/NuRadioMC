# 3D interferometric reconstruction: quickstart

This guide explains how to reproduce the 3D interferometric reconstruction results on the GZK neutrino simulation and the simulated pulser calibration dataset.

## Prerequisites

1. **NuRadioMC/NuRadioReco** installed (this repo, `reco3d_release` branch)
2. **Python 3.11+** with numpy, scipy, h5py, pyyaml, numba (optional but recommended for speed)
3. **Multiray travel time tables** for your station. Two table schemes are supported:
   - **Ray-type tables** (default): 44 NPZ files per station (11 channels x 4 tables: direct, refracted, reflected, combined). File pattern: `st{ID}_ch{CH}_rz_table_{ray_type}.npz`.
   - **Solution-ordered tables** (recommended, faster): 22 NPZ files per station (11 channels x 2 tables: solution_0 = fastest arrival, solution_1 = slowest). File pattern: `st{ID}_ch{CH}_rz_table_solution_{0,1}.npz`. Set `table_scheme: "solution_ordered"` in the config to use these. See [Solution-ordered tables](#solution-ordered-tables) below.
   
   Tables are not included in the repo. On the Chicago cluster, pre-generated tables (both ray-type and solution-ordered) for stations 11, 12, 13, 21, 22, 23, and 24 are at `/data/reconstruction/validation_sets/test_tables/multiray_tables/`. For other stations, generate them using the scripts in `../tables/` (see below).
4. **RNO-G MongoDB access** (detector description is loaded from `radio.zeuthen.desy.de:27017` by default) or a local detector export file (set `detector_file` in the config)
5. **Simulation datasets** (see below)

## Datasets

### GZK neutrino simulation

200 paired NUR/HDF5 files. 27,667 triggered events at station 23 with 300K thermal noise, energy range 10^18 to 10^20 eV with GZK-weighted spectrum.

Filename pattern:
```
nu_e_ccnc_1e18_1e20eV_GZK-2_IceCube-nu-2022_{NNNNNN}.nur
nu_e_ccnc_1e18_1e20eV_GZK-2_IceCube-nu-2022_{NNNNNN}.hdf5
```

On the Chicago cluster, the dataset is at `/data/reconstruction/validation_sets/sim_neutrinos/sim_output_gzk/`.

### Simulated pulser calibration

6,840 NUR files covering a 3D grid of emitter positions around station 23.

| Parameter | Range | Steps |
|-----------|-------|-------|
| Distance from PA center | 10-200 m | 20 values |
| Zenith angle | 20-160 deg | 15 values |
| Azimuth angle | 0-350 deg | 36 values |
| Events per grid point | 3 | |
| Total triggered events | 18,879 | |

Filename pattern: `output_r{R}_zen{ZEN}_az{AZ}.nur`

On the Chicago cluster, the dataset is at `/data/reconstruction/validation_sets/sim_cal_pulsers/test_set/`.

## Setup

1. Edit `time_delay_tables` in the config file to point to the parent directory of `station23/`:

```yaml
time_delay_tables: "/path/to/multiray_tables"
```

The code appends `station{ID}/` internally, so it will look for files at `<time_delay_tables>/station23/st23_ch{N}_rz_table_{ray_type}.npz`.

2. Verify the table files are present:

```bash
ls /path/to/multiray_tables/station23/st23_ch0_rz_table_direct.npz
```

## Using a different station

The shipped configs are for station 23. To run on a different station, copy a config and change `station_id` to your station number. Make sure travel time tables for your station exist at the path specified by `time_delay_tables`. The code looks for files at `<time_delay_tables>/station{ID}/st{ID}_ch{N}_rz_table_{ray_type}.npz`. All other config parameters (channels, grid limits, preprocessing) are the same across stations.

## Running reconstruction

### Single file (interactive)

```bash
cd reco3d/

# Neutrino, hw mode (no antenna dedispersion, ~2 s/event)
python interferometric_reco_3d_example.py \
    --config configs/reco3d_neutrino_gzk.yaml \
    --mode hw \
    -i /path/to/nu_e_ccnc_1e18_1e20eV_GZK-2_IceCube-nu-2022_000000.nur \
    -o results/test_neutrino_hw.h5

# Pulser, rxtx mode (Rx + Tx antenna dedispersion, ~10 s/event)
python interferometric_reco_3d_example.py \
    --config configs/reco3d_pulser_sim.yaml \
    --mode rxtx \
    -i /path/to/output_r50.0_zen90.0_az0.0.nur \
    -o results/test_pulser_rxtx.h5

# Pulser, hw mode (no antenna dedispersion, ~5 s/event)
python interferometric_reco_3d_example.py \
    --config configs/reco3d_pulser_sim.yaml \
    --mode hw \
    -i /path/to/output_r50.0_zen90.0_az0.0.nur \
    -o results/test_pulser_hw.h5
```

### Batch (SLURM)

```bash
# Neutrino GZK, hw mode, 100 chunks
bash submit_reco3d_example.sh \
    --config configs/reco3d_neutrino_gzk.yaml \
    --data-dir /path/to/gzk_nur_files/ \
    --output-dir /path/to/output/neutrino_hw/ \
    --account your_account \
    --mode hw --n-chunks 100

# Pulser sim, rxtx mode, 200 chunks
bash submit_reco3d_example.sh \
    --config configs/reco3d_pulser_sim.yaml \
    --data-dir /path/to/pulser_scan_data/ \
    --output-dir /path/to/output/pulser_rxtx/ \
    --account your_account \
    --mode rxtx --n-chunks 200
```

The script splits NUR files across chunks, submits parallel SLURM jobs, and queues a merge job (inline HDF5 concatenation) that runs after all chunks complete. The final output is `merged_reco_results.h5`.

### Evaluating results

Use `evaluate_reco_results.py` to compute angular separations against simulation truth and compare to the validated results:

```bash
# Neutrino GZK (truth from paired HDF5 files)
python evaluate_reco_results.py \
    --reco-file /path/to/output/neutrino_hw/merged_reco_results.h5 \
    --dataset neutrino \
    --sim-dir /path/to/gzk_hdf5_files/

# Pulser sim (truth parsed from NUR filenames)
python evaluate_reco_results.py \
    --reco-file /path/to/output/pulser_rxtx/merged_reco_results.h5 \
    --dataset pulser
```

The script prints median angular separation, percentiles, and the fraction of events below 1 and 2 degrees. It also prints reference values from the validated results below for comparison. These reference values are specific to the shipped validation datasets and station 23. If you are running on a different simulation set or station, your numbers will differ; treat them as a ballpark sanity check, not an exact target.

## Multiray travel time tables

Standard interferometric reconstruction assumes a single ray path between source and receiver. In a medium with a depth-dependent refractive index profile, signals can propagate via multiple paths: direct, reflected (off the surface), and refracted (bent by the index gradient). At many source geometries, two or more of these paths arrive with comparable amplitude, so the correct ray type varies by channel and source position.

The 3D module uses per-channel, per-ray-type travel time tables. In `grouped` mode (`multiray_combo_mode: "grouped"` in the config), it evaluates all physically valid ray-type combinations and selects the one that maximizes the summed correlation. Channels at similar depths are grouped together (they see the same ray type), reducing the combinatorial cost.

Each table is a 2D (R, Z) grid of travel times for one channel and one ray type, stored as an NPZ file. For station 23 with 11 channels and 4 table types (direct, reflected, refracted, plus combined), this is 44 files. The 3D configs in this directory use the per-ray-type tables (direct, refracted, reflected). The combined tables (no suffix, min travel time across ray types) are used when `multi_ray_types: false`.

### Generating tables

The table generator and SLURM submission template are in `../tables/` (shared with the 2D scripts):

```bash
cd ../tables/

# Single channel, multiray only (default)
python rz_lookup_table_creator_inice.py \
    --station 23 --channel 0 --num_threads 8 \
    --output-dir /path/to/multiray_tables/station23

# Multiray + combined tables in one pass
python rz_lookup_table_creator_inice.py \
    --station 23 --channel 0 --mode all --num_threads 8 \
    --output-dir /path/to/multiray_tables/station23

# All 11 VPOL channels via SLURM
# Args: STATION (default 23), MODE (default "all"), DET_DATE, OUTPUT_DIR
sbatch submit_rz_table_jobs.slurm 23 all "2022-10-01" /path/to/multiray_tables/station23
```

The `--mode` flag controls output: `multiray` (3 per-ray-type files), `combined` (1 min-time file), `solution_ordered` (2 solution-ordered files), or `all` (all of the above). The `--det-date` argument sets the detector description date used for antenna positions (default `2022-10-01`). Use `--detector-file` for batch jobs where MongoDB is unreachable. Tables are computed using NuRadioMC analytic raytracing with the `greenland_simple` exponential ice model. Each channel takes roughly 6 minutes on 9 cores and uses about 5 GB of memory.

### Solution-ordered tables

With the `greenland_simple` ice model, each (R,Z) geometry has 0 or 2 ray solutions (direct + refracted). The reflected solution is rare. Solution-ordered tables reorder by travel time: solution_0 = fastest arrival, solution_1 = slowest. This reduces the grouped correlator combinations from 3^N_groups (81 for 4 depth groups) to 2^N_groups (16), giving a 1.2-1.6x speedup with slightly better accuracy in the optimization tail.

To generate solution-ordered tables:

```bash
python rz_lookup_table_creator_inice.py \
    --station 23 --channel 0 --mode solution_ordered --num_threads 8 \
    --output-dir /path/to/multiray_tables/station23
```

To use them, set `table_scheme: "solution_ordered"` in the config (see the `_2table` config variants).

## Mode reference

| Mode | What it does | When to use | Runtime |
|------|-------------|-------------|---------|
| `hw` | Pass 1 only: cable delay + HW phase removal + grid search | Neutrinos (unknown source), fast pulser baseline | ~2 s/event |
| `rx` | Pass 1 + Pass 2: Rx antenna dedispersion at estimated arrival angles, local re-search | Neutrinos when runtime is acceptable | ~15 s/event |
| `rxtx` | Pass 1 + Pass 2: Rx + Tx antenna dedispersion (requires known emitter position in filename) | Pulser simulations only | ~10 s/event |

All modes run pass 1: a hierarchical 3D grid search (coarse log-spaced rho scan, peak extraction, linear refine grid, L-BFGS-B optimization) using the multiray tables.

In `rx` and `rxtx` modes, pass 2 re-reads the event and removes antenna phase dispersion before a local re-search around the pass 1 result. Antenna dispersion introduces frequency-dependent phase shifts that broaden the cross-correlation peak. Removing them sharpens the peak and improves localization.

- **Rx dedispersion** removes the receiving antenna's phase response at the arrival angles estimated from pass 1. Since the arrival direction is unknown beforehand, it requires the pass 1 result.
- **Tx dedispersion** (rxtx only) also removes the transmitting antenna's phase response at the launch angles computed from the known emitter position. The emitter position is parsed from the NUR filename (pattern `output_r{R}_zen{ZEN}_az{AZ}.nur`), so this mode only applies to pulser simulations where the source location is known.

## Input format and preprocessing

The driver script (`interferometric_reco_3d_example.py`) expects standard NuRadioMC simulation output: NUR files containing voltage traces (not electric fields). No manual preprocessing is needed. The driver applies all necessary waveform processing (cable delays, hardware response removal, upsampling, optional bandpass/CW filtering) internally before calling the reconstruction module. You do not need to apply voltage calibration, antenna deconvolution, bandpass filtering, or any other signal processing yourself.

Which preprocessing steps are applied is controlled by the config file (see table below). The shipped configs use validated defaults, so for a first pass you only need to update the paths.

If you are integrating the reconstruction module into your own processing chain rather than using the driver script, the module expects waveforms that have at minimum had cable delays applied. See the config options below for the full set of preprocessing the driver applies.

All preprocessing options are set in the YAML config file and default to the values shown below.

| Config key | Default | Description |
|------------|---------|-------------|
| `apply_cable_delay` | `true` | Subtract cable delays (`channelAddCableDelay`) |
| `apply_hw_phase_removal` | `true` | Remove hardware phase response (`hardwareResponseIncorporator`, phase-only) |
| `apply_upsampling` | `true` | Resample to 10 GHz (`channelResampler`) |
| `apply_bandpass` | `false` | Bandpass filter, 100-600 MHz Butterworth order 10 (`channelBandPassFilter`) |
| `apply_cw_removal` | `false` | CW sinewave subtraction (`channelSinewaveSubtraction`) |
| `apply_dedispersion` | `false` | Antenna phase dedispersion at broadside (`channelAntennaDedispersion`) |

The reconstruction module handles additional correlation-level options internally (hilbert envelope mode, hann windowing, correlation normalization, SNR pair weighting). These are also set in the config file and match the interface of the 2D `interferometricDirectionReconstruction` module.

For simulation data, bandpass, CW removal, and dedispersion are typically unnecessary (the permutation study confirmed bandpass is negligible for neutrinos). For real data with CW contamination, enable `apply_cw_removal: true`.

## Resource estimates

Estimates below use the solution-ordered (2-table) scheme. Ray-type (3-table) runtimes are 1.2-1.6x longer.

### Neutrino GZK (27,667 events, hw mode)

| Resource | Estimate |
|----------|----------|
| Time per event | ~1.4 s |
| Total CPU time | ~11 CPU-hours |
| Recommended chunks | 200 |
| Walltime per chunk | 10 min |
| Memory per chunk | 4 GB |

### Pulser sim (18,879 events, rxtx mode)

| Resource | Estimate |
|----------|----------|
| Time per event | ~7.8 s |
| Total CPU time | ~41 CPU-hours |
| Recommended chunks | 200 |
| Walltime per chunk | 25 min |
| Memory per chunk | 4 GB |

### Pulser sim (18,879 events, hw mode)

| Resource | Estimate |
|----------|----------|
| Time per event | ~3.5 s |
| Total CPU time | ~18 CPU-hours |
| Recommended chunks | 100 |
| Walltime per chunk | 15 min |
| Memory per chunk | 4 GB |

See [`benchmarking/README.md`](benchmarking/README.md) for detailed per-stage
breakdowns and percentile distributions.

## Output format

Each HDF5 output file contains a `results` group with these datasets:

| Dataset | Shape | Description |
|---------|-------|-------------|
| `rho` | (N,) | Reconstructed radial distance (m) |
| `phi` | (N,) | Reconstructed azimuth (deg) |
| `z` | (N,) | Reconstructed depth (m) |
| `max_corr` | (N,) | Peak correlation value |
| `run_number` | (N,) | Event group ID from NUR file |
| `event_number` | (N,) | Sub-event index |
| `source_file` | (N,) | Source NUR filename |
| `pass1_rho` | (N,) | Pass 1 rho (rx/rxtx mode only) |
| `pass1_phi` | (N,) | Pass 1 phi (rx/rxtx mode only) |
| `pass1_z` | (N,) | Pass 1 z (rx/rxtx mode only) |
| `pass1_corr` | (N,) | Pass 1 correlation (rx/rxtx mode only) |

For neutrino truth comparison, the paired HDF5 files contain `xx`, `yy`, `zz` vertex coordinates in the simulation frame. Convert to cylindrical relative to the phased array center for angular separation calculations.

## Validated results

### Neutrino GZK (hw mode, 27,667 events, 300K noise)

| Table scheme | Median | 68th | < 1 deg | < 2 deg | Runtime |
|-------------|--------|------|---------|---------|---------|
| Solution-ordered (2-table) | 1.48 deg | 2.89 deg | 38% | 59% | ~1.4 s/event |
| Ray-type (3-table) | 1.48 deg | 2.89 deg | 38% | 59% | ~2.2 s/event |

### Pulser sim (rxtx mode, 18,879 events)

| Table scheme | Median | 68th | 90th | < 1 deg | < 2 deg | Runtime |
|-------------|--------|------|------|---------|---------|---------|
| Solution-ordered (2-table) | 0.41 deg | 1.20 deg | 11.20 deg | 65% | 76% | ~7.8 s/event |
| Ray-type (3-table) | 0.42 deg | 1.21 deg | 13.99 deg | 65% | 76% | ~9.5 s/event |

The 2-table scheme is recommended: same or better accuracy with 1.2-1.6x faster runtime. The 90th percentile improvement on pulsers (11.2 vs 14.0 deg) comes from fewer optimizer combos leading to more reliable convergence.

Neutrino results use `reco3d_neutrino_gzk.yaml` (or `_2table` variant). Pulser results use `reco3d_pulser_sim.yaml` (or `_2table` variant) in rxtx mode. Your results should match when using the same configs, tables, and datasets. For other simulation sets, stations, or energy ranges, treat these as a ballpark reference rather than an exact target.

## Benchmarking

Detailed per-stage timing breakdowns, memory profiles, and profiling scripts
are in the `benchmarking/` directory. See
[`benchmarking/README.md`](benchmarking/README.md) for canonical results on the
GZK neutrino and simulated pulser datasets.

Quick timing check for a single event:

```bash
python benchmarking/benchmark_kernels.py \
    --config configs/reco3d_neutrino_gzk.yaml \
    --nur-file /path/to/neutrino.nur
```

## File listing

```
NuRadioReco/modules/
  interferometricDirectionReconstruction3D.py   Core 3D reconstruction module

NuRadioReco/examples/RNOG/interferometric_reco_ex/
  tables/
    rz_lookup_table_creator_inice.py      Table generator (multiray, combined, or both)
    submit_rz_table_jobs.slurm            SLURM submission for all VPOL channels
  reco3d/
    interferometric_reco_3d_example.py    Driver: preprocessing + pass1 + optional pass2
    evaluate_reco_results.py          Evaluate reco results against sim truth (neutrino or pulser)
    fast_grouped_multiray.py              Numba-accelerated grouped multiray correlator
    submit_reco3d_example.sh              Example SLURM batch submission
    RECO3D_QUICKSTART.md                  This file
    benchmarking/
      README.md                           Benchmark results and methodology
      benchmark_kernels.py                Single-event kernel comparison
      summarize_batch_timing.py           Summarize timing from batch HDF5 results
      profile_memory.py                   Peak RSS memory profiling
    configs/
      reco3d_neutrino_gzk.yaml           Neutrino config, ray-type tables (hw mode)
      reco3d_neutrino_gzk_2table.yaml    Neutrino config, solution-ordered tables (recommended)
      reco3d_pulser_sim.yaml             Pulser config, ray-type tables (rxtx mode)
      reco3d_pulser_sim_2table.yaml      Pulser config, solution-ordered tables (recommended)
      reco3d_pulser_sim_fast.yaml        Fast pulser config (hw mode, no dedispersion)
```
