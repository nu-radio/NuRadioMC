# 3D interferometric reconstruction: quickstart

This guide explains how to reproduce the 3D interferometric reconstruction results on the GZK neutrino simulation and the simulated pulser calibration dataset.

## Prerequisites

1. **NuRadioMC/NuRadioReco** installed (this repo, `deep_cr_reco` branch or later)
2. **Python 3.11+** with numpy, scipy, h5py, pyyaml, numba (optional but recommended for speed)
3. **Multiray travel time tables** for station 23 (44 NPZ files: 11 channels x 4 ray types each). Not included in the repo. On the Chicago cluster, the tables are at `/data/reconstruction/validation_sets/test_tables/multiray_tables/station23/`. For other systems, see the RNO-G internal wiki for download instructions.
4. **RNO-G MongoDB access** (detector description is loaded from `radio.zeuthen.desy.de:27017` by default)
5. **Simulation datasets** (see below)

## Datasets

### GZK neutrino simulation

100 paired NUR/HDF5 files. 12,916 triggered events at station 23, energy range 10^18 to 10^20 eV with GZK-weighted spectrum.

Filename pattern:
```
nu_e_ccnc_1e18_1e20eV_GZK-2_IceCube-nu-2022_{NNNNNN}.nur
nu_e_ccnc_1e18_1e20eV_GZK-2_IceCube-nu-2022_{NNNNNN}.hdf5
```

On the Chicago cluster, the dataset is at `/data/reconstruction/validation_sets/sim_neutrinos/sim_output_gzk/`. For other systems, see the RNO-G internal wiki.

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

On the Chicago cluster, the dataset is at `/data/reconstruction/validation_sets/sim_cal_pulsers/test_set/`. For other systems, see the RNO-G internal wiki.

## Setup

1. Edit `time_delay_tables` in the config file to point to the parent directory of `station23/`. On the Chicago cluster:

```yaml
time_delay_tables: "/data/reconstruction/validation_sets/test_tables/multiray_tables"
```

The code appends `station{ID}/` internally, so it will look for files at `<time_delay_tables>/station23/st23_ch{N}_rz_table_{ray_type}.npz`.

2. Verify the table files are present:

```bash
ls /data/reconstruction/validation_sets/test_tables/multiray_tables/station23/st23_ch0_rz_table_direct.npz
```

## Running reconstruction

### Single file (interactive)

```bash
cd reco3d/

# Neutrino, hw mode (no antenna dedispersion, ~9 s/event)
python interferometric_reco_3d_example.py \
    --config configs/reco3d_neutrino_gzk.yaml \
    --mode hw \
    -i /path/to/nu_e_ccnc_1e18_1e20eV_GZK-2_IceCube-nu-2022_000000.nur \
    -o results/test_neutrino_hw.h5

# Pulser, rxtx mode (Rx + Tx antenna dedispersion, ~94 s/event)
python interferometric_reco_3d_example.py \
    --config configs/reco3d_pulser_sim.yaml \
    --mode rxtx \
    -i /path/to/output_r50.0_zen90.0_az0.0.nur \
    -o results/test_pulser_rxtx.h5

# Pulser, hw mode (no antenna dedispersion, ~9 s/event)
python interferometric_reco_3d_example.py \
    --config configs/reco3d_pulser_sim.yaml \
    --mode hw \
    -i /path/to/output_r50.0_zen90.0_az0.0.nur \
    -o results/test_pulser_hw.h5
```

### Batch (SLURM)

Edit `submit_reco3d_example.sh` to set your ACCOUNT, PARTITION, N_CHUNKS, then:

```bash
# Neutrino GZK, hw mode
bash submit_reco3d_example.sh \
    configs/reco3d_neutrino_gzk.yaml \
    /path/to/gzk_nur_files/ \
    /path/to/output/neutrino_hw/ \
    hw

# Pulser sim, rxtx mode
bash submit_reco3d_example.sh \
    configs/reco3d_pulser_sim.yaml \
    /path/to/pulser_scan_data/ \
    /path/to/output/pulser_rxtx/ \
    rxtx
```

The script splits NUR files across chunks, submits parallel SLURM jobs, and queues a merge job (inline HDF5 concatenation) that runs after all chunks complete. The final output is `merged_reco_results.h5`.

## Multiray travel time tables

Standard interferometric reconstruction assumes a single ray path between source and receiver. In a medium with a depth-dependent refractive index profile, signals can propagate via multiple paths: direct, reflected (off the surface), and refracted (bent by the index gradient). At many source geometries, two or more of these paths arrive with comparable amplitude, so the correct ray type varies by channel and source position.

The 3D module uses per-channel, per-ray-type travel time tables. In `grouped` mode (`multiray_combo_mode: "grouped"` in the config), it evaluates all physically valid ray-type combinations and selects the one that maximizes the summed correlation. Channels at similar depths are grouped together (they see the same ray type), reducing the combinatorial cost.

Each table is a 2D (R, Z) grid of travel times for one channel and one ray type, stored as an NPZ file. For station 23 with 11 channels and 4 table types (direct, reflected, refracted, plus combined), this is 44 files.

## Mode reference

| Mode | What it does | When to use | Runtime |
|------|-------------|-------------|---------|
| `hw` | Pass 1 only: cable delay + HW phase removal + grid search | Neutrinos (unknown source), fast pulser baseline | ~9 s/event |
| `rx` | Pass 1 + Pass 2: Rx antenna dedispersion at estimated arrival angles, local re-search | Neutrinos when runtime is acceptable | ~30 s/event |
| `rxtx` | Pass 1 + Pass 2: Rx + Tx antenna dedispersion (requires known emitter position in filename) | Pulser simulations only | ~94 s/event |

All modes run pass 1: a hierarchical 3D grid search (coarse log-spaced rho scan, peak extraction, linear refine grid, L-BFGS-B optimization) using the multiray tables.

In `rx` and `rxtx` modes, pass 2 re-reads the event and removes antenna phase dispersion before a local re-search around the pass 1 result. Antenna dispersion introduces frequency-dependent phase shifts that broaden the cross-correlation peak. Removing them sharpens the peak and improves localization.

- **Rx dedispersion** removes the receiving antenna's phase response at the arrival angles estimated from pass 1. Since the arrival direction is unknown beforehand, it requires the pass 1 result.
- **Tx dedispersion** (rxtx only) also removes the transmitting antenna's phase response at the launch angles computed from the known emitter position. The emitter position is parsed from the NUR filename (pattern `output_r{R}_zen{ZEN}_az{AZ}.nur`), so this mode only applies to pulser simulations where the source location is known.

## Preprocessing options

The driver applies preprocessing modules before calling the reconstruction module. All options are set in the YAML config file and default to the values shown below.

| Config key | Default | Description |
|------------|---------|-------------|
| `apply_cable_delays` | `true` | Subtract cable delays (`channelAddCableDelay`) |
| `apply_hw_phase_removal` | `true` | Remove hardware phase response (`hardwareResponseIncorporator`, phase-only) |
| `apply_upsampling` | `true` | Resample to 10 GHz (`channelResampler`) |
| `apply_bandpass` | `false` | Bandpass filter, 100-600 MHz Butterworth order 10 (`channelBandPassFilter`) |
| `apply_cw_removal` | `false` | CW sinewave subtraction (`channelSinewaveSubtraction`) |
| `apply_dedispersion` | `false` | Antenna phase dedispersion at broadside (`channelAntennaDedispersion`) |

The reconstruction module handles additional correlation-level options internally (hilbert envelope mode, hann windowing, correlation normalization, SNR pair weighting). These are also set in the config file and match the interface of the 2D `interferometricDirectionReconstruction` module.

For simulation data, bandpass, CW removal, and dedispersion are typically unnecessary (the permutation study confirmed bandpass is negligible for neutrinos). For real data with CW contamination, enable `apply_cw_removal: true`.

## Resource estimates

### Neutrino GZK (12,916 events, hw mode)

| Resource | Estimate |
|----------|----------|
| Time per event | ~9 s |
| Total CPU time | ~32 CPU-hours |
| Recommended chunks | 100 |
| Walltime per chunk | 30 min |
| Memory per chunk | 4 GB |

### Pulser sim (18,879 events, rxtx mode)

| Resource | Estimate |
|----------|----------|
| Time per event | ~94 s |
| Total CPU time | ~493 CPU-hours |
| Recommended chunks | 200 |
| Walltime per chunk | 3 hours |
| Memory per chunk | 4 GB |

### Pulser sim (18,879 events, hw mode)

| Resource | Estimate |
|----------|----------|
| Time per event | ~9 s |
| Total CPU time | ~47 CPU-hours |
| Recommended chunks | 100 |
| Walltime per chunk | 25 min |
| Memory per chunk | 4 GB |

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

### Neutrino GZK (hw mode, 99 stratified events)

| Metric | Value |
|--------|-------|
| Median angular separation | 1.10 deg |
| 68th percentile | 2.41 deg |
| 90th percentile | 13.63 deg |
| Fraction < 1 deg | 44% |
| Fraction < 2 deg | 64% |

### Pulser sim (rxtx mode, 27 stratified events)

| Metric | Value |
|--------|-------|
| Median angular separation | 0.27 deg |
| 68th percentile | 1.15 deg |
| Fraction < 1 deg | 67% |
| Fraction < 2 deg | 74% |

These numbers are from the preprocessing permutation study (March 2026) using the configs in this directory. Your results should match within statistical noise when using the same configs, tables, and datasets.

## Preprocessing permutation study summary

288 preprocessing configurations were tested on the neutrino dataset, and 432 on the pulser dataset. The configs provided here represent the best validated combinations.

Key findings:
- **HW phase removal** is the single most important preprocessing step for both datasets.
- **Hilbert envelope mode** has opposite preferences: neutrinos prefer `correlation`, pulsers prefer `none`. This reflects the difference between multiray far-field signals (where envelope extraction avoids destructive interference) and single-path near-field signals.
- **Hann windowing and bandpass** are important for pulsers but negligible for neutrinos.
- **Antenna dedispersion** (rx/rxtx mode) improves pulser results substantially (0.27 deg rxtx vs 0.74 deg hw) but offers marginal improvement for neutrinos at ~3x runtime cost.

## File listing

```
NuRadioReco/modules/
  interferometricDirectionReconstruction3D.py   Core 3D reconstruction module

NuRadioReco/examples/RNOG/interferometric_reco_ex/reco3d/
  interferometric_reco_3d_example.py        Driver: preprocessing + pass1 + optional pass2
  fast_grouped_multiray.py          Numba-accelerated grouped multiray correlator
  submit_reco3d_example.sh          Example SLURM batch submission
  RECO3D_QUICKSTART.md              This file
  configs/
    reco3d_neutrino_gzk.yaml        Best neutrino config (hw mode)
    reco3d_pulser_sim.yaml          Best pulser config (rxtx mode)
    reco3d_pulser_sim_fast.yaml     Fast pulser config (hw mode, no dedispersion)
```
