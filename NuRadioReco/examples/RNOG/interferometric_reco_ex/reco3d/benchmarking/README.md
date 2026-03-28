# Benchmarking: 3D interferometric reconstruction

Per-stage timing breakdowns and memory profiles for the 3D interferometric
reconstruction on the canonical validation datasets. Use these numbers to
verify your setup produces comparable performance and to estimate resource
requirements before submitting batch jobs.

## Canonical timing: neutrino GZK (hw mode)

27,667 triggered events from 200 noisy GZK simulation files (10^18-10^20 eV,
300K noise), station 23.
Config: `reco3d_neutrino_gzk.yaml` (hilbert=traces, hann=on, snr_weight=off).

| Stage | Median (s) | Mean (s) | 25th (s) | 75th (s) | 90th (s) |
|-------|-----------|---------|---------|---------|---------|
| Preprocessing | 0.07 | 0.28 | 0.07 | 0.07 | 0.08 |
| Coarse scan | 1.08 | 1.08 | 1.02 | 1.14 | 1.17 |
| Refine scan | 0.64 | 1.32 | 0.56 | 1.80 | 3.01 |
| Optimizer | 0.33 | 0.34 | 0.28 | 0.38 | 0.44 |
| **Total** | **2.24** | **3.02** | **2.04** | **3.35** | **4.57** |

The coarse scan is consistent across events (~1.08s). Refine scan varies with
the number and spread of coarse peaks (0.35-3.24s at 25th-90th percentile).
Preprocessing is dominated by the first event's JIT compilation and table I/O;
subsequent events take ~0.07s.

### Speedup vs baseline (point-major kernels)

| Stage | Baseline (s) | Pair-major (s) | Speedup |
|-------|-------------|---------------|---------|
| Coarse scan | 1.99 | 1.08 | 1.84x |
| Refine scan | 1.68 | 0.64 | 2.63x |
| Optimizer | 0.31 | 0.33 | ~1x |
| **Total** | **4.11** | **2.24** | **1.84x** |

Results are bitwise identical between baseline and pair-major (verified on all
27,667 events: zero difference in rho, phi, z, and max_corr).

## Canonical timing: simulated pulser (rxtx mode)

18,879 triggered events from 6,840 simulated pulser NUR files (r: 10-200m,
zen: 20-160 deg, az: 0-360 deg), station 23.
Config: `reco3d_pulser_sim.yaml` (hilbert=none, hann=on, energy normalization,
pass2_hierarchical=true).

| Stage | Median (s) | Mean (s) | 25th (s) | 75th (s) | 90th (s) |
|-------|-----------|---------|---------|---------|---------|
| P1 preproc | 0.08 | 0.38 | 0.08 | 0.09 | 0.09 |
| P1 coarse | 4.33 | 4.32 | 4.14 | 4.48 | 4.62 |
| P1 refine | 2.40 | 2.72 | 2.16 | 3.00 | 3.93 |
| P1 optimizer | 0.47 | 0.47 | 0.40 | 0.54 | 0.61 |
| **P1 total** | **7.34** | **7.64** | **7.06** | **7.94** | **8.97** |
| P2 dedispersion | 0.13 | 0.16 | 0.10 | 0.14 | 0.15 |
| P2 coarse | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| P2 refine | 1.33 | 1.49 | 1.17 | 1.63 | 2.13 |
| P2 optimizer | 0.47 | 0.48 | 0.41 | 0.55 | 0.63 |
| **P2 reco** | **1.91** | **2.06** | **1.73** | **2.19** | **2.71** |
| **Total** | **9.45** | **10.24** | **9.05** | **10.33** | **11.92** |

Speedup vs previous baseline (hierarchical pass 2, point-major kernels):
19.1 s/event -> 9.5 s/event = **2.01x**.

Accuracy: 0.42 deg median angular separation, 65% < 1 deg, 76% < 2 deg
(identical to previous baseline, evaluated with `evaluate_reco_results.py`).

## Memory usage

Peak RSS measured from SLURM MaxRSS across the validation batches:

| Dataset | Mode | Peak RSS |
|---------|------|----------|
| Neutrino GZK | hw | ~3.0 GB |
| Pulser sim | rxtx | ~3.7 GB |

Memory is dominated by the travel-time table load at initialization
(44 NPZ files per station). Per-event processing adds negligible memory
beyond the initial footprint. The pulser rxtx mode uses more memory due to
the antenna model loaded for Tx dedispersion.

## Reproducing these results

### Quick check: single-event kernel benchmark

Compares point-major and pair-major kernels on one event (takes ~2 min
including JIT warmup):

```bash
python benchmarking/benchmark_kernels.py \
    --config configs/reco3d_neutrino_gzk.yaml \
    --nur-file /path/to/neutrino.nur
```

### Summarize timing from a batch run

After running a SLURM batch with `submit_reco3d_example.sh`, summarize the
per-stage timing from the merged output:

```bash
# Neutrino hw mode
python benchmarking/summarize_batch_timing.py \
    --reco-file /path/to/merged_reco_results.h5 --mode hw

# Pulser rxtx mode
python benchmarking/summarize_batch_timing.py \
    --reco-file /path/to/merged_reco_results.h5 --mode rxtx
```

### Memory profiling

Profile peak RSS through each reconstruction stage on a single event:

```bash
python benchmarking/profile_memory.py \
    --config configs/reco3d_neutrino_gzk.yaml \
    --nur-file /path/to/neutrino.nur \
    --mode hw
```

Add `--tracemalloc` for Python-side allocation detail (adds overhead).

### Full batch reproduction

See `RECO3D_QUICKSTART.md` for SLURM batch submission instructions. The
canonical results above were produced with:

```bash
# Neutrino (200 chunks, hw mode)
bash submit_reco3d_example.sh \
    --config configs/reco3d_neutrino_gzk.yaml \
    --data-dir /path/to/gzk_noise_nur_files/ \
    --output-dir /path/to/output/ \
    --account your_account --mode hw --n-chunks 200 \
    --mem 4GB --walltime 00:20:00

# Pulser (200 chunks, rxtx mode)
bash submit_reco3d_example.sh \
    --config configs/reco3d_pulser_sim.yaml \
    --data-dir /path/to/pulser_scan_data/ \
    --output-dir /path/to/output/ \
    --account your_account --mode rxtx --n-chunks 200 \
    --mem 4GB --walltime 00:30:00
```

## HDF5 timing keys reference

The driver script writes per-event timing to the output HDF5. Which keys
are present depends on the reconstruction mode.

### hw mode

| Key | Description |
|-----|-------------|
| `preproc_time` | Cable delay + HW phase + upsampling |
| `coarse_time` | Coarse grid scan (per-pair correlator) |
| `refine_time` | Refine grid scan (grouped correlator) |
| `opt_time` | L-BFGS-B optimization |
| `post_time` | Post-processing (if enabled) |
| `raw_refine_time` | Raw trace refinement (if enabled) |

Total = preproc_time + coarse_time + refine_time + opt_time

### rx / rxtx mode

| Key | Description |
|-----|-------------|
| `p1_preproc_time` | Pass 1 preprocessing |
| `p1_coarse_time` | Pass 1 coarse scan |
| `p1_refine_time` | Pass 1 refine scan |
| `p1_opt_time` | Pass 1 optimizer |
| `p1_total_time` | Pass 1 total (coarse + refine + opt, excludes preproc) |
| `p2_dedisp_time` | Pass 2 antenna dedispersion |
| `p2_coarse_time` | Pass 2 coarse scan |
| `p2_refine_time` | Pass 2 refine scan |
| `p2_opt_time` | Pass 2 optimizer |
| `p2_reco_time` | Pass 2 total reconstruction |

Total = p1_preproc_time + p1_total_time + p2_dedisp_time + p2_reco_time

## Environment

The canonical results were measured on:

- SLURM cluster compute nodes: Intel Xeon Gold 6342 @ 2.80 GHz
- Python 3.11, Numba 0.62.1, NumPy 2.3.4, SciPy 1.16.2
- Single-threaded per job (Numba parallel within each job)
- Date: 2026-03-25
