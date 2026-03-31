# Benchmarking: 3D interferometric reconstruction

Per-stage timing breakdowns and memory profiles for the 3D interferometric
reconstruction on the canonical validation datasets. Use these numbers to
verify your setup produces comparable performance and to estimate resource
requirements before submitting batch jobs.

## Canonical timing: neutrino GZK (hw mode)

27,667 triggered events from 200 noisy GZK simulation files (10^18-10^20 eV,
300K noise), station 23.

### Solution-ordered tables (2-table, recommended)

Config: `reco3d_neutrino_gzk_2table.yaml` (hilbert=traces, hann=on, snr_weight=off, table_scheme=solution_ordered).

| Stage | Median (s) |
|-------|-----------|
| Coarse scan | 0.68 |
| Refine scan | 0.56 |
| Optimizer | 0.12 |
| **Total** | **1.43** |

### Ray-type tables (3-table)

Config: `reco3d_neutrino_gzk.yaml` (same preprocessing, table_scheme=ray_type).

| Stage | Median (s) |
|-------|-----------|
| Coarse scan | 1.12 |
| Refine scan | 0.78 |
| Optimizer | 0.35 |
| **Total** | **2.27** |

### Speedup summary (neutrino hw)

| Optimization | Median s/event | Cumulative speedup |
|-------------|---------------|-------------------|
| Original (point-major, 3-table) | ~6.0 | 1x |
| Pair-major kernels, 3-table | 2.27 | 2.6x |
| Pair-major kernels, 2-table | 1.43 | 4.2x |

The optimizer sees the largest per-stage improvement from 2-table (2.86x)
because it evaluates the grouped combo function at every L-BFGS-B iteration.
Results are bitwise identical between 3-table pair-major and the original
point-major baseline (verified on all 27,667 events).

## Canonical timing: simulated pulser (rxtx mode)

18,879 triggered events from 6,840 simulated pulser NUR files (r: 10-200m,
zen: 20-160 deg, az: 0-360 deg), station 23.

### Solution-ordered tables (2-table, recommended)

Config: `reco3d_pulser_sim_2table.yaml` (hilbert=none, hann=on, energy normalization,
pass2_hierarchical=true, table_scheme=solution_ordered).

| Stage | Median (s) |
|-------|-----------|
| P1 coarse | 3.00 |
| P1 refine | 2.25 |
| P1 optimizer | 0.17 |
| **P1 total** | **5.54** |
| P2 reco | 1.99 |
| **Total** | **7.78** |

Accuracy: 0.41 deg median, 65% < 1 deg, 76% < 2 deg, 90th percentile 11.20 deg.

### Ray-type tables (3-table)

Config: `reco3d_pulser_sim.yaml` (same preprocessing, table_scheme=ray_type).

| Stage | Median (s) |
|-------|-----------|
| P1 coarse | 4.31 |
| P1 refine | 2.42 |
| P1 optimizer | 0.46 |
| **P1 total** | **7.35** |
| P2 reco | 1.91 |
| **Total** | **9.46** |

Accuracy: 0.42 deg median, 65% < 1 deg, 76% < 2 deg, 90th percentile 13.99 deg.

### Speedup summary (pulser rxtx)

| Optimization | Median s/event | Cumulative speedup |
|-------------|---------------|-------------------|
| Original (hierarchical P2, point-major, 3-table) | 19.1 | 1x |
| Pair-major kernels, 3-table | 9.46 | 2.0x |
| Pair-major kernels, 2-table | 7.78 | 2.5x |

The 2-table scheme also slightly improves the 90th percentile accuracy
(11.2 vs 14.0 deg) because fewer combos help the optimizer converge more
reliably.

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
- Date: 2026-03-28 (2-table), 2026-03-25 (3-table)
