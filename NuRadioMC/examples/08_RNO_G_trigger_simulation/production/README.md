# Production workflow

Snakemake workflow that throws `simulate.py` chunks until each energy bin reaches a
target triggered count, then writes a per-bin manifest of the chunks needed to hit that
target. One chunk is one SLURM job producing a NUR, an HDF5, and a per-event ledger CSV.
Everything site-specific lives in `config/config.yaml`; the Snakefile and scripts contain
no absolute paths.

## Layout

```
production/
  Snakefile                     # rules: throw_chunk, truncate_lgE, register_dataset
  config/
    config.yaml.example         # copy to config.yaml and edit
    config_station13.yaml       # calibrated 2022 station-13 production parameters
    config_station23.yaml       # calibrated 2022 station-23 production parameters
    profile/config.yaml         # snakemake profile (slurm executor, jobs, retries)
  scripts/
    truncate_to_target.py       # pick chunks summing to the target, write manifest.txt
    write_readme.py             # write the dataset README into data_dir
  workflow_logs/                # per-chunk logs (created at run time)
```

Output goes to `data_dir` (set in config):

```
<data_dir>/
  lgE16.0/
    lgE16.0_c0000.nur           # waveforms
    lgE16.0_c0000.hdf5          # generator input event list
    lgE16.0_c0000_ledger.csv    # per-event outcome (status column)
    ...
    manifest.txt                # chunk basenames kept for the target-trigger sample
  lgE16.5/ ...
  README.md                     # auto-generated dataset summary
```

## Config keys

| key | meaning |
|-----|---------|
| `sim_script` | path to `simulate.py` |
| `sim_config` | path to the NuRadioMC YAML config (`RNO_config.yaml`) |
| `python_bin` | interpreter used to run the sim (default `python3`) |
| `pythonpath` | prepended to `PYTHONPATH` so the sim imports the intended NuRadioMC checkout; empty to use the active environment |
| `env_setup` | shell command run in each chunk job before the sim, to activate the environment that has `mattak`, `proposal`, and the NuRadioMC stack (for example `source <conda.sh> && conda activate <env>`); empty if the submitting environment already has them |
| `data_dir` | output directory for all bins |
| `station_id` | RNO-G station id |
| `detector_file` | detector description file; empty string queries MongoDB at `event_time` |
| `event_time` | detector epoch for the MongoDB query when `detector_file` is empty |
| `ft_noise_dir` | forced-trigger noise directory the sim draws injected noise from |
| `ft_clean_mask` | clean-mask npz excluding contaminated FT events |
| `trigger_vrms` | YAML of trigger-path Vrms per channel (empty lets the sim default). Stations 13 and 23 use `trigger_vrms_station{13,23}_calibrated.yaml`, which pair with the calibrated season-2022 readout detector description; the other stations' `trigger_vrms_station{NN}.yaml` carry DB-transfer values |
| `clip_thresholds` | YAML of per-channel ADC clip bounds; use the `pedestal_extraction/clip_thresholds_station{station_id}.yaml` matching `station_id` (empty falls back to the uniform `pedestal_voltage` clip). The shipped per-station YAMLs carry the measured 2022 values used in production |
| `pedestal_voltage` | ADC pedestal voltage in volts; the uniform-clip fallback when `clip_thresholds` is empty |
| `fiducial_rmax` | selects the near-surface CR-proxy fiducial volume with this radius in m (see the example README's CR proxy section); empty falls back to the energy-dependent neutrino volume, and an explicit `fiducial_volume` block in the sim config overrides both |
| `flavor` | neutrino flavor (`e`, `mu`, `tau`, `all`) |
| `interaction_type` | `cc`, `nc`, or `ccnc` |
| `ft_seed_base` | added to `chunk_id` for a deterministic per-chunk FT seed |
| `target_triggers_per_bin` | triggered events to reach per energy bin |
| `safety_margin` | throw this multiple of the naive rate estimate |
| `energies` | list of lgE bin labels; the sim receives `10^lgE` eV |
| `trigger_rates` | per-bin trigger probability, sets the chunk count per bin |
| `thrown_per_chunk` | events thrown per chunk per bin |
| `slurm_resources` | per-bin `mem_mb` and `runtime_min` for attempt 1 (retries scale up) |
| `accounts` | list of `{account, partition, weight}` for round-robin routing |

The lgE labels in `energies` are converted to `10^lgE` eV for `simulate.py --energy`, which
takes energy in eV. `throw_chunk` always passes `--nur_output` (so the NUR is written) and
`--trigger_vrms` / `--pedestal_voltage` for the calibrated FLOWER trigger.

## Data requirements

A collaborator needs, for their station and year:

- Standard RNO-G full-waveform run data (`station{id}_run*.root`) in `ft_noise_dir`, obtained
  through normal collaboration data access. The pool selects `FORCE` events itself, so no
  pre-filtering is needed.
- A clean mask for that station and year (`ft_clean_mask`); the shipped
  `noise_analysis/ft_cleaning/clean_mask_station{NN}.npz` are 2022. Running without a mask is
  allowed but injects contaminated FT events.
- The detector description: MongoDB access (leave `detector_file` empty, it queries at
  `event_time`) or a detector file.
- Trigger vrms and ADC clip thresholds: stations 13 and 23 use the shipped
  `noise_analysis/trigger_vrms/trigger_vrms_station{13,23}_calibrated.yaml` with the
  calibrated season-2022 readout detector file; the other stations'
  `trigger_vrms_station{NN}.yaml` carry DB-transfer values. Clip thresholds
  (`pedestal_extraction/clip_thresholds_station{NN}.yaml`) are measured 2022 values.
  Other years or detector epochs need re-derivation with the tools in those
  directories.
- Antenna response models: NuRadioReco downloads them on first use (~1.5 GB into
  `NuRadioReco/detector/AntennaModels/`). Compute nodes without network access need the
  models pre-provisioned by a first run (or copy) on a connected node.

## Use

Copy and edit the config, then dry-run:

```bash
cd production
cp config/config.yaml.example config/config.yaml
# edit config/config.yaml
snakemake -n
```

To regenerate the calibrated 2022 station-13 or station-23 datasets, start from
`config_station13.yaml` or `config_station23.yaml` instead: they carry the production
parameters (targets, rates, seeds, trigger model, per-station input files) with the
site-specific entries (`/path/to/...`, `env_setup`, `accounts`) left to fill in. The
station-13 config uses the `measured_8x` ch0 trigger model; both need the calibrated
season-2022 readout detector file, distributed separately.

Single-chunk pilot (validate resources and end-to-end submission):

```bash
snakemake --executor slurm --jobs 1 --workflow-profile config/profile \
  <data_dir>/lgE18.5/lgE18.5_c0000_ledger.csv
```

Full production (run the driver detached, e.g. in tmux, so it survives an SSH drop):

```bash
tmux new-session -d -s production \
  'snakemake --executor slurm --jobs 200 --workflow-profile config/profile'
```

Resume or re-run failed chunks: rerun the same command. `--rerun-incomplete` re-throws
chunks with partial outputs, `restart-times: 2` retries transient failures, and
`keep-going: true` lets healthy chunks finish while one is failing. To raise the target
later, edit `target_triggers_per_bin` (and `safety_margin` if needed) and rerun; only the
bins that need more chunks throw them, and the manifests are rewritten.

## Account routing

Each chunk is round-robin assigned to an account by `chunk_id % 10` in proportion to the
`accounts` weights (weights 100/100 give a 50/50 split). Multiple pools fill in parallel;
SLURM picks which CPU runs each job. Adjust the weights, accounts, or partitions in
`config.yaml` as the available capacity changes.

## Notes

- The sim writes NUR + HDF5 + ledger on success; partial outputs are detected by
  `--rerun-incomplete` and re-thrown.
- Chunks beyond the target stay on disk as overage. To load the analysis sample, read only
  the NUR files listed in each bin's `manifest.txt`.
- The per-energy `trigger_rates`, `thrown_per_chunk`, and `slurm_resources` are estimates;
  measure them with a small pilot per station before a large production and update the config.
