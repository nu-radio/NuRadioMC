# RNO-G trigger simulation with FLOWER trigger model

General-purpose RNO-G neutrino/CR simulation with a FLOWER trigger model and two noise modes.

## Table of contents

- [Overview](#overview)
- [Files](#files)
- [Noise modes](#noise-modes)
- [FLOWER trigger model](#flower-trigger-model)
- [ADC pedestal and asymmetric saturation](#adc-pedestal-and-asymmetric-saturation)
- [Usage](#usage)
- [CLI arguments](#cli-arguments)
- [Output](#output)
- [Known limitations](#known-limitations)
- [Framework changes on this branch](#framework-changes-on-this-branch)

## Overview

`simulate.py` wraps the NuRadioMC simulation framework with three features:

1. **Measured noise injection.** Real forced-trigger (FT) waveforms replace synthetic thermal noise via `--ft_noise_dir`. See [noise modes](#noise-modes).

2. **Asymmetric ADC saturation.** Models the off-center pedestal bias of the RADIANT ADC via `--pedestal_voltage`. See [ADC pedestal](#adc-pedestal-and-asymmetric-saturation) and [`pedestal_extraction/`](pedestal_extraction/).

3. **FLOWER trigger model.** `triggerBoardResponse` + `highLowThreshold`. First-pass approximation; see [known limitations](#known-limitations).

All three are optional. Without `--ft_noise_dir`, thermal noise is used. Without `--pedestal_voltage`, the ADC range is symmetric. The FLOWER trigger is always active.

## Files

| File | Description |
|------|-------------|
| `simulate.py` | Simulation script supporting thermal and FT noise modes |
| `RNO_config.yaml` | Default NuRadioMC config (`noise: False` for FT mode) |
| [`noise_analysis/`](noise_analysis/) | FT noise cleaning (clean mask) and trigger-path Vrms extraction |
| [`pedestal_extraction/`](pedestal_extraction/) | ADC pedestal extraction from satellite data, produces per-channel clip thresholds |

## Noise modes

### Thermal (default)

Without `--ft_noise_dir`, the framework generates noise from a 300 K temperature model through the signal chain response.

### Measured FT noise (`--ft_noise_dir`)

FT waveforms are recorded through the readout signal chain (RADIANT, 3.2 GHz). The trigger path uses a different signal chain after the 3 dB splitter (arXiv:2411.12922, Sec. 3.2). So FT noise must be injected differently for each path:

1. **Trigger path** (at 5 GHz internal sim rate): FT noise is upsampled from 3.2 to 5 GHz (to match the internal sim rate), then multiplied by a transfer function (`trigger_response / readout_response`, from the detector description) to convert from readout domain to trigger domain. Injected into trigger channel copies only.

2. **Readout path** (at 3.2 GHz): the same FT event is added directly to readout channels at native rate. No transform needed since the noise was already recorded through the readout chain.

Both stages use the same noise realization. The config must set `noise: False`.

Point `--ft_noise_dir` at a directory of ROOT files (`station{id}_run*.root` or `run*/waveforms.root`). To exclude non-thermal FT events, pass `--ft_clean_mask` with an NPZ mask file from [`noise_analysis/ft_cleaning/`](noise_analysis/ft_cleaning/).

## FLOWER trigger model

`triggerBoardResponse` (VGA gain + 8-bit ADC) followed by `highLowThreshold`:

- Threshold: ~3.76 sigma at 1 Hz rate
- Coincidence: 2-fold across PA channels 0-3
- High-low window: 6 samples at FLOWER rate (~472 MSa/s)
- Coincidence window: 20 samples at FLOWER rate

In FT mode, the trigger-path Vrms comes from `TRIGGER_VRMS_FT` (measured from FT data). In thermal mode, it is computed from the noise temperature and the trigger signal chain response. See [`noise_analysis/ft_cleaning/`](noise_analysis/ft_cleaning/) for how the FT Vrms values were derived and their limitations.

## ADC pedestal and asymmetric saturation

The RADIANT ADC digitizes a 0-2.5V range. The pedestal bias sits at ~1.5V, off-center from the 1.25V midpoint, making the effective clip range asymmetric in pedestal-subtracted coordinates: [-1500, +1000] mV for a 1.5V pedestal.

`--pedestal_voltage` accepts a single value for all channels. For per-channel precision, use `analogToDigitalConverter.set_pedestal_voltage(dict)` programmatically. See [`pedestal_extraction/`](pedestal_extraction/) for measured per-channel values.

## Usage

### FT noise mode

```bash
python simulate.py \
    --config /path/to/config.yaml \
    --station_id 23 \
    --energy 1e18 \
    --n_events 1000 \
    --ft_noise_dir /path/to/forced_triggers/station23 \
    --trigger_vrms /path/to/trigger_vrms.yaml \
    --ft_clean_mask /path/to/clean_mask_station23.npz \
    --ft_seed 12345 \
    --pedestal_voltage 1.5 \
    --output_file output.hdf5 \
    --data_dir /path/to/output
```

### Thermal noise mode

```bash
python simulate.py \
    --station_id 23 \
    --energy 1e18 \
    --n_events 1000 \
    --output_file output.hdf5 \
    --data_dir /path/to/output
```

### Parallel production via SLURM

```bash
#SBATCH --array=0-99
python simulate.py --station_id 23 --energy 1e18 --n_events 100 \
    --index $SLURM_ARRAY_TASK_ID \
    --output_file "chunk_${SLURM_ARRAY_TASK_ID}.hdf5" \
    --data_dir /path/to/output ...
```

## CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `RNO_config.yaml` | NuRadioMC YAML config (can include a `fiducial_volume` section) |
| `--station_id` | required | Station ID |
| `--energy` | 1e18 | Neutrino energy in eV |
| `--n_events` | 1000 | Number of events to simulate |
| `--detector_file` | None (MongoDB) | Fallback detector file when MongoDB is unavailable |
| `--ft_noise_dir` | None | FT data directory (enables measured noise mode) |
| `--ft_seed` | None | Reproducibility seed for FT noise selection |
| `--ft_clean_mask` | None | NPZ clean mask to exclude non-thermal FT events |
| `--trigger_vrms` | None | YAML with per-channel trigger-path Vrms (required for FT mode) |
| `--pedestal_voltage` | 1.5 | ADC pedestal in V for asymmetric clipping |
| `--noise_temperatures` | None | JSON with per-channel noise temperatures (K), overrides DB values |
| `--fiducial_rmax` | from config | Max fiducial radius in m (overrides `fiducial_volume.rmax` in config) |
| `--min_zenith` | from config (0) | Min zenith in degrees (overrides `fiducial_volume.min_zenith` in config) |
| `--max_zenith` | from config (60) | Max zenith in degrees (overrides `fiducial_volume.max_zenith` in config) |
| `--nur_output` | False | Also write NUR files |
| `--index` | 0 | Chunk index for parallel runs |

## Output

- **HDF5**: standard NuRadioMC output with triggered event data
- **NUR** (optional): NuRadioReco event files
- **Ledger CSV**: one row per input event with `event_group_id`, `zenith_deg`, `azimuth_deg`, `energy_eV`, `flavor`, `status` (`triggered` / `trigger_failed` / `efield_cut`), `max_amp_ch{0-3}_mV`

## Known limitations

- **Trigger Vrms must be pre-extracted.** `simulate.py` reads trigger-path Vrms from a YAML file (`--trigger_vrms`). Extract it using `noise_analysis/trigger_vrms/extract_trigger_vrms.py` before running. See [`noise_analysis/trigger_vrms/`](noise_analysis/trigger_vrms/) for the extraction script and convergence study.

- **VGA gain mismatch.** The simulated VGA gain selection does not match the real FLOWER hardware. Under investigation.

- **Single pedestal voltage for all channels.** `--pedestal_voltage` applies one value. Real per-channel pedestals vary. For per-channel values, call `analogToDigitalConverter.set_pedestal_voltage()` with a dict in your own script. See [`pedestal_extraction/`](pedestal_extraction/).

- **No pedestal in the detector database.** The RNO-G MongoDB doesn't store pedestal voltages yet, so they must be passed via `--pedestal_voltage`.

## Framework changes on this branch

This branch (`ft_noise_trigger_sim`) adds to NuRadioMC:

1. **`noiseImporter`**: trigger copy injection, two-stage mode, explicit file list parameter
2. **`analogToDigitalConverter`**: pedestal voltage support (`set_pedestal_voltage()`)
3. **`readRNOGDataMattak`**: `ValueError` catch for corrupt ROOT files
4. **`efieldToVoltageConverterPerEfield`**: pre/post pulse zero-padding for linear convolution
5. **`rnog_detector`**: response_chain dict-to-list format conversion for exported detector files
6. **`highLowThreshold`**: channel ID included in trace_start_time warning
