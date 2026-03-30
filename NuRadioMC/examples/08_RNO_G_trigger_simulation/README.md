# RNO-G trigger simulation with calibrated FLOWER trigger

General-purpose RNO-G neutrino/CR simulation with realistic trigger evaluation. Supports two noise modes: thermal (generated from temperature model) and measured forced-trigger (FT) noise injection from real data.

## Files

| File | Description |
|------|-------------|
| `simulate.py` | Merged simulation script supporting thermal and FT noise modes |
| `RNO_config.yaml` | Default NuRadioMC config (`noise: False` for FT mode, thermal used otherwise) |
| `noise_study/` | FT noise cleaning pipeline: contamination analysis, threshold validation, clean mask |
| `pedestal_analysis/` | Per-channel ADC pedestal extraction from satellite data (script + 2022 YAML) |

## Noise modes

### Thermal (default)

Without `--ft_noise_dir`, the framework generates noise from a temperature model (300 K blackbody through the signal chain response). This is the standard NuRadioMC noise mode, suitable for generic neutrino simulations where idealized noise is acceptable.

### Measured FT noise (`--ft_noise_dir`)

Real forced-trigger waveforms are injected in two stages:

1. **Trigger path** (in `_detector_simulation_filter_amp`, at 5 GHz): FT noise is upsampled and transformed from the readout signal chain domain to the trigger signal chain domain via a cached transfer function, then injected into trigger channel copies for FLOWER evaluation.

2. **Readout path** (in `resampler_with_noise_and_clip`, at 3.2 GHz): the same FT event is added to all readout channels at native sampling rate.

Both stages use the same noise realization (drawn and cached by `noiseImporter`). The NuRadioMC config must have `noise: False` to disable the thermal noise generator.

The readout-to-trigger transfer function is `trigger_response(f) / readout_response(f)` from the detector description, regularized at band edges to prevent division artifacts.

## FLOWER trigger model

The trigger evaluation uses `triggerBoardResponse` (VGA gain selection + ADC digitization to 8-bit counts) followed by `highLowThreshold` (bipolar threshold trigger with coincidence). Parameters:

- Threshold: ~3.76 sigma at 1 Hz rate (from `RNO_G_HighLow_Thresh`)
- Coincidence: 2-fold across PA channels 0-3
- High-low window: 6 samples at FLOWER rate
- Coincidence window: 20 samples at FLOWER rate

## ADC saturation

The RADIANT ADC operates on a 0-2.5V range with a DC pedestal bias (~1.5V). In pedestal-subtracted coordinates (what simulated traces use), the effective clip range is asymmetric: [-1500, +1000] mV for a 1.5V pedestal. The `--pedestal_voltage` argument controls this.

## Usage

### CR proxy simulation (FT noise)

```bash
python simulate.py \
    --config /path/to/cr_proxy_config.yaml \
    --station_id 23 \
    --energy 1e18 \
    --n_events 1000 \
    --detector_file /path/to/rnog_station23_2022-10-01.json.xz \
    --ft_noise_dir /path/to/forced_triggers/station23 \
    --ft_seed 12345 \
    --pedestal_voltage 1.5 \
    --output_file output.hdf5 \
    --data_dir /path/to/output
```

### Standard neutrino simulation (thermal noise)

```bash
python simulate.py \
    --station_id 23 \
    --energy 1e18 \
    --n_events 1000 \
    --detector_file /path/to/detector.json.xz \
    --output_file output.hdf5 \
    --data_dir /path/to/output
```

## CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `RNO_config.yaml` | NuRadioMC YAML config (can include `fiducial_volume` section) |
| `--station_id` | required | Station ID |
| `--energy` | 1e18 | Neutrino energy in eV |
| `--n_events` | 1000 | Number of events to simulate |
| `--detector_file` | None (MongoDB) | Detector description file |
| `--ft_noise_dir` | None | FT data directory (enables measured noise mode) |
| `--ft_seed` | None | Reproducibility seed for FT noise selection |
| `--pedestal_voltage` | 1.5 | ADC pedestal in V |
| `--fiducial_rmax` | None | If set, uses CR proxy volume (0-1m depth) |
| `--min_zenith` | 0.0 | Min zenith angle in degrees |
| `--max_zenith` | 60.0 | Max zenith angle in degrees |
| `--nur_output` | False | Also write NUR files |
| `--index` | 0 | Chunk index for parallel runs |

## Output

- **HDF5**: standard NuRadioMC output with triggered event data
- **NUR** (optional): NuRadioReco event files for waveform inspection
- **Ledger CSV**: one row per input event with columns: `event_group_id`, `zenith_deg`, `azimuth_deg`, `energy_eV`, `flavor`, `status` (triggered/trigger_failed/efield_cut), `max_amp_ch{0-3}_mV`

## Framework changes on this branch

This branch (`cr_proxy_with_FT_noise`) adds to NuRadioMC:

1. **`noiseImporter`**: trigger copy injection, two-stage mode, explicit file list parameter
2. **`analogToDigitalConverter`**: pedestal voltage support (`set_pedestal_voltage()`)
3. **`channelSinewaveSubtraction`**: None check fix for `save_filtered_freqs=False`
4. **`readRNOGDataMattak`**: ValueError catch for corrupt ROOT files
5. **`efieldToVoltageConverterPerEfield`**: pre/post pulse zero-padding for linear convolution
