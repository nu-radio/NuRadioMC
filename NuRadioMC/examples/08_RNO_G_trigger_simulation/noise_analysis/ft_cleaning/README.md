# FT noise cleaning

Forced-trigger (FT) events serve as the noise pool for simulations using measured noise (see `../simulate.py` with `--ft_noise_dir`). Non-thermal FT events need to be excluded so only thermal noise is injected.

This directory contains scripts to identify non-thermal FT events and produce a clean mask (`--ft_clean_mask` in `simulate.py`). All numbers below are from station 23, 2022 FT data. Results will vary for other stations and time periods.

## Pipeline

```
extract_ft_rms.py       (slow: reads ROOT files, saves per-channel RMS as NPZ)
        |
        v
generate_clean_mask.py  (fast: reads RMS NPZ, applies 4-sigma cut, saves mask NPZ)
validate_threshold.py   (fast: reads RMS NPZ, sweeps thresholds, plots convergence)
```

## Files

| File | Description |
|------|-------------|
| `extract_ft_rms.py` | Extracts per-channel noise RMS from FT ROOT files |
| `generate_clean_mask.py` | Applies per-channel MAD-sigma threshold to RMS data, produces clean mask |
| `validate_threshold.py` | Sweeps cleaning thresholds and validates against simulated thermal reference |
| `clean_mask_station23.npz` | Output mask: 712,913 clean / 718,979 total events (99.16%) |
| `figures/ft_cleaning_threshold_convergence.png` | Cleaning threshold sweep + before/after z-score distributions |

## Data sources

**FT data:** 718,979 FORCE trigger events from station 23, all 2022 runs (1-1135). `extract_ft_rms.py` reads FT events from ROOT files via `readRNOGDataMattak`, filtering for FORCE triggers, and computes the per-channel waveform RMS after voltage calibration and median baseline correction. The output NPZ is consumed by all downstream scripts.

## Cleaning method

The initial approach used a composite flag based on 5 derived features (`chAvgSNR`, `maxAmplitude`, `impulsivity`, `coherentSNR`, `outlier_score`) plus a multi-channel RMS criterion, drawn from the full 372-feature set produced by the feature extraction pipeline. This flagged 4,416 events (0.61%).

Of those, 774 were flagged by the composite criteria but not by a per-channel RMS cut. These events had low z-scores (median 1.73, max 3.78) and looked like normal thermal noise on inspection, indicating the composite flag was likely producing false positives, so the final method uses only per-channel `noise_RMS`. For each of the 15 deep channels (ch0-11, 21-23), compute the median and MAD (median absolute deviation) of noise_RMS across all FT events. Convert MAD to Gaussian-equivalent sigma: `sigma = 1.4826 * MAD`. Flag any event where `(noise_RMS - median) / sigma > 4` on any channel.

## Threshold calibration

The 4-sigma threshold was calibrated by sweeping from 2.5 to 8 sigma and comparing the post-cut noise_RMS distribution shape (excess kurtosis, skewness) against a reference of 1000 pure simulated thermal noise events for station 23 (generated using per-channel calibrated effective temperatures and signal chain responses from 2023 data; contact Ruben for the generation script and inputs). The goal is to find the threshold where the post-cut FT distribution matches the simulated thermal reference (kurtosis < 0.28, skewness < 0.22).

The figure shows kurtosis for helper string channels (ch9-11, 21-23) in the top panel because they have the highest contamination rates and need the tightest cuts to reach the thermal reference. PA channels reach it at looser thresholds and are shown in the bottom-left panel.

![FT cleaning threshold convergence](figures/ft_cleaning_threshold_convergence.png)

| Threshold | Events removed | Helper max kurtosis | Converged to thermal? |
|-----------|---------------|--------------------|-----------------------|
| 2.5 sigma | 11.6% | 0.11 | Over-cut (negative skewness) |
| 3.0 sigma | 3.7% | 0.17 | Over-cut |
| 3.5 sigma | 1.4% | 0.23 | Marginal |
| **4.0 sigma** | **0.84%** | **0.28** | **Yes** |
| 4.5 sigma | 0.72% | 0.32 | Residual tails |
| 5.0 sigma | 0.69% | 0.33 | Residual tails |
| 8.0 sigma | 0.63% | 0.89 | Non-thermal |
| No cut | 0% | 463 | Non-thermal |

At 4 sigma, the post-cut kurtosis matches the simulated thermal reference relatively well. Tighter cuts over-sculpt (skewness goes negative). Looser cuts leave non-Gaussian tails.

## Clean mask format

The output `clean_mask_station23.npz` contains:

| Field | Type | Description |
|-------|------|-------------|
| `runNum` | int32 | Run number per event |
| `eventNum` | int32 | Event number per event |
| `is_clean` | int8 | 1 = clean (thermal), 0 = flagged (non-thermal) |

To exclude non-thermal events from simulations, pass the mask via `--ft_clean_mask` in `simulate.py`.

## Results

| Metric | Value |
|--------|-------|
| Total FT events (FORCE only, 2022) | 718,979 |
| Per-channel 4-sigma flagged | 6,066 (0.84%) |
| Clean events | 712,913 (99.16%) |

Per-channel breakdown (number of events exceeding 4 sigma on each channel). A single event can be flagged on multiple channels, so these sum to more than 6,066:

| Channel | Role | Flagged |
|---------|------|---------|
| ch0 | PA VPOL | 3,640 |
| ch1 | PA VPOL | 2,204 |
| ch21 | Helper C HPOL | 4,469 |
| ch22 | Helper C VPOL | 4,366 |
| ch10 | Helper B VPOL | 1,426 |
| ch23 | Helper C VPOL | 1,669 |
| ch11 | Helper B HPOL | 782 |

## Usage

```bash
# Step 1: Extract per-channel RMS from ROOT files (slow, run once)
python extract_ft_rms.py \
    --ft_noise_dir /path/to/forced_triggers/station23 \
    --station_id 23

# Step 2: Generate the clean mask
python generate_clean_mask.py \
    --rms_npz ft_rms_station23.npz

# Validate the threshold choice (requires simulated thermal noise NUR)
python validate_threshold.py \
    --rms_npz ft_rms_station23.npz \
    --sim_nur /path/to/simulated_noise.nur \
    --output_dir figures/
```

## Known limitations

- **Station 23, 2022 only.** The included `clean_mask_station23.npz` covers station 23 FT data from 2022. Other stations and years require re-running `extract_ft_rms.py` and `generate_clean_mask.py` on their own FT data.

- **Simulated thermal reference from a different season.** The 4-sigma threshold was validated against simulated noise generated with effective temperatures from 2023 (from Ruben), while the FT data is from 2022. If the noise environment differs between seasons (which it probably does), the threshold may need recalibration with a matching reference.

- **Per-channel RMS only.** The cleaning criterion uses per-channel noise_RMS and does not flag events with contamination that preserves the per-channel RMS. In practice, the 4-sigma cut captures all visually identifiable non-thermal events in the station 23 dataset.
