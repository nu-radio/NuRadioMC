# Pedestal extraction: per-channel ADC clip thresholds

The RADIANT 12-bit ADC digitizes over a 0-2.5 V range (4096 counts). The pedestal (DC baseline) sits at approximately 1.5 V, not the 1.25 V midpoint. This makes the effective dynamic range asymmetric: signals can swing further negative (~1.5 V headroom) than positive (~1.0 V headroom) before clipping.

`simulate.py` applies asymmetric clipping via `--pedestal_voltage`.

## Files

| File | Description |
|------|-------------|
| `pedestal_analysis.py` | Extracts per-channel pedestal voltages from pedestal.root files |
| `clip_thresholds_station23_2022.yaml` | Output: per-channel clip thresholds from 1,124 runs (2022 only) |
| `pedestal_analysis_results.npz` | Raw per-run, per-channel pedestal voltages for further analysis |

## Method

Each RNO-G run includes a pedestal measurement that records the ADC pedestal distribution (4096 bins) for all 24 channels. The script:

1. Crawls a data directory for `run*/pedestal.root` files
2. Extracts the mean pedestal (in ADC counts, converted to mV) per channel per run
3. Optionally filters by year using the UTC timestamp in the ROOT file
4. Computes the median pedestal across all qualifying runs for each channel
5. Derives asymmetric clip thresholds: `clip_negative = -median`, `clip_positive = 2500 - median` (in mV)

## 2022 clip thresholds

Derived from 1,124 runs for station 23 between 2022-06-27 and 2022-10-01. Sample values:

| Channel | Pedestal (mV) | Clip- (mV) | Clip+ (mV) |
|---------|--------------|------------|------------|
| ch0 | 1416 | -1416 | +1084 |
| ch1 | 1467 | -1467 | +1033 |
| ch3 | 1593 | -1593 | +907 |
| ch9 | 1560 | -1560 | +940 |

The full set is in `clip_thresholds_station23_2022.yaml`. An earlier version used all 6,849 runs spanning 2021-2024, but seasonal pedestal drift of up to 108 mV on some channels makes year-specific thresholds more accurate.

## Usage

```bash
# Extract thresholds for 2022 (requires pedestal data and SLURM)
python pedestal_analysis.py \
    --data_dir /path/to/satellite/station23/ \
    --year 2022 \
    --outdir .

# Use in the CR proxy simulation
python ../simulate.py \
    ... \
    --pedestal_voltage 1.5
```

The script uses `joblib` for parallel processing of ROOT files. Run with `--cpus-per-task=20` on SLURM for the full 6,849-file dataset.

## Known limitations

- **Station 23, 2022 only.** The included `clip_thresholds_station23_2022.yaml` was derived from station 23 runs between 2022-06-27 and 2022-10-01. Other stations require separate extraction.

- **Year-specific thresholds required.** Seasonal pedestal drift of up to 108 mV on some channels means all-year averages are not reliable. Extract thresholds per year (or per season) using the `--year` flag.

- **Single median per channel.** The extraction computes one median pedestal per channel across all qualifying runs. Intra-run variation and run-to-run drift within a season are not captured. For most channels the variation is small (<10 mV), but a few channels show larger excursions.

- **No database integration.** The RNO-G MongoDB does not currently store `adc_pedestal_voltage`. Pedestals must be set at runtime via `--pedestal_voltage` (single value) or `set_pedestal_voltage(dict)` (per-channel). Adding this field to the DB would allow automatic pedestal loading.

- **Requires satellite pedestal data.** The extraction reads `pedestal.root` files from the satellite data mirror. These files may not be available for all stations or all runs.
