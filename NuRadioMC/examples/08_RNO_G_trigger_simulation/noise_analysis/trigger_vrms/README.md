# Trigger-path Vrms extraction

FT noise is recorded through the readout signal chain (RADIANT). The trigger path (FLOWER) has a different signal chain after the 3 dB splitter (arXiv:2411.12922, Sec. 3.2), so the noise Vrms seen by the trigger differs from the readout Vrms. The simulation needs the trigger-path Vrms to set the trigger threshold properly.

`extract_trigger_vrms.py` measures the trigger-path Vrms by drawing FT events, applying the readout-to-trigger transfer function (from the detector description), and computing the per-channel RMS. The output YAML is passed to `simulate.py` via `--trigger_vrms`. Then, `triggerBoardResponse` uses the Vrms to select the VGA gain stage and digitize to 8-bit counts, and `highLowThreshold` converts the target trigger rate (1 Hz, ~3.76x the Vrms via `RNO_G_HighLow_Thresh`) into an ADC-count threshold.

## Files

| File | Description |
|------|-------------|
| `extract_trigger_vrms.py` | Extracts per-channel trigger Vrms and saves as YAML |
| `vrms_convergence_study.py` | Sweeps N to determine how many FT events are needed for stable Vrms |

## Usage

```bash
# Extract trigger Vrms (run once per station/detector/FT dataset)
python extract_trigger_vrms.py \
    --ft_noise_dir /path/to/forced_triggers/station23 \
    --station_id 23 \
    --n_events 100 \
    --output trigger_vrms_station23.yaml

# Use in simulation
python ../../simulate.py \
    --trigger_vrms trigger_vrms_station23.yaml \
    --ft_noise_dir /path/to/forced_triggers/station23 \
    ...
```

## Convergence study

`vrms_convergence_study.py` measures the Vrms at several N values with multiple random seeds to quantify how many FT events are needed for the per-channel Vrms to stabilize. Supports SLURM array parallelization via `--chunk_id` / `--n_chunks`.

```bash
# Serial
python vrms_convergence_study.py \
    --ft_noise_dir /path/to/forced_triggers/station23 \
    --n_repeats 10 --outdir .

# Parallel (SLURM array)
#SBATCH --array=0-9
python vrms_convergence_study.py \
    --ft_noise_dir /path/to/forced_triggers/station23 \
    --n_repeats 10 --outdir . \
    --chunk_id $SLURM_ARRAY_TASK_ID --n_chunks 10

# Merge chunks
python vrms_convergence_study.py --merge --outdir .
```

### Results (station 23, 2022 FT data, 10 repeats per N)

| N | ch0 (mV) | ch1 (mV) | ch2 (mV) | ch3 (mV) | Worst-channel spread |
|---|----------|----------|----------|----------|---------------------|
| 10 | 4.33 +/- 0.09 | 5.19 +/- 0.15 | 4.25 +/- 0.12 | 2.98 +/- 0.07 | 2.9% |
| 25 | 4.31 +/- 0.07 | 5.10 +/- 0.07 | 4.26 +/- 0.09 | 2.98 +/- 0.03 | 2.1% |
| 50 | 4.34 +/- 0.08 | 5.13 +/- 0.08 | 4.27 +/- 0.09 | 3.02 +/- 0.04 | 2.2% |
| **100** | **4.32 +/- 0.04** | **5.10 +/- 0.05** | **4.21 +/- 0.04** | **3.01 +/- 0.02** | **1.0%** |
| 200 | 4.32 +/- 0.02 | 5.12 +/- 0.03 | 4.21 +/- 0.03 | 3.01 +/- 0.02 | 0.8% |
| 500 | 4.32 +/- 0.01 | 5.11 +/- 0.02 | 4.23 +/- 0.02 | 3.01 +/- 0.01 | 0.5% |

The means converge by N=25. The spread drops below 1% at N=100, which is the default for `extract_trigger_vrms.py`.

## Known limitations

- The readout-to-trigger transfer function comes from the detector description. If the detector description is inaccurate, the Vrms will be too.
- The VGA gain selection in `triggerBoardResponse` does not match the real FLOWER hardware, so the relationship between the measured Vrms and the actual trigger behavior is approximate.
  - The real FLOWER board has a 14-stage VGA that auto-selects gain so noise fills ~5 ADC counts on its 8-bit digitizer. `triggerBoardResponse` replicates this, but for the same input Vrms it picks a systematically lower gain stage than the real board (compared against gain codes from `aux/flower_gain_codes.0.txt`). The cause is unknown. This means the digitized amplitude scaling in the simulation differs from real hardware, so the ADC-count threshold doesn't map to the same physical voltage threshold as on the real board.
- Results are specific to the station, detector description, and FT data used. Re-extract when any of these change.
