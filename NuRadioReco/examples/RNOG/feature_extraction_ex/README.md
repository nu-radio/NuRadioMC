# RNO-G feature extraction driver

Per-event scalar feature extraction for RNO-G. Reads data with
`dataProviderRNOG` (ROOT) or `dataProviderNuRadio` (NUR), runs
`NuRadioReco.modules.channelFeatureExtractor` per channel, then applies
RNO-G-specific aggregation (PA/VPOL coherent sums, channel-group
averages) and writes a pandas DataFrame to HDF5.

## Files

- `feature_extraction.py` — CLI driver (subprocess-callable).

## Usage

```
python feature_extraction.py \
    --config features_config.yaml \
    -i /path/to/station21_run1000.root \
    --station_id 21 --run_chunk 0
```

Per-event rows are flattened into a single DataFrame with these
conventions:

- `ch{N}_{feature}` — per-channel values from the generic extractor
  (`snr`, `noise_rms`, `impulsivity`, `kurtosis`, spectral descriptors,
  impulse-template correlations, etc.).
- `{feature}_avg_{pa,vpol,deep}` — mean of the per-channel value over
  the PA (ch0-3), VPOL (ch0,1,2,3,5,6,7,9,10,22,23), and deep
  (VPOL + HPOL) groups.
- `pa_avg_snr`, `vpol_avg_snr` — mean of the per-channel SNR over PA
  and VPOL groups.
- `coherent_*` — features of the PA coherent sum.
- `coherent_*_vpol` — features of the VPOL coherent sum.

## Relation to the generic module

The per-channel physics lives in
`NuRadioReco/modules/channelFeatureExtractor.py`. This driver is the
RNO-G-specific layer: channel groupings, coherent-sum construction,
HDF5 column naming. Other experiments can write their own drivers
against the same module.

See also `NuRadioReco/examples/RNOG/interferometric_reco_ex/` for the
reco equivalent of the same module + RNO-G driver split.
