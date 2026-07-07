# RNO-G feature extraction driver

Per-event scalar feature extraction for RNO-G. Reads data with
`dataProviderRNOG` (ROOT) or `dataProviderNuRadio` (NUR), runs
`NuRadioReco.modules.channelFeatureExtractor` per channel, then applies
RNO-G-specific aggregation (per-group coherent sums, channel-group
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
  (`snr`, `noise_rms`, `rpr`, `max_amplitude`, `impulsivity`,
  `kurtosis`, `entropy`, spectral descriptors, impulse-template
  correlations, etc.).
- `{feature}_avg_{pa,vpol,hpol,deep}` — mean of the per-channel value
  over each named antenna group. PA = ch0-3, VPOL = all string-deployed
  VPOLs (ch0,1,2,3,5,6,7,9,10,22,23), HPOL = all string-deployed HPOLs
  (ch4,8,11,21), deep = VPOL ∪ HPOL (15 in-ice channels, no surface
  LPDAs). Covers `snr`, `kurtosis`, `entropy`, `max_amplitude`,
  `impulsivity`, each spectral descriptor, and each impulse-template
  correlation.
- `coherent_{feature}_{pa,vpol,hpol,deep}` — features of the per-group
  coherent-sum trace (SNR, impulsivity, kurtosis, entropy, spectral
  descriptors, impulse-template correlations). The `hpol` group has
  only 4 channels and the `deep` group mixes polarizations, so their
  coherent alignment is less physically interpretable than `pa`/`vpol`
  but still useful as classifier features.
- `passed_hit_filter`, `n_coincident_pairs_{pa,deep}`,
  `n_high_hits_{pa,deep}` — only present when `hit_filter.enabled` is
  set in the config (see below).

## Optional: station hit filter

The driver can run `NuRadioReco.modules.RNO_G.stationHitFilter` per
event, emit its outputs as extra columns, and optionally skip events
that fail the filter. Enable via the config:

```yaml
hit_filter:
  enabled: true          # run the filter (default: false)
  require_pass: false    # drop events that don't pass (default: false)
  add_features: true     # write hit-filter columns to output (default: true)
  complete_time_check: true
  complete_hit_check: true
  log_summary: true      # log pass/fail summary at end of chunk
```

## Relation to the generic module

The per-channel physics lives in
`NuRadioReco/modules/channelFeatureExtractor.py`. This driver is the
RNO-G-specific layer: channel groupings, coherent-sum construction,
HDF5 column naming. Other experiments can write their own drivers
against the same module.

See also `NuRadioReco/examples/RNOG/interferometric_reco_ex/` for the
reco equivalent of the same module + RNO-G driver split.
