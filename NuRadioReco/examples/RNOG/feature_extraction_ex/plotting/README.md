# Feature plotting

Plots derived from a merged feature-extraction HDF5 (and, for joins, the merged
3D reco H5). Each script is independently runnable; `plot_orchestrator.py`
reads a yaml and dispatches the enabled subset.

## Scripts

| Script | Inputs | Outputs |
|---|---|---|
| `outlier_correlation_analysis.py` | merged features (`--features`) + merged 3D reco (`--reco`) | per-event z-score outlier score vs reco zenith / impulsivity (1D + 2D), summary text |

## Orchestrator

`plot_orchestrator.py`:

```yaml
# feature_plotting.yaml
enabled:
  - outlier_correlation
```

Standalone usage:

```bash
python plot_orchestrator.py \
  --config feature_plotting.yaml \
  --features path/to/merged_feature_output.h5 \
  --reco-merged path/to/merged_3d_reco.h5 \
  --output-dir plots/ \
  --label burn
```

Skipped (with warning) if the required inputs aren't passed.

## Adding a new plot

1. Drop a new `plot_X.py` here.
2. Add an `if "X" in enabled:` block to `plot_orchestrator.py`.
3. Add `X` to `feature_plotting.yaml.example`.
