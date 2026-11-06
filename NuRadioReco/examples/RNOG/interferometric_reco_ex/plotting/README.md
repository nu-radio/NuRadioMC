# Reco plotting

Raw-reconstruction plots are derived from a merged 3D-interferometric-reco
HDF5. PA-centred geometry and simulation-error plots consume saved columns
from the combined event-variables H5. Plotting never loads detector geometry
or recalculates PA-centred quantities.

Each script is independently runnable; `plot_orchestrator.py` reads a yaml
and dispatches the enabled subset.

## Scripts

| Script | Inputs | Outputs |
|---|---|---|
| `plot_reco_summary.py` | merged 3D reco H5 (`--input`) | raw-reco histograms (peak_z, peak_rho, peak_phi, peak_corr, peak_map_snr, max_corr, surface-region correlations) |
| `plot_pa_geometry.py` | combined event-variables H5 (`--input`, `--source`) | saved reconstructed PA-zenith distribution |
| `plot_sim_zenith_error.py` | combined event-variables H5 (`--input`, sim-only rows used) | saved reco-vs-truth zenith hist, abs-error hist, 2D comparison, truth-zenith hist |

## Orchestrator

`plot_orchestrator.py` dispatches the enabled subset from a yaml config:

```yaml
# reco_plotting.yaml
enabled:
  - reco_summary       # plot_reco_summary.py
  - pa_geometry        # plot_pa_geometry.py (requires --combined-source)
  - sim_zenith_error   # plot_sim_zenith_error.py (skipped if --combined missing)
```

Use ``--only NAME`` to run one enabled group. The option may be repeated;
Snakemake uses it so that each rule creates only the outputs it declares.

Standalone usage:

```bash
python plot_orchestrator.py \
  --config reco_plotting.yaml \
  --reco-merged path/to/merged_3d_reco.h5 \
  --combined  path/to/combined_event_variables.h5 \
  --combined-source burn \
  --output-dir plots/ \
  --label burn
```

CLI flags are passed through to whichever scripts need them. Plots whose
required inputs are not supplied are skipped with a warning.

## Adding a new plot

1. Drop a new `plot_X.py` here with `--input`, `--output-dir`, `--label` (or whatever fits).
2. Add an `if "X" in enabled:` block to `plot_orchestrator.py` that subprocess-runs your script.
3. Add `X` as an option to `reco_plotting.yaml.example`.
