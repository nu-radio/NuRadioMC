# Reco plotting

Plots derived from a 3D-interferometric-reco merged HDF5 (and, for sim, the
combined event-variables H5 carrying truth columns). Each script is
independently runnable; `plot_orchestrator.py` reads a yaml and dispatches
the enabled subset.

## Scripts

| Script | Inputs | Outputs |
|---|---|---|
| `plot_reco_summary.py` | merged 3D reco H5 (`--input`) | per-quantity histograms (peak_z, peak_rho, peak_phi, peak_corr, peak_map_snr, max_corr, peak_zen, surf_zen variants) |
| `plot_sim_zenith_error.py` | combined event-variables H5 (`--input`, sim-only rows used) | reco-vs-truth zenith hist, abs-error hist, 2D scatter, truth-zenith hist |

## Orchestrator

`plot_orchestrator.py` dispatches the enabled subset from a yaml config:

```yaml
# reco_plotting.yaml
enabled:
  - reco_summary       # plot_reco_summary.py
  - sim_zenith_error   # plot_sim_zenith_error.py (skipped if --combined missing)
```

Standalone usage:

```bash
python plot_orchestrator.py \
  --config reco_plotting.yaml \
  --reco-merged path/to/merged_3d_reco.h5 \
  --combined  path/to/combined_event_variables.h5 \
  --output-dir plots/ \
  --label burn
```

CLI flags `--reco-merged`, `--combined`, `--output-dir`, `--label` are passed through to whichever scripts need them. Plots whose required inputs aren't supplied are skipped with a warning (so the burn-side rule can omit `--combined` and just get `reco_summary`).

## Adding a new plot

1. Drop a new `plot_X.py` here with `--input`, `--output-dir`, `--label` (or whatever fits).
2. Add an `if "X" in enabled:` block to `plot_orchestrator.py` that subprocess-runs your script.
3. Add `X` as an option to `reco_plotting.yaml.example`.
