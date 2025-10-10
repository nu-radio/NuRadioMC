# Interferometric Direction Reconstruction

This module performs directional reconstruction of radio signals by fitting time delays between channels to pre-calculated time delay maps.

## Files in this Directory

- **`interferometric_reco_runner.py`**: Command-line script for running reconstructions
- **`example_config.yaml`**: Example configuration file with all available options
- **`INTERFEROMETRIC_RECONSTRUCTION_README.md`**: This documentation file

## Overview

The interferometric direction reconstruction works by:
1. Computing cross-correlations between pairs of antenna channels
2. Comparing measured time delays to pre-calculated time delay maps
3. Finding the source location that maximizes the correlation

The core reconstruction functionality is implemented in the NuRadioReco module:
`NuRadioReco.modules.interferometricDirectionReconstruction`

## Table of Contents
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Configuration File](#configuration-file)
- [Usage Examples](#usage-examples)
- [Output Formats](#output-formats)
- [Coordinate Systems](#coordinate-systems)
- [Time Delay Tables](#time-delay-tables)
- [Advanced Options](#advanced-options)
- [Troubleshooting](#troubleshooting)

---

## Requirements

### Pre-calculated Time Delay Tables
You **must** have pre-calculated time delay tables for your station and channels. These are `.npz` files containing travel time information for each channel.

**Location:** Tables should be in: `{time_delay_tables}/station{station_id}/ch{channel}_rz_table_rel_ant.npz`

**Format:** Each table is a 2D grid of travel times as a function of (r, z) coordinates **relative to that antenna**:
- **r**: perpendicular distance from the antenna
- **z**: vertical offset from the antenna (positive above, negative below)

### Python Dependencies
- NuRadioReco
- numpy
- scipy
- matplotlib
- h5py
- pyyaml

---

## Quick Start

### 1. Create a Configuration File

Create a YAML file (e.g., `reco_config.yaml`) with your reconstruction parameters:

```yaml
# Coordinate system: "cylindrical" or "spherical"
coord_system: "cylindrical"

# Reconstruction type (depends on coord_system):
#   For cylindrical: "phiz" (azimuth + depth) or "rhoz" (radius + depth)
#   For spherical: not used (automatically uses azimuth + zenith)
rec_type: "phiz"

# Channels to use for reconstruction (list of antenna channel IDs)
channels: [0, 1, 2, 3]

# Search grid limits: [coord0_min, coord0_max, coord1_min, coord1_max]
# Units depend on coordinate system:
#   phiz: [phi_min(deg), phi_max(deg), z_min(m), z_max(m)]
#   rhoz: [rho_min(m), rho_max(m), z_min(m), z_max(m)]
#   spherical: [phi_min(deg), phi_max(deg), theta_min(deg), theta_max(deg)]
limits: [0, 360, -200, 0]

# Step sizes for grid: [coord0_step, coord1_step]
# Same units as limits
step_sizes: [0.5, 0.5]

# Fixed coordinate (the coordinate not being reconstructed)
#   For phiz: fixed rho (radius) in meters
#   For rhoz: fixed phi (azimuth) in degrees
#   For spherical: fixed r (radial distance) in meters
fixed_coord: 125.0

# Path to time delay tables directory
time_delay_tables: "/path/to/time_delay_tables/"

# Path to save correlation map plots (if requested)
save_plots_to: "./figures/correlation_maps/"

# Detector configuration (for real data)
detector_json: "RNO_G/RNO_season_2024.json"

# Signal processing options
apply_cable_delays: true          # Apply cable delay corrections
apply_upsampling: true           # Upsample waveforms to 5 GHz
apply_bandpass: false             # Apply bandpass filter
apply_cw_removal: false           # Remove CW interference
apply_waveform_scaling: false     # Normalize waveforms
apply_hann_window: false          # Apply Hann window to correlations
use_hilbert_envelope: false       # Use Hilbert envelope for correlations

# Alternate reconstruction options
find_alternate_reco: false        # Find alternate reconstruction coordinates
alternate_exclude_radius_deg: 5.0 # Exclusion radius around primary maximum (degrees)

# Plotting options
create_minimaps: false            # Create minimap insets in correlation plots

# CW removal parameters (if apply_cw_removal: true)
cw_freq_band: [0.1, 0.7]         # Frequency band in GHz
cw_peak_prominence: 4.0          # Peak prominence threshold
```

### 2. Run Reconstruction

**Basic usage:**
```bash
python interferometricDirectionReconstruction.py \
    --config reco_config.yaml \
    --inputfile /path/to/data.root \
    --outputfile /path/to/output.h5
```

**Process specific events:**
```bash
python interferometricDirectionReconstruction.py \
    --config reco_config.yaml \
    --inputfile /path/to/data.root \
    --events 100 200 300 \
    --outputfile output.h5 \
    --save_plots \
    --verbose
```

**Multiple input files (same station):**
```bash
python interferometricDirectionReconstruction.py \
    --config reco_config.yaml \
    --inputfile file1.root file2.root file3.root \
    --outputfile merged_output.h5
```

---

## Configuration File

### Required Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `coord_system` | string | Coordinate system: "cylindrical" or "spherical" | `"cylindrical"` |
| `rec_type` | string | Type of reconstruction (only for cylindrical) | `"phiz"` |
| `channels` | list[int] | Antenna channels to use | `[0, 1, 2, 3]` |
| `limits` | list[float] | Search grid bounds [min0, max0, min1, max1] | `[0, 360, -200, 0]` |
| `step_sizes` | list[float] | Grid step sizes [step0, step1] | `[5, 5]` |
| `fixed_coord` | float | Value of fixed coordinate | `125.0` |
| `time_delay_tables` | string | Path to time delay tables directory | `"/path/to/tables/"` |
| `station_id` | int | Station ID for processing | `21` |

### Reconstruction Types

| `coord_system` | `rec_type` | What it reconstructs | Units |
|----------------|------------|---------------------|-------|
| `cylindrical` | `phiz` | Azimuth (φ) and depth (z), with fixed radius (ρ) | φ: degrees, z: meters |
| `cylindrical` | `rhoz` | Radius (ρ) and depth (z), with fixed azimuth (φ) | ρ: meters, z: meters |
| `spherical` | N/A | Azimuth (φ) and zenith (θ), with fixed radius (r) | φ, θ: degrees |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `apply_cable_delays` | bool | `true` | Apply cable delay corrections |
| `apply_upsampling` | bool | `false` | Upsample to 5 GHz |
| `apply_bandpass` | bool | `false` | Apply 100-600 MHz bandpass filter |
| `apply_cw_removal` | bool | `false` | Remove CW interference |
| `apply_waveform_scaling` | bool | `false` | Normalize waveforms (not recommended) |
| `apply_hann_window` | bool | `false` | Apply Hann window to correlations |
| `use_hilbert_envelope` | bool | `false` | Use Hilbert envelope for correlations |
| `find_alternate_reco` | bool | `false` | Find alternate reconstruction coordinates |
| `alternate_exclude_radius_deg` | float | `5.0` | Exclusion radius around primary maximum (degrees) |
| `create_minimaps` | bool | `false` | Create minimap insets in correlation plots |
| `station_id` | int | None | Station ID for processing and cache organization |
| `cw_freq_band` | list[float] | `[0.1, 0.7]` | CW removal frequency band (GHz) |
| `cw_peak_prominence` | float | `4.0` | CW peak detection threshold |
| `detector_json` | string | `"RNO_G/RNO_season_2024.json"` | Detector configuration file |
| `save_plots_to` | string | `"./"` | Directory for correlation map plots |

---

## Usage Examples

### Example 1: Azimuth Reconstruction (phiz)

Reconstruct azimuth and depth with fixed radius of 125m:

```yaml
# config_phiz.yaml
coord_system: "cylindrical"
rec_type: "phiz"
channels: [0, 1, 2, 3]
limits: [0, 360, -200, 0]        # φ: 0-360°, z: 0 to -200m
step_sizes: [0.5, 0.5]           # 0.5° in φ, 0.5m in z
fixed_coord: 125.0               # ρ = 125m
time_delay_tables: "/path/to/tables/"
save_plots_to: "./phiz_maps/"
apply_cable_delays: true
```

```bash
python interferometricDirectionReconstruction.py \
    --config config_phiz.yaml \
    --inputfile station23_run1234.root \
    --outputfile azimuth_reco.h5
```

### Example 2: Zenith Reconstruction (rhoz)

Reconstruct radius and depth with fixed azimuth:

```yaml
# config_rhoz.yaml
coord_system: "cylindrical"
rec_type: "rhoz"
channels: [0, 1, 2, 3]
limits: [0, 200, -200, 0]        # ρ: 0-200m, z: 0 to -200m
step_sizes: [0.5, 0.5]           # 0.5m in both
fixed_coord: 180.0               # φ = 180° (west)
time_delay_tables: "/path/to/tables/"
apply_cable_delays: true
```

### Example 3: Full Spherical

Reconstruct both azimuth and zenith with fixed distance:

```yaml
# config_spherical.yaml
coord_system: "spherical"
# rec_type not needed for spherical
channels: [0, 1, 2, 3]
limits: [0, 360, 0, 180]         # φ: 0-360°, θ: 0° (directly up) to 180° (directly down)
step_sizes: [0.5, 0.2]           # 0.5° in φ, 0.2° in θ
fixed_coord: 150.0               # r = 150m
time_delay_tables: "/path/to/tables/"
apply_cable_delays: true
```

### Example 4: With Signal Processing

Apply upsampling and CW removal:

```yaml
coord_system: "cylindrical"
rec_type: "phiz"
channels: [0, 1, 2, 3]
limits: [0, 360, -200, 0]
step_sizes: [5, 5]
fixed_coord: 125.0
time_delay_tables: "/path/to/tables/"
apply_cable_delays: true
apply_upsampling: true           # Upsample to 5 GHz
apply_cw_removal: true           # Remove CW interference
apply_hann_window: false         # Apply Hann window to correlations
use_hilbert_envelope: false      # Use Hilbert envelope for correlations
find_alternate_reco: true        # Find alternate reconstruction coordinates
alternate_exclude_radius_deg: 5.0 # Exclusion radius around primary (degrees)
create_minimaps: true            # Create minimap insets in plots
cw_freq_band: [0.1, 0.7]
cw_peak_prominence: 4.0
```

---

## Output Formats

### HDF5 Output (`.h5`)

Structured table with reconstruction results:

```python
import h5py
import pandas as pd

# Read HDF5 file
with h5py.File('output.h5', 'r') as f:
    data = f['reconstruction'][:]
    config = dict(f.attrs)  # Configuration stored as attributes

df = pd.DataFrame(data)
print(df.columns)
# Columns depend on rec_type:
# - All: runNum, eventNum, maxCorr, surfCorr
# - phiz: phi, z, phi_alt, z_alt (if alternate reconstruction enabled)
# - rhoz: rho, z, rho_alt, z_alt (if alternate reconstruction enabled)
# - phitheta: phi, theta, phi_alt, theta_alt (if alternate reconstruction enabled)
```

**Column Descriptions:**
- `runNum`: Run number
- `eventNum`: Event ID
- `maxCorr`: Maximum correlation value (0-1, higher is better)
- `surfCorr`: Maximum correlation in top 10m (surface correlation)
- `phi`: Reconstructed azimuth in degrees (for phiz, phitheta)
- `phi_alt`: Alternate reconstructed azimuth in degrees (if find_alternate_reco enabled)
- `theta`: Reconstructed zenith in degrees (for spherical)
- `theta_alt`: Alternate reconstructed zenith in degrees (if find_alternate_reco enabled)
- `rho`: Reconstructed perpendicular distance in meters (for rhoz)
- `rho_alt`: Alternate reconstructed perpendicular distance in meters (if find_alternate_reco enabled)
- `z`: Reconstructed depth in meters (for phiz, rhoz)
- `z_alt`: Alternate reconstructed depth in meters (if find_alternate_reco enabled)

### NUR Output (`.nur`)

NuRadioReco event format with reconstruction parameters stored in station parameters:

```python
from NuRadioReco.modules.io.eventReader import eventReader

reader = eventReader()
reader.begin('output.nur', read_detector=True)

for event in reader.run():
    station = event.get_station()
    
    max_corr = station.get_parameter(stnp.rec_max_correlation)
    surf_corr = station.get_parameter(stnp.rec_surf_corr)
    coord0 = station.get_parameter(stnp.rec_coord0)
    coord1 = station.get_parameter(stnp.rec_coord1)
    
    # For phiz: coord0 = phi, coord1 = z
    # For rhoz: coord0 = rho, coord1 = z
    # For spherical: coord0 = phi, coord1 = theta
```

---

## Coordinate Systems

Coordinates are relative to the power string position at the surface.

### Cylindrical Coordinates

#### phiz Mode
- **Reconstructed:** φ (azimuth), z (depth)
- **Fixed:** ρ (perpendicular distance)
- **Use case:** When you know the approximate distance and want to find direction and depth

```
φ: Azimuth angle (0° = East, 90° = North, 180° = West, 270° = South)
z: Depth (negative values represent going under the surface, positive above)
ρ: Perpendicular distance from power string
```

#### rhoz Mode
- **Reconstructed:** ρ (perpendicular distance), z (depth)
- **Fixed:** φ (azimuth)
- **Use case:** When you know the direction and want to find distance and depth (zenith angle reconstruction)

```
ρ: Perpendicular distance from power string
z: Depth (negative values represent going under the surface, positive above)
φ: Fixed azimuth direction (0° = East, 90° = North, 180° = West, 270° = South)
```

### Spherical Coordinates

**For spherical coordinate system (no rec_type needed)**
- **Reconstructed:** φ (azimuth), θ (zenith angle)
- **Fixed:** r (radial distance)
- **Use case:** Full directional reconstruction with known distance

```
φ: Azimuth angle (0° = East, 90° = North, 180° = West, 270° = South)
θ: Zenith angle (0° = up, 90° = perpendicular, 180° = down)
r: Radial distance from power string at surface of the ice.
```

---

## Time Delay Tables

### Required Directory Structure

```
time_delay_tables/
└── station23/
    ├── ch0_rz_table_rel_ant.npz
    ├── ch1_rz_table_rel_ant.npz
    ├── ch2_rz_table_rel_ant.npz
    ├── ch3_rz_table_rel_ant.npz
    └── ...
```

### Table Format

Each `.npz` file must contain:
- `data`: 2D array of travel times (nanoseconds) with shape: (len(r_range_vals), len(z_range_vals))
- `r_range_vals`: 1D array of **perpendicular distances** from the antenna (meters)
- `z_range_vals`: 1D array of **vertical offsets** from the antenna (meters)

**Important coordinate details:**
- **r (perpendicular)**: Distance from antenna in the perpendicular plane
  - Example: r=100m means 100 meters away perpendicularly from the antenna
- **z (vertical)**: **Vertical offset from the antenna position**
  - Example: z=-50m means 50 meters below the antenna
  - z=0 is at the antenna depth, z<0 is below antenna, z>0 is above antenna

**Table interpretation:**
- `data[i, j]` = travel time from source (at offset `(r[i], 0, z[j])` from the antenna) to the antenna
- The source is placed at perpendicular distance `r[i]` and vertical offset `z[j]` from the antenna
- During reconstruction, absolute source positions are converted to antenna-relative (r, z) before lookup

### Generating Time Delay Tables

Time delay tables can be generated using ray-tracing codes that account for:
- Ice refractivity profile (e.g., exponential density model for Greenland)
- Signal propagation paths (direct, refracted, and reflected rays)
- Antenna positions in station coordinate system
- The fastest arrival time is stored (minimum across all ray types)

**Example generation code:**
```python
# For each grid point (r, z):
src_position = antenna_position + [r, 0, 0]  # Offset perpendicularly
src_position[2] = z                          # Set to absolute depth
travel_time = raytracer.get_travel_time(src_position, antenna_position)
```

*Note: Full table generation code is in the `tools/reconstruction/tables/` directory.*

---

## Advanced Options

### Alternate Reconstruction

The module can find alternate reconstruction coordinates by excluding regions around the primary maximum:

```yaml
find_alternate_reco: true        # Enable alternate coordinate finding
alternate_exclude_radius_deg: 5.0 # Exclusion radius around primary maximum
```

When enabled, alternate coordinates are saved in the HDF5 output as `phi_alt`, `z_alt`, etc., and displayed in plots.

### Enhanced Plotting

Correlation maps can include enhanced visualizations:

```yaml
create_minimaps: true            # Add minimap insets showing zoomed regions
save_plots_to: "./figures/"     # Directory for correlation plots
```

When `save_plots` is used, plots will show:
- Primary and alternate reconstruction points
- Exclusion zones (when alternate reconstruction is enabled)
- Minimap insets for detailed views around correlation peaks

### Signal Processing Options

Additional correlation analysis options:

```yaml
apply_hann_window: true          # Apply Hann window to reduce spectral leakage
use_hilbert_envelope: true       # Use envelope correlation for better SNR
```

### Caching

The module automatically caches delay matrices to speed up repeated runs:
- **Cache location:** `~/.cache/nuradio_delay_matrices/`
- **Cache key:** Generated from station, channels, grid, and cable delays
- **Behavior:** Cache is automatically loaded if available

To force regeneration, delete the cache directory.

### Parallel Processing

For processing many files, use job arrays:

```bash
#!/bin/bash
#SBATCH --array=0-99

FILES=(run_*.root)
FILE=${FILES[$SLURM_ARRAY_TASK_ID]}

python interferometricDirectionReconstruction.py \
    --config config.yaml \
    --inputfile $FILE \
    --outputfile output_${SLURM_ARRAY_TASK_ID}.h5
```

## Troubleshooting

### "No results to save"
**Cause:** No events matched the selection criteria or all events failed processing.

**Solutions:**
- Check that event IDs exist in input file
- For `.nur` files, use run numbers (not event IDs)
- Remove `--events` flag to process all events

### No clear maximum correlation
**Cause:** Poor signal quality, wrong channels, or incorrect grid.

**Solutions:**
- Check SNR of event (use quality cuts)
- Verify channel selection
- Adjust search grid (`limits`, `step_sizes`)
- Try different `fixed_coord` value
- Consider using `use_hilbert_envelope: true` for better correlation

### Slow performance
**Cause:** Large search grids or missing cache.

**Solutions:**
- Use coarser grid (`step_sizes`)
- Let cache build (first run is slower)
- Process fewer events at once
- Check cache is being used: look for "Loaded delay matrices from cache" message

---

## Command-Line Arguments Reference

```bash
python interferometricDirectionReconstruction.py [OPTIONS]

Required:
  --config CONFIG            Path to YAML configuration file
  --inputfile FILE [FILE ...]  Input data file(s) (.root or .nur)

Optional:
  --outputfile FILE          Output file (.h5 for HDF5, .nur for NuRadioReco)
  --events N [N ...]         Specific event IDs/indices to process
  --save_plots               Save correlation map plots for processed events
  --verbose                  Print reconstruction results for each event
```

---

## Contact & Support

For questions or issues:
- Review examples in this README
- Check NuRadioReco documentation
- Contact Bryan Hendricks (blh5615@psu.edu)

---

## Version History

- **v1.0** (2025-10): Initial version
