# Interferometric Direction Reconstruction

This module performs directional reconstruction of radio signals by fitting time delays between channels to pre-calculated time delay maps.

## Files in this Directory

- **`interferometric_reco_example.py`**: Main reconstruction script with preprocessing options
- **`correlation_map_plotter.py`**: Standalone script for plotting saved correlation maps with comprehensive visualization options
- **`example_config.yaml`**: Example configuration file with all available options
- **`INTERFEROMETRIC_RECONSTRUCTION_README.md`**: This documentation file

## Supporting Modules

The reconstruction uses several NuRadioReco utility modules:

- **`NuRadioReco.utilities.interferometry_io_utilities`**: I/O functions for saving/loading results and correlation maps
  - `save_interferometric_results_hdf5()`: Save reconstruction results to HDF5
  - `save_interferometric_results_nur()`: Save results to NUR format
  - `save_correlation_map()`: Save correlation map data to pickle
  - `load_correlation_map()`: Load correlation map from pickle
  - `create_organized_paths()`: Create organized directory structure for outputs

- **`NuRadioReco.utilities.caching_utilities`**: Caching system for delay matrices
  - Automatically caches computed delay matrices in `~/.cache/nuradio_delay_matrices/`
  - Cache key based on station, channels, grid parameters, and interpolation method
  - Significantly speeds up repeated runs with same configuration

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
- [Visualizing Results](#visualizing-results)
- [Coordinate Systems](#coordinate-systems)
- [Time Delay Tables](#time-delay-tables)
- [Advanced Options](#advanced-options)
- [Troubleshooting](#troubleshooting)

---

## Requirements

### Pre-calculated Time Delay Tables
You **must** have pre-calculated time delay tables for your station and channels. These are `.npz` files containing travel time information for each channel.

**Location:** Tables should be in: `{time_delay_tables}/station{station_id}/ch{channel}_rz_table_rel_ant.npz`

**Format:** Each table is a 2D grid of travel times as a function of (r, z) coordinates:
- **r**: perpendicular distance from the antenna (relative)
- **z**: absolute depth coordinate (NOT relative to antenna)

### Python Dependencies
- NuRadioReco
- Mattak
- numpy
- scipy
- matplotlib
- h5py
- pyyaml

**Note:** The detector configuration is automatically loaded from the RNO-G MongoDB database. No detector JSON file is needed.

---

## Quick Start

### 0. Try the Example First

To quickly see the reconstruction in action, before even looking at setting it up yourself in the next steps, first download the provided calibration pulser root file at /data/reconstruction/validation_sets/cal_pulsers/station21/run476/combined.root and the pre-generated time tables (at least for station 21 since this pulser run is from it) from /data/reconstruction/travel_times_analytic/ from the Chicago server. Once you've downloaded the root file and time tables, you can run the example script like so (make sure if you download the tables to any location other than a tables/ dir in this directory to change the default time_delay_tables setting in the example_config.yaml to the new location; same for specifying the inputfile for combined.root below):

```bash
# Run reconstruction with example config and data
python interferometric_reco_example.py \
    --config example_config.yaml \
    --inputfile combined.root \
    --events 7 \
    --save_maps \
    --verbose

# Plot the correlation map
python correlation_map_plotter.py \
    --input results/station21/run476/corr_map_data/station21_run476_evt7_corrmap.pkl \
    --minimaps
```

This should reproduce the example figure shown at `/data/reconstruction/example_plots/station21_run476_evt7_corrmap.png` on the Chicago server. The "combined.root" file used to reproduce this is from a calibration pulsing run with pulser on helper string C, which is near Vpol channels 22 and 23, so we exclude those channels due to saturation.

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

# Station ID (required for processing and cache organization)
station_id: 21

# Path to time delay tables directory
time_delay_tables: "/path/to/time_delay_tables/"

# Interpolation method for time delay tables (optional - defaults to 'linear')
interp_method: "linear"          # Options: 'linear' or 'nearest'

# Output directory settings (optional - will use defaults if not specified)
save_results_to: "./results/"    # Base directory for all reconstruction data (default: "./results/")
                                 # Structure: {base}/station{ID}/run{NUM}/reco_data/ (results)
                                 #           {base}/station{ID}/run{NUM}/corr_map_data/ (correlation maps)

# Signal processing options
apply_cable_delay: true          # Apply cable delay correction (default: true)
                                 # WARNING: Only set to false if using preprocessed data 
                                 # that already has cable delays removed!
apply_upsampling: true           # Upsample waveforms to 5 GHz
apply_bandpass: false             # Apply bandpass filter
apply_cw_removal: false           # Remove CW interference
apply_hann_window: false          # Apply Hann window to correlations
use_hilbert_envelope: false       # Use Hilbert envelope for correlations

# Alternate reconstruction options
find_alternate_reco: false        # Find alternate reconstruction coordinates
alternate_exclude_radius_deg: 5.0 # Exclusion radius around primary maximum (degrees)

# CW removal parameters (if apply_cw_removal: true)
cw_freq_band: [0.1, 0.7]         # Frequency band in GHz
cw_peak_prominence: 4.0          # Peak prominence threshold
```

### 2. Run Reconstruction

**Basic usage:**
```bash
python interferometric_reco_example.py \
    --config reco_config.yaml \
    --inputfile /path/to/data.root \
    --output_type hdf5
```

**Process specific events:**
```bash
python interferometric_reco_example.py \
    --config reco_config.yaml \
    --inputfile /path/to/data.root \
    --events 7 10 15 \
    --save_maps \
    --verbose
```

**Multiple input files (same station):**
```bash
python interferometric_reco_example.py \
    --config reco_config.yaml \
    --inputfile file1.root file2.root file3.root \
    --output_type hdf5
```

### 3. Visualize Correlation Maps (Optional)

If you saved correlation maps with `--save_maps`, use the separate plotting script:

```bash
# Plot a single correlation map
python correlation_map_plotter.py --input results/station21/run476/corr_map_data/station21_run476_evt7_corrmap.pkl

# Plot all maps in a directory
python correlation_map_plotter.py --input results/station21/run476/corr_map_data/

# Plot with minimaps enabled
python correlation_map_plotter.py --input results/station21/run476/corr_map_data/ --minimaps

# Custom output directory
python correlation_map_plotter.py --input map.pkl --output custom_figures/

# Create comprehensive plot with waveforms and event information
python correlation_map_plotter.py \
    --input results/station21/run476/corr_map_data/station21_run476_evt7_corrmap.pkl \
    --comprehensive results/station21/run476/reco_data/station21_run476_reco_results.h5 \
    --minimaps
```

The `--comprehensive` option creates a multi-panel visualization including:
- Correlation map with reconstruction results
- Ray path visualization showing signal propagation
- Event information table with reconstruction parameters
- Waveform grid showing all channels used in reconstruction

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
| `apply_cable_delay` | bool | `true` | Apply cable delay correction. **WARNING:** Only disable if using preprocessed data with cable delays already removed! |
| `apply_upsampling` | bool | `false` | Upsample to 5 GHz |
| `apply_bandpass` | bool | `false` | Apply 100-600 MHz bandpass filter |
| `apply_cw_removal` | bool | `false` | Remove CW interference |
| `apply_hann_window` | bool | `false` | Apply Hann window to correlations |
| `use_hilbert_envelope` | bool | `false` | Use Hilbert envelope for correlations |
| `find_alternate_reco` | bool | `false` | Find alternate reconstruction coordinates |
| `alternate_exclude_radius_deg` | float | `5.0` | Exclusion radius around primary maximum (degrees) |
| `save_results_to` | string | `"./results/"` | Base directory for organized output structure |
| `cw_freq_band` | list[float] | `[0.1, 0.7]` | CW removal frequency band (GHz) |
| `cw_peak_prominence` | float | `4.0` | CW peak detection threshold |
| `interp_method` | string | `"linear"` | Interpolation method: 'linear' or 'nearest' |

---

## Usage Examples

### Example 1: Azimuth Reconstruction (phiz)

Reconstruct azimuth and depth with fixed perpendicular distance of 125m:

```yaml
# config_phiz.yaml
coord_system: "cylindrical"
rec_type: "phiz"
channels: [0, 1, 2, 3]
limits: [0, 360, -200, 0]        # φ: 0-360°, z: 0 to -200m
step_sizes: [0.5, 0.5]           # 0.5° in φ, 0.5m in z
fixed_coord: 125.0               # ρ = 125m
station_id: 23
time_delay_tables: "/path/to/tables/"
```

```bash
python interferometric_reco_example.py \
    --config config_phiz.yaml \
    --inputfile station23_run1234.root \
    --output_type hdf5
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
fixed_coord: 0.0                 # φ = 0° (east) - doesn't matter if only using antennas on power string
station_id: 23
time_delay_tables: "/path/to/tables/"
```

### Example 3: Full Spherical

Reconstruct both azimuth and zenith with fixed distance:

```yaml
# config_spherical.yaml
coord_system: "spherical"
# rec_type not needed for spherical
channels: [0, 1, 2, 3]
limits: [0, 360, 0, 180]         # φ: 0-360°, θ: 0° (up) to 180° (down)
step_sizes: [0.5, 0.2]           # 0.5° in φ, 0.2° in θ
fixed_coord: 50.0               # r = 50m
station_id: 23
time_delay_tables: "/path/to/tables/"
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
station_id: 23
time_delay_tables: "/path/to/tables/"

apply_upsampling: true           # Upsample to 5 GHz
apply_cw_removal: true           # Remove CW interference
apply_hann_window: false         # Apply Hann window to correlations
use_hilbert_envelope: false      # Use Hilbert envelope for correlations
find_alternate_reco: true        # Find alternate reconstruction coordinates
alternate_exclude_radius_deg: 5.0 # Exclusion radius around primary (degrees)
```

---

## Visualizing Results

### Basic Correlation Map Plots

The `correlation_map_plotter.py` script can create standalone correlation map visualizations:

```bash
# Plot a single correlation map
python correlation_map_plotter.py --input station21_run476_evt7_corrmap.pkl

# Plot all maps in a directory
python correlation_map_plotter.py --input results/station21/run476/corr_map_data/

# Enable minimap insets (zoomed views around peaks)
python correlation_map_plotter.py --input map.pkl --minimaps

# Plot specific pattern
python correlation_map_plotter.py --input results/ --pattern "*run476*" --minimaps
```

### Comprehensive Event Visualization

For detailed event analysis, use the `--comprehensive` flag to create multi-panel plots combining:
- **Correlation map** with reconstruction results including alternate coordinates if enabled during the reconstruction
- **Ray path visualization** showing signal propagation from vertex to antennas
- **Event information table** with truth values from simulations
- **Waveform grid** displaying all channels used in reconstruction

```bash
python correlation_map_plotter.py \
    --input results/station21/run476/corr_map_data/station21_run476_evt7_corrmap.pkl \
    --comprehensive results/station21/run476/reco_data/station21_run476_reco_results.h5 \
    --minimaps
```

**Requirements for comprehensive plots:**
- Correlation map pickle file (saved with `--save_maps`)
- Reconstruction results HDF5 file (contains event metadata and data filename)
- Access to original data file (for waveform extraction)

### Channel Pair Correlation Grids

If you saved pairwise correlation maps with `--save_pair_maps`, you can visualize all channel pair correlations:

```bash
python correlation_map_plotter.py \
    --pair-grid results/station21/run476/corr_map_data/pairwise_maps/
```

This creates a grid showing individual correlation maps for each antenna pair.

---

## Output Formats

The reconstruction automatically creates an organized directory structure:

```
results/
└── station{ID}/
    └── run{NUM}/
        ├── reco_data/
        │   └── station{ID}_run{NUM}_reco.h5  (or .nur)
        └── corr_map_data/  (if --save_maps used)
            ├── station{ID}_run{NUM}_evt{N}_corrmap.pkl
            └── ...
```

You can customize the base directory with the `save_results_to` config parameter.

### HDF5 Output (`.h5`)

Structured table with reconstruction results saved to `{base}/station{ID}/run{NUM}/reco_data/`:

```python
import h5py
import pandas as pd

# Read HDF5 file
with h5py.File('results/station21/run476/reco_data/station21_run476_reco.h5', 'r') as f:
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

### Correlation Map Data

When using `--save_maps`, correlation map data is saved to pickle files in `{base}/station{ID}/run{NUM}/corr_map_data/`:

Each `.pkl` file contains:
- `corr_matrix`: 2D correlation map
- `station_id`, `run_number`, `event_number`: Event identifiers
- `config`: Reconstruction configuration
- `coord_system`, `rec_type`: Coordinate system information
- `limits`: Grid boundaries
- `channels`: Channels used
- `fixed_coord`: Fixed coordinate value
- `coord0_alt`, `coord1_alt`: Alternate reconstruction coordinates (if enabled)
- `exclusion_bounds`: Exclusion zone information (if alternate reco enabled)

These files can be visualized using `correlation_map_plotter.py` (see next section).

---

## Visualizing Results

### Plotting Correlation Maps

Use the standalone `correlation_map_plotter.py` script to visualize saved correlation maps:

```bash
# Plot a single event
python correlation_map_plotter.py --input results/station21/run476/corr_map_data/station21_run476_evt7_corrmap.pkl

# Plot all events in a run
python correlation_map_plotter.py --input results/station21/run476/corr_map_data/

# Enable minimap insets for detailed views
python correlation_map_plotter.py --input results/station21/run476/corr_map_data/ --minimaps

# Save to custom directory
python correlation_map_plotter.py --input map.pkl --output my_figures/

# Process with pattern matching
python correlation_map_plotter.py --input results/station21/run476/corr_map_data/ --pattern "*evt7*"
```

**Output:** Plots are automatically saved to an organized structure:
- Default: `figures/station{ID}/run{NUM}/station{ID}_run{NUM}_evt{N}_corrmap.png`
- Custom: `{output_dir}/station{ID}/run{NUM}/station{ID}_run{NUM}_evt{N}_corrmap.png`

### Enhanced Plotting

The `correlation_map_plotter.py` script has options for enhanced visualizations with minimap insets.

**Note:** The `create_minimaps` parameter is used in the **plotter script**, not in the reconstruction config:

```bash
# Enable minimaps when plotting
python correlation_map_plotter.py --input corr_map_data/ --minimaps
```

It also allows for plotting an arbitrary number of additional points through the optional --extra-points argument. It expects 3 values: x value, y value, and label. This is most useful for adding a known location point to the map to compare to the reconstructed location. For example, let's say I know that my calibration pulser is at 182.1 degrees in azimuth and 87.5 m in depth with respect to the center of the phased array (the origin for the spherical case) and I want the known pulser location to be on the map. Then I would do:

```bash
# Enable extra point when plotting
python correlation_map_plotter.py --input corr_map_data/ --extra-points "182.1, 87.5, Pulser"
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
- `z_range_vals`: 1D array of **absolute depths** (meters)

**Important coordinate details:**
- **r (perpendicular)**: Distance from antenna in the perpendicular plane (relative to antenna)
  - Example: r=100m means 100 meters away perpendicularly from the antenna
- **z (depth)**: **Absolute depth coordinate** (NOT relative to antenna)
  - Example: z=-50m means 50 meters below the ice surface (absolute depth)
  - z=0 is at the ice surface, z<0 is below surface (in the ice)

**Table interpretation:**
- `data[i, j]` = travel time from source at position `(ant_x + r[i], ant_y, z[j])` to the antenna
- The source is placed at perpendicular distance `r[i]` from the antenna, at absolute depth `z[j]`
- The horizontal position is offset by `r` from the antenna, but the vertical position is **absolute**, not relative to antenna depth

### Generating Time Delay Tables

Time delay tables can be generated using ray-tracing codes that account for:
- Ice refractivity profile (e.g., exponential density model for Greenland)
- Signal propagation paths (direct, refracted, and reflected rays)
- Antenna positions in station coordinate system
- The fastest arrival time is stored (minimum across all ray types)

**Example generation code:**
```python
# For each grid point (r, z):
src_position = antenna_position + [r, 0, 0]  # Offset perpendicularly by r
src_position[2] = z                          # Set to absolute depth z (not relative!)
travel_time = raytracer.get_travel_time(src_position, antenna_position)
```

## Advanced Options

### Alternate Reconstruction

The module can find alternate reconstruction coordinates by excluding regions around the primary maximum:

```yaml
find_alternate_reco: true        # Enable alternate coordinate finding
alternate_exclude_radius_deg: 5.0 # Exclusion radius around primary maximum
```

When enabled, alternate coordinates are saved in the HDF5 output as `phi_alt`, `z_alt`, etc., and displayed in plots.

### Simulation Truth Fixed Coordinate (for NUR simulation files only)

When processing NUR simulation files, you can use the `--sim_truth_fixed_coord` flag to automatically set the fixed coordinate to the true simulation value for each event. This is useful for validation and debugging reconstruction performance:

```bash
python interferometric_reco_example.py \
    --config config.yaml \
    --inputfile simulation.nur \
    --sim_truth_fixed_coord \
    --verbose
```

**How it works:**
- For **phiz** reconstruction (fixed ρ): Uses true perpendicular distance from simulation vertex to PA center
- For **rhoz** reconstruction (fixed φ): Uses true azimuth from PA center to simulation vertex
- For **spherical** reconstruction (fixed r): Uses true radial distance from PA center to simulation vertex

**Important notes:**
- Only works with NUR simulation files that contain shower information (`event.get_sim_showers()`)
- The fixed coordinate is calculated **per-event** from the true interaction vertex
- Delay matrix caching is disabled when using this flag (since fixed_coord varies per event)
- This mode is primarily for validation - it tells you "if I knew the correct fixed coordinate, how well could I reconstruct the other coordinates?"

**Example use case:**
To test if your phiz reconstruction can accurately find azimuth and depth when given the true radius:

```bash
python interferometric_reco_example.py \
    --config phiz_config.yaml \
    --inputfile some_sim_file.nur \
    --sim_truth_fixed_coord \
    --save-maps \
    --verbose
```

This will reconstruct φ and z for each event using the true ρ value, allowing you to isolate reconstruction errors in the free parameters.

### Signal Processing Options

Additional correlation analysis options:

```yaml
apply_hann_window: true          # Apply Hann window to reduce spectral leakage
use_hilbert_envelope: true       # Use envelope correlation for better SNR
```

### Caching

The module uses `NuRadioReco.utilities.caching_utilities` to automatically cache delay matrices:
- **Cache location:** `~/.cache/nuradio_delay_matrices/`
- **Cache key:** Generated from station ID, channels, grid parameters, and interpolation method
- **Behavior:** Automatically loads from cache if available, significantly speeding up repeated runs
- **Cache management:** To force regeneration, delete the cache directory or specific cache files

The cache is particularly beneficial when:
- Running the same configuration multiple times
- Processing multiple events from the same station/run
- Testing different preprocessing options with the same reconstruction grid

**Bypassing the cache for debugging:**

If you need to force recalculation of delay matrices (e.g., after updating time delay tables or for debugging), use the `--ignore_cache` flag:

```bash
python interferometric_reco_example.py \
    --config config.yaml \
    --inputfile data.root \
    --ignore_cache
```

This will recompute delay matrices from scratch even if cached versions exist. The newly computed matrices will still be saved to cache for future use.

### Parallel Processing

For processing many files, use job arrays:

```bash
#!/bin/bash
#SBATCH --array=0-99

FILES=(run_*.root)
FILE=${FILES[$SLURM_ARRAY_TASK_ID]}

python interferometric_reco_example.py \
    --config config.yaml \
    --inputfile $FILE \
    --output_type hdf5
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
python interferometric_reco_example.py [OPTIONS]

Required:
  --config CONFIG              Path to YAML configuration file
  --inputfile FILE [FILE ...]  Input data file(s) (.root or .nur)

Optional:
  --output_type {hdf5,nur}     Output format (default: hdf5)
  --outputfile FILE            Manually specify output file path (overrides organized structure)
  --events N [N ...]           Specific event IDs/indices to process
  --runs N [N ...]             Specific run numbers to process (NUR files only)
  --sim_truth_fixed_coord      Use simulation truth for fixed coordinate (NUR files only)
  --ignore_cache               Force recalculation of delay matrices (ignore cached data)
  --save-maps                  Save correlation map data to pickle files
  --save-pair-maps             Save individual channel pair correlation maps
  --verbose                    Print reconstruction results for each event
```

---

## Contact & Support

For questions or issues:
- Review examples in this README
- Check NuRadioReco documentation
- Contact Bryan Hendricks (blh5615@psu.edu)

---

